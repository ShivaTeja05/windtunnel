"""Wind Tunnel web app: the JSON API and the static frontend, served from one FastAPI app.

    uv run uvicorn windtunnel.api:app

Without NEBIUS_API_KEY the app runs in offline mode: real paper search with heuristic
method cards, a simulated ("mock") race and no plain-LLM answer. Every response and
page says which mode produced it.
"""

from __future__ import annotations

import ipaddress
import json
import os
import threading
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .cli import load_dotenv
from .designs import FAMILY_SUMMARIES, Design
from .engine import mermaid_for
from .graph import graph_for
from .jobs import LIMITS, InvalidRun, JobStore, ServerBusy, list_samples, strict_dumps, tiers_for, validate_request
from .models import CATALOG

WEB_DIR = Path(__file__).resolve().parent.parent / "web"

# The largest valid body is about 200 * (2000 + 240) chars of examples plus a 2000-char problem
# (~0.5 MB, more with JSON escapes); anything past this is refused before it is parsed.
MAX_BODY_BYTES = 1_000_000

# The Explore gallery: one representative design per family.
FAMILY_GALLERY = [
    {
        "family": "single_call",
        "name": "Single call",
        "design": Design("single-call-super", "single_call", "super"),
        "helps": "Well-defined tasks with a short answer: classification, routing, extraction. Cheapest and fastest, "
        "and often tied with far bigger designs.",
        "hurts": "Multi-step reasoning, or when one wrong answer is costly and nothing checks it.",
    },
    {
        "family": "self_check",
        "name": "Answer + self-check",
        "design": Design("self-check-nano", "self_check", "nano"),
        "helps": "When errors are easy to spot once written down (format slips, arithmetic, missed constraints). "
        "A small model checking itself can match a bigger model answering once.",
        "hurts": "When the checker shares the answerer's blind spots: it confidently approves the same mistake, "
        "and you pay for two calls.",
    },
    {
        "family": "self_consistency",
        "name": "Self-consistency (majority vote)",
        "design": Design("self-consistency-nano", "self_consistency", "nano", {"n": 5, "temperature": 0.7}),
        "helps": "Tasks with one correct answer where the model is right more often than not. Voting averages "
        "away random slips.",
        "hurts": "Open-ended outputs that can't be voted on, and systematic errors: five wrong answers that agree "
        "are still wrong. Cost grows with n.",
    },
    {
        "family": "plan_execute",
        "name": "Plan, then execute",
        "design": Design("plan-execute-super", "plan_execute", "super"),
        "helps": "Multi-step problems where laying out the steps first prevents skipped ones: math word problems, "
        "procedures, structured writing.",
        "hurts": "Simple lookups and labels. The plan adds latency and cost and gives the model room to talk itself "
        "out of the obvious answer.",
    },
    {
        "family": "orchestrator_workers",
        "name": "Orchestrator + workers",
        "design": Design("crew-ultra", "orchestrator_workers", "ultra", {"worker_model": "super", "max_subtasks": 3}),
        "helps": "Inputs that really split into independent parts (long documents, multi-part questions, research). "
        "Workers run in parallel.",
        "hurts": "Short, atomic inputs. It is the design LLMs recommend most, and it costs the most per request "
        "of these families, so it has to earn that with accuracy.",
    },
    {
        "family": "cascade",
        "name": "Cascade (escalate on disagreement)",
        "design": Design("cascade-nano", "cascade", "nano", {"cheap_model": "nano", "escalate_model": "super", "samples": 2}),
        "helps": "Mostly easy inputs with a hard tail. The cheap model answers twice; only inputs where its answers "
        "disagree pay for the strong model (the dashed branch).",
        "hurts": "When the cheap model is confidently and consistently wrong: two matching wrong answers never escalate.",
    },
]


class StrictJSONResponse(JSONResponse):
    """JSON that strict parsers accept: NaN and infinity are sent as null, never as bare NaN."""

    def render(self, content) -> bytes:
        return strict_dumps(content, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


class RateLimiter:
    """Sliding-window limit on new runs per client IP."""

    SWEEP_EVERY_S = 60.0

    def __init__(self, max_runs: int, window_s: float = 3600.0):
        self.max_runs = max_runs
        self.window_s = window_s
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()
        self._last_sweep = time.monotonic()

    def _sweep(self, now: float) -> None:
        """Drop keys whose hits have all expired, so one-off clients don't pile up in memory."""
        for key in [k for k, hits in self._hits.items() if not hits or now - hits[-1] > self.window_s]:
            del self._hits[key]
        self._last_sweep = now

    def check(self, key: str) -> float | None:
        """Record a hit. Returns None if allowed, else seconds until the next slot frees up."""
        now = time.monotonic()
        with self._lock:
            if now - self._last_sweep > self.SWEEP_EVERY_S:
                self._sweep(now)
            hits = self._hits[key]
            while hits and now - hits[0] > self.window_s:
                hits.popleft()
            if len(hits) >= self.max_runs:
                return self.window_s - (now - hits[0])
            hits.append(now)
            return None


def _networks(spec: str) -> list[ipaddress.IPv4Network | ipaddress.IPv6Network]:
    return [ipaddress.ip_network(part.strip(), strict=False) for part in spec.split(",") if part.strip()]


def _in(host: str, networks) -> bool:
    try:
        addr = ipaddress.ip_address(host)
    except ValueError:
        return False
    return any(addr in net for net in networks)


def _bucket(host: str) -> str:
    """The rate-limit key for an address: IPv6 clients usually own a whole /64, so they share one bucket."""
    try:
        addr = ipaddress.ip_address(host)
    except ValueError:
        return host[:64]
    if isinstance(addr, ipaddress.IPv6Address):
        if addr.ipv4_mapped is not None:
            return str(addr.ipv4_mapped)
        return str(ipaddress.ip_network(f"{addr}/64", strict=False))
    return str(addr)


def client_ip(request: Request, trust_proxy: bool, *, hops: int = 1, trusted_proxies=()) -> str:
    """The client's address for rate limiting.

    X-Forwarded-For is read only when the app is told it runs behind a reverse proxy
    (WINDTUNNEL_TRUST_PROXY=1) and, if WINDTUNNEL_TRUSTED_PROXIES lists the proxies' addresses,
    only when the request really came from one of them. Each proxy appends the address it saw,
    so behind `hops` proxies the client is the hops-th entry from the end; earlier entries come
    from the client and can be forged, so they are ignored.
    """
    peer = request.client.host if request.client else "unknown"
    forwarded = request.headers.get("x-forwarded-for") if trust_proxy else None
    if forwarded and (not trusted_proxies or _in(peer, trusted_proxies)):
        parts = [p.strip() for p in forwarded.split(",") if p.strip()]
        if parts:
            return _bucket(parts[-hops] if len(parts) >= hops else parts[0])
    return _bucket(peer)


async def read_json_body(request: Request, max_bytes: int = MAX_BODY_BYTES):
    """The request body as JSON, refusing (413) anything over max_bytes before it is buffered whole."""
    declared = request.headers.get("content-length")
    if declared is not None:
        try:
            too_big = int(declared) > max_bytes
        except ValueError:
            raise HTTPException(400, "bad Content-Length") from None
        if too_big:
            raise HTTPException(413, f"request body is too large (max {max_bytes // 1000} KB)")
    chunks, size = [], 0
    async for chunk in request.stream():  # covers chunked uploads that declare no length
        size += len(chunk)
        if size > max_bytes:
            raise HTTPException(413, f"request body is too large (max {max_bytes // 1000} KB)")
        chunks.append(chunk)
    try:
        return json.loads(b"".join(chunks))
    except (ValueError, UnicodeDecodeError):
        raise HTTPException(400, "body must be JSON") from None


def create_app(
    store: JobStore | None = None,
    *,
    rate_limit: int | None = None,
    warm_samples: bool | None = None,
    trust_proxy: bool | None = None,
    daily_limit: int | None = None,
) -> FastAPI:
    load_dotenv()
    store = store or JobStore()
    limiter = RateLimiter(rate_limit if rate_limit is not None else int(os.environ.get("WINDTUNNEL_RUNS_PER_HOUR", "5")))
    daily = RateLimiter(daily_limit if daily_limit is not None else int(os.environ.get("WINDTUNNEL_RUNS_PER_DAY", "10")), 86400.0)
    warm = warm_samples if warm_samples is not None else os.environ.get("WINDTUNNEL_WARM_SAMPLES", "1") == "1"
    # Off unless the app really runs behind a reverse proxy: without one, clients write X-Forwarded-For themselves.
    proxied = trust_proxy if trust_proxy is not None else os.environ.get("WINDTUNNEL_TRUST_PROXY") == "1"
    proxy_hops = max(1, int(os.environ.get("WINDTUNNEL_PROXY_HOPS", "1")))
    trusted_proxies = _networks(os.environ.get("WINDTUNNEL_TRUSTED_PROXIES", ""))

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        if warm:  # fill the sample cache in the background so the first click is instant
            threading.Thread(target=store.warm_samples, daemon=True, name="windtunnel-warm").start()
        yield

    app = FastAPI(
        title="Wind Tunnel",
        version="0.2",
        docs_url="/api/docs",
        openapi_url="/api/openapi.json",
        lifespan=lifespan,
        default_response_class=StrictJSONResponse,
    )
    app.state.store = store
    app.state.limiter = limiter

    @app.get("/api/health")
    def health() -> dict:
        env = store.env
        client = env.client
        budget = None
        if client is not None and hasattr(client, "limit_usd"):
            budget = {"spent_usd": round(client.spent_usd, 4), "limit_usd": client.limit_usd}
        return {
            "ok": True,
            "mode": env.mode,
            "race_provenance": env.race_provenance,
            "nemotron": client is not None,
            "tavily": bool(os.environ.get("TAVILY_API_KEY")),
            "providers": list(env.providers),
            "models": {k: m.id for k, m in CATALOG.items()},
            "budget": budget,
            "limits": {**LIMITS, "runs_per_hour": limiter.max_runs, "runs_per_day": daily.max_runs, "run_budget_usd": store.run_budget_usd},
        }

    @app.get("/api/samples")
    def samples() -> list[dict]:
        return list_samples(store.tasks_dir)

    @app.get("/api/families")
    def families() -> list[dict]:
        out = []
        for f in FAMILY_GALLERY:
            d: Design = f["design"]
            g = graph_for(d)
            out.append(
                {
                    "family": f["family"],
                    "name": f["name"],
                    "summary": FAMILY_SUMMARIES[f["family"]],
                    "example_design": d.id,
                    "helps": f["helps"],
                    "hurts": f["hurts"],
                    "mermaid": mermaid_for(g, f["family"]),
                    "tiers": tiers_for({"id": d.id, "family": d.family, "model": d.model, "params": d.params}),
                }
            )
        return out

    @app.post("/api/runs", status_code=202)
    async def create_run(request: Request) -> JSONResponse:
        body = await read_json_body(request)
        if not isinstance(body, dict):
            raise HTTPException(400, "body must be a JSON object")
        try:
            req = validate_request(body)
        except InvalidRun as exc:
            raise HTTPException(422, str(exc)) from None
        hit = store.cached(req)
        if hit is None:  # only fresh runs count against the limit; cached samples are free
            if store.busy():
                raise HTTPException(503, store.busy_message())
            ip = client_ip(request, proxied, hops=proxy_hops, trusted_proxies=trusted_proxies)
            wait = limiter.check(ip)
            if wait is not None:
                raise HTTPException(
                    429,
                    f"Rate limit: {limiter.max_runs} new runs per hour on this public demo. "
                    f"Try again in {int(wait // 60) + 1} min, or try a sample (cached, instant).",
                )
            wait = daily.check(ip)
            if wait is not None:
                raise HTTPException(
                    429,
                    f"Rate limit: {daily.max_runs} new runs per day on this public demo. "
                    f"Try again in {int(wait // 3600) + 1} h, or try a sample (cached, instant).",
                )
        try:
            run, cached = (hit, True) if hit else store.submit(req, use_cache=False)
        except ServerBusy as exc:
            raise HTTPException(503, str(exc)) from None
        return StrictJSONResponse({"id": run.id, "cached": cached, "status": run.status, "url": f"/r/{run.id}"}, status_code=202)

    @app.get("/api/runs/{run_id}")
    def get_run(run_id: str) -> dict:
        run = store.get(run_id)
        if run is None:
            raise HTTPException(404, "no such run")
        return store.snapshot(run)

    # ------------------------------------------------------------ frontend

    index = WEB_DIR / "index.html"

    def spa() -> FileResponse:
        return FileResponse(index, headers={"Cache-Control": "no-cache"})

    for path in ("/", "/explore", "/editor", "/new"):
        app.add_api_route(path, spa, methods=["GET"], include_in_schema=False)

    @app.get("/r/{run_id}", include_in_schema=False)
    def share(run_id: str) -> FileResponse:
        return spa()

    if (WEB_DIR / "static").is_dir():
        app.mount("/static", StaticFiles(directory=WEB_DIR / "static"), name="static")

    return app


app = create_app()
