"""Wind Tunnel web app: the JSON API and the static frontend, served from one FastAPI app.

    uv run uvicorn windtunnel.api:app

Without NEBIUS_API_KEY the app runs in offline mode: real paper search with heuristic
method cards, a simulated ("mock") race and no plain-LLM answer. Every response and
page says which mode produced it.
"""

from __future__ import annotations

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
from .graph import graph_for
from .jobs import LIMITS, InvalidRun, JobStore, list_samples, mermaid_for, validate_request
from .models import CATALOG

WEB_DIR = Path(__file__).resolve().parent.parent / "web"

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
        "of the five, so it has to earn that with accuracy.",
    },
]


class RateLimiter:
    """Sliding-window limit on new runs per client IP."""

    def __init__(self, max_runs: int, window_s: float = 3600.0):
        self.max_runs = max_runs
        self.window_s = window_s
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def check(self, key: str) -> float | None:
        """Record a hit. Returns None if allowed, else seconds until the next slot frees up."""
        now = time.monotonic()
        with self._lock:
            hits = self._hits[key]
            while hits and now - hits[0] > self.window_s:
                hits.popleft()
            if len(hits) >= self.max_runs:
                return self.window_s - (now - hits[0])
            hits.append(now)
            return None


def client_ip(request: Request, trust_proxy: bool) -> str:
    # Behind one reverse proxy, the last X-Forwarded-For hop is the address the proxy saw.
    # Earlier hops come from the client and can be forged, so they are ignored.
    forwarded = request.headers.get("x-forwarded-for") if trust_proxy else None
    if forwarded:
        return forwarded.split(",")[-1].strip()
    return request.client.host if request.client else "unknown"


def create_app(
    store: JobStore | None = None,
    *,
    rate_limit: int | None = None,
    warm_samples: bool | None = None,
    trust_proxy: bool | None = None,
) -> FastAPI:
    load_dotenv()
    store = store or JobStore()
    limiter = RateLimiter(rate_limit if rate_limit is not None else int(os.environ.get("WINDTUNNEL_RUNS_PER_HOUR", "5")))
    warm = warm_samples if warm_samples is not None else os.environ.get("WINDTUNNEL_WARM_SAMPLES", "1") == "1"
    proxied = trust_proxy if trust_proxy is not None else os.environ.get("WINDTUNNEL_TRUST_PROXY") == "1"

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        if warm:  # fill the sample cache in the background so the first click is instant
            threading.Thread(target=store.warm_samples, daemon=True, name="windtunnel-warm").start()
        yield

    app = FastAPI(title="Wind Tunnel", version="0.1", docs_url="/api/docs", openapi_url="/api/openapi.json", lifespan=lifespan)
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
            "limits": {**LIMITS, "runs_per_hour": limiter.max_runs},
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
                    "tiers": g.models(),
                }
            )
        return out

    @app.post("/api/runs", status_code=202)
    async def create_run(request: Request) -> JSONResponse:
        try:
            body = await request.json()
        except Exception:
            raise HTTPException(400, "body must be JSON") from None
        if not isinstance(body, dict):
            raise HTTPException(400, "body must be a JSON object")
        try:
            req = validate_request(body)
        except InvalidRun as exc:
            raise HTTPException(422, str(exc)) from None
        hit = store.cached(req)
        if hit is None:  # only fresh runs count against the limit; cached samples are free
            wait = limiter.check(client_ip(request, proxied))
            if wait is not None:
                raise HTTPException(
                    429,
                    f"Rate limit: {limiter.max_runs} new runs per hour on this public demo. "
                    f"Try again in {int(wait // 60) + 1} min, or try a sample (cached, instant).",
                )
        run, cached = (hit, True) if hit else store.submit(req, use_cache=False)
        return JSONResponse({"id": run.id, "cached": cached, "status": run.status, "url": f"/r/{run.id}"}, status_code=202)

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
