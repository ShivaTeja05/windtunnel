"""Web API tests. Offline: research providers are faked (no network), the race uses MockClient."""

import json
import math
import os
import threading
import time

import pytest
from fastapi.testclient import TestClient

from windtunnel import jobs
from windtunnel.api import RateLimiter, create_app
from windtunnel.budget import BudgetGuard
from windtunnel.engine import run_wind_tunnel
from windtunnel.jobs import REPORT_VERSION, STAGES, Environment, JobStore
from windtunnel.llm import MockClient
from windtunnel.scout import Source


def fake_providers():
    def arxiv(query, limit, fetcher):
        return [
            Source("Self-Consistency Improves Chain of Thought Reasoning", "https://arxiv.org/abs/2203.11171", "paper", 2022,
                   authors=["Xuezhi Wang"], arxiv_id="2203.11171", abstract="Majority voting over sampled reasoning paths.",
                   found_via=["arxiv"], queries=[query]),
            Source("Self-Refine: Iterative Refinement with Self-Feedback", "https://arxiv.org/abs/2303.17651", "paper", 2023,
                   arxiv_id="2303.17651", abstract="The model critiques and verifies its own output.", found_via=["arxiv"],
                   queries=[query]),
        ]

    def openalex(query, limit, fetcher):
        return [Source(f"Fault tolerance via voting ({query})", "https://doi.org/10.1/x", "paper", 1995, doi=f"10.1/{query}",
                       citation_count=12, abstract="Redundant voting for fault tolerance.", found_via=["openalex"], queries=[query])]

    return {"arxiv": arxiv, "openalex": openalex}


def make_store(tmp_path):
    return JobStore(env_factory=lambda: Environment(client=None, providers=fake_providers(), fetcher=object()), reports_dir=tmp_path / "reports")


def _reject_constant(name):
    raise AssertionError(f"response contains {name}, which strict JSON forbids")


def strict(response):
    """Parse a response as strict JSON (no NaN or Infinity) and check it round-trips with allow_nan=False."""
    body = json.loads(response.text, parse_constant=_reject_constant)
    json.dumps(body, allow_nan=False)
    return body


@pytest.fixture
def client(tmp_path):
    app = create_app(make_store(tmp_path), rate_limit=3, warm_samples=False)
    with TestClient(app) as c:
        yield c


def wait_done(client, run_id, timeout=30.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        run = strict(client.get(f"/api/runs/{run_id}"))
        if run["status"] in ("done", "error"):
            return run
        time.sleep(0.05)
    raise AssertionError(f"run {run_id} did not finish")


EXAMPLES = [
    {"input": f"ticket {i}: {'refund please' if i % 2 else 'the app crashes'}", "expected": "refund" if i % 2 else "bug"}
    for i in range(12)
]


def test_health_reports_offline_mode(client):
    h = strict(client.get("/api/health"))
    assert h["mode"] == "offline" and h["race_provenance"] == "mock" and h["nemotron"] is False
    assert h["providers"] == ["arxiv", "openalex"]
    assert h["limits"]["runs_per_hour"] == 3 and "tavily" in h


def test_samples_and_families(client):
    samples = strict(client.get("/api/samples"))
    triage = next(s for s in samples if s["name"] == "ticket_triage")
    assert triage["n_examples"] == 40 and "billing" in triage["labels"]
    fams = strict(client.get("/api/families"))
    assert len(fams) == 6
    assert all(f["mermaid"].startswith("flowchart") and f["helps"] and f["hurts"] for f in fams)
    cascade = next(f for f in fams if f["family"] == "cascade")
    assert cascade["tiers"] == ["nano", "super"]  # the escalation tier comes from the graph, not just design.model
    assert "-.->" in cascade["mermaid"]  # the escalation branch is drawn dashed


def test_sample_run_end_to_end_is_mock_labelled_persisted_and_cached(client, tmp_path):
    r = client.post("/api/runs", json={"sample": "ticket_triage"})
    assert r.status_code == 202 and r.json()["cached"] is False
    run = wait_done(client, r.json()["id"])
    assert run["status"] == "done", run["error"]
    assert run["mode"] == "offline" and run["race_provenance"] == "mock" and run["report_version"] == REPORT_VERSION
    stages = {s["key"]: s["status"] for s in run["stages"]}
    assert list(stages) == STAGES
    assert stages == {
        "plain_llm_answer": "skipped",
        "research": "done",
        "design_candidates": "done",
        "race": "done",
        "breed": "done",
        "llm_pick": "skipped",
        "limitations": "done",
    }

    rep = run["report"]  # the engine's report, as run_wind_tunnel returns it
    assert rep["mode"] == "offline" and rep["provenance"] == "mock"
    assert rep["plain_answer"]["available"] is False
    assert rep["research"]["cards"] and rep["research"]["methods_map"]
    assert all(c["mermaid"].startswith("flowchart") for c in rep["candidates"])
    assert {c["role"] for c in rep["candidates"]} >= {"baseline", "llm-default"}
    assert rep["race"]["provenance"] == "mock" and rep["race"]["task"]["n_examples"] == 40
    assert all("outputs" not in d and d["tiers"] for d in rep["race"]["designs"])
    assert rep["verdict"]["tested"] and rep["verdict"]["provenance"] == "mock"
    assert rep["verdict"]["headline"].startswith("[MOCK")
    raced = {c["id"] for c in rep["candidates"]} | {b["id"] for b in rep["breeding"]["bred"]}
    assert rep["verdict"]["recommended_id"] in raced
    assert rep["advisor"]["available"] is False

    receipt = {e["stage"]: e for e in rep["model_receipt"]}
    assert receipt["race"]["produced_by"] == "mock" and receipt["race"]["cost_usd"] == 0.0
    assert receipt["race"]["simulated_cost_usd"] > 0
    assert rep["total_cost_usd"] == 0.0  # nothing real was spent
    assert any("MOCK" in line for line in rep["limitations"]) and any("OFFLINE" in line for line in rep["limitations"])
    assert isinstance(rep["warnings"], list)

    assert (tmp_path / "reports" / f"{run['id']}.json").is_file()
    again = client.post("/api/runs", json={"sample": "ticket_triage"}).json()
    assert again == {"id": run["id"], "cached": True, "status": "done", "url": f"/r/{run['id']}"}

    page = client.get(f"/r/{run['id']}")
    assert page.status_code == 200 and "text/html" in page.headers["content-type"]


def test_breeding_section_data(client):
    run = wait_done(client, client.post("/api/runs", json={"sample": "ticket_triage"}).json()["id"])
    b = run["report"]["breeding"]
    assert b["enabled"] and b["bred"] and b["diagnoses"][0]
    # Which losers were diagnosed, with their failure modes.
    for d in b["diagnoses"][0]:
        assert d["design_id"] and d["failure_modes"]
        assert all(m["mode"] in b["failure_mode_help"] for m in d["failure_modes"])
    # Which cross-domain mechanism was applied, and the bred design's diagram and tiers.
    for bred in b["bred"]:
        assert bred["mechanism"]["name"] and bred["mechanism"]["field"]
        assert bred["parent_id"] in {d["design_id"] for gen in b["diagnoses"] for d in gen}
        assert bred["mermaid"].startswith("flowchart") and bred["tiers"]
        assert b["lineage"][bred["id"]]["mechanism"] == bred["mechanism"]["name"]
    # Its score: the deciding race includes the bred designs that were raced.
    final = {d["id"]: d for d in run["report"]["race"]["designs"]}
    assert any(bred["id"] in final for bred in b["bred"])
    # 40 examples is too few to hold any out, so the selection-bias note must say so.
    assert b["holdout_ids"] == [] and b["selection_bias"].startswith("present") and "optimistic" in b["note"]
    assert any("optimistic" in line for line in run["report"]["limitations"])
    stage = next(s for s in run["stages"] if s["key"] == "breed")
    assert stage["status"] == "done" and "bred" in stage["detail"]


def test_breeding_can_be_turned_off(client):
    run_id = client.post("/api/runs", json={"problem": "Route support tickets to refund or bug queues.", "examples": EXAMPLES, "breed": False}).json()["id"]
    run = wait_done(client, run_id)
    assert run["status"] == "done", run["error"]
    assert run["report"]["breeding"]["enabled"] is False
    assert next(s for s in run["stages"] if s["key"] == "breed")["status"] == "skipped"
    assert client.post("/api/runs", json={"problem": "Route support tickets please.", "breed": "no"}).status_code == 422


def test_progress_streams_stage_status(tmp_path):
    """The engine's progress callback drives the live stages that GET /api/runs/{id} returns."""
    gate = threading.Event()

    def slow_arxiv(query, limit, fetcher):
        gate.wait(10)
        return fake_providers()["arxiv"](query, limit, fetcher)

    providers = {**fake_providers(), "arxiv": slow_arxiv}
    store = JobStore(env_factory=lambda: Environment(client=None, providers=providers), reports_dir=tmp_path / "reports")
    with TestClient(create_app(store, warm_samples=False)) as c:
        run_id = c.post("/api/runs", json={"problem": "Route support tickets to refund or bug queues.", "examples": EXAMPLES}).json()["id"]
        deadline = time.time() + 10
        while time.time() < deadline:
            run = c.get(f"/api/runs/{run_id}").json()
            stages = {s["key"]: s["status"] for s in run["stages"]}
            if stages["research"] == "running":
                break
            time.sleep(0.02)
        assert stages["plain_llm_answer"] == "skipped" and stages["research"] == "running"
        assert stages["race"] == "pending" and run["report"] is None and run["status"] == "running"
        gate.set()
        run = wait_done(c, run_id)
        assert run["status"] == "done" and all(s["status"] in ("done", "skipped") for s in run["stages"])


class CountingMock:
    """A stand-in for a real client: counts calls, never touches the network."""

    def __init__(self, task):
        self.mock = MockClient.for_task(task)
        self.calls = 0

    def complete(self, model, messages, **kw):
        self.calls += 1
        return self.mock.complete(model, messages, **kw)


def test_engine_failure_marks_the_stage_and_keeps_the_report_page(tmp_path, monkeypatch):
    def broken(*args, **kwargs):
        progress = kwargs["progress"]
        progress("plain_llm_answer", "running")
        progress("plain_llm_answer", "skipped")
        progress("research", "running")
        progress("research", "failed")
        raise RuntimeError("search exploded")

    monkeypatch.setattr(jobs, "run_wind_tunnel", broken)
    with TestClient(create_app(make_store(tmp_path), warm_samples=False)) as c:
        run = wait_done(c, c.post("/api/runs", json={"problem": "Summarise long legal contracts for a founder."}).json()["id"])
    assert run["status"] == "error" and "search exploded" in run["error"] and run["report"] is None
    stages = {s["key"]: s["status"] for s in run["stages"]}
    assert stages["research"] == "error" and stages["race"] == "skipped" and stages["plain_llm_answer"] == "skipped"


def test_unmeasured_designs_never_produce_nan(tmp_path):
    """A tier whose every call fails for infrastructure reasons leaves designs with no accuracy (NaN in the engine)."""

    class FlakyUltra(MockClient):
        def complete(self, model, messages, **kw):
            if model == "ultra":
                raise ConnectionError("Token Factory unreachable")
            return super().complete(model, messages, **kw)

    task = jobs.build_task("Route support tickets to refund or bug queues.", EXAMPLES, None, None)
    report = jobs.web_report(
        run_wind_tunnel("Route support tickets to refund or bug queues.", task, FlakyUltra.for_task(task), providers=fake_providers(), fetcher=object())
    )
    unmeasured = [d for d in report["race"]["designs"] if d["n"] == 0]
    assert unmeasured and all(d["accuracy"] is None and d["accuracy_ci"] == [None, None] for d in unmeasured)
    assert all(d["unmeasured"] for d in unmeasured)
    json.dumps(report, allow_nan=False)
    # Even a report with raw NaN (e.g. one written by older code) is served as strict JSON.
    store = make_store(tmp_path)
    run = jobs.Run(id="nanrun01", request={"problem": "p", "sample": None, "examples": [], "labels": None}, mode="offline", race_provenance="mock")
    run.status, run.report = "done", {"race": {"designs": [{"accuracy": math.nan, "p50_latency_s": math.inf}]}}
    store._save(run)
    with TestClient(create_app(store, warm_samples=False)) as c:
        got = strict(c.get("/api/runs/nanrun01"))
    assert got["report"]["race"]["designs"][0] == {"accuracy": None, "p50_latency_s": None}


def test_real_client_is_counted_live_and_wrapped(tmp_path):
    task = jobs.build_task("x" * 20, EXAMPLES, None, None)
    inner = CountingMock(task)
    guard = BudgetGuard(inner, limit_usd=5.0, ledger=tmp_path / "spend.json")
    store = JobStore(env_factory=lambda: Environment(client=guard, providers=fake_providers()), reports_dir=tmp_path / "reports")
    with TestClient(create_app(store, warm_samples=False)) as c:
        assert c.get("/api/health").json()["budget"]["limit_usd"] == 5.0
        run = wait_done(c, c.post("/api/runs", json={"problem": "Route support tickets to refund or bug queues.", "examples": EXAMPLES}).json()["id"])
    assert run["status"] == "done", run["error"]
    assert run["mode"] == "nemotron"  # a non-mock client is treated as real (here it only answers like the mock)
    # Every model call went through the BudgetGuard and was counted for the live view.
    assert inner.calls > 0 and run["live"]["calls"] == inner.calls


def test_stale_sample_runs_are_kept_as_share_pages_but_never_replayed(tmp_path):
    reports = tmp_path / "reports"
    reports.mkdir()
    (reports / "sample-ticket_triage-offline.json").write_text("{}")  # what the old pipeline wrote
    (reports / "sample-ticket_triage-offline-0123456789.json").write_text("{}")  # another engine version
    (reports / "userrun1.json").write_text("{}")  # a user's share link is kept
    store = make_store(tmp_path)
    # A redeploy must not break share links: nothing is deleted at startup.
    assert sorted(p.name for p in reports.iterdir()) == [
        "sample-ticket_triage-offline-0123456789.json",
        "sample-ticket_triage-offline.json",
        "userrun1.json",
    ]
    key = store._cache_key(jobs.validate_request({"sample": "ticket_triage"}))
    assert key.startswith("sample-ticket_triage-offline-") and key.endswith(jobs.code_fingerprint(store.tasks_dir, "ticket_triage"))
    # A cached run from another report version is not replayed.
    old = jobs.Run(id=key, request=jobs.validate_request({"sample": "ticket_triage"}), mode="offline", race_provenance="mock")
    old.status, old.report_version = "done", REPORT_VERSION - 1
    store._save(old)
    assert store.cached(old.request) is None
    # A run under another fingerprint is still served as its own page, just never as a cache hit.
    stale = jobs.Run(id="sample-ticket_triage-offline-0123456789", request=old.request, mode="offline", race_provenance="mock")
    stale.status = "done"
    store._save(stale)
    assert store.get(stale.id).status == "done" and store.cached(old.request) is None


def test_legacy_report_still_loads(client, tmp_path):
    legacy = {
        "id": "oldrun01", "status": "done", "mode": "offline", "race_provenance": "mock", "created_at": "2026-10-27T00:00:00+00:00",
        "finished_at": None, "request": {"problem": "p", "sample": None, "examples": [], "labels": None},
        "stages": [{"key": "plain_llm", "title": "Ask a plain LLM", "status": "skipped", "detail": ""}],
        "results": {}, "receipt": [], "limitations": [], "error": None, "cached_from": None,
    }
    (tmp_path / "reports").mkdir(exist_ok=True)
    (tmp_path / "reports" / "oldrun01.json").write_text(json.dumps(legacy))
    run = strict(client.get("/api/runs/oldrun01"))
    assert run["report_version"] is None and run["report"] is None and run["status"] == "done"


def test_report_survives_restart(tmp_path):
    with TestClient(create_app(make_store(tmp_path), warm_samples=False)) as c1:
        run_id = c1.post("/api/runs", json={"problem": "Route support tickets to refund or bug queues.", "examples": EXAMPLES}).json()["id"]
        wait_done(c1, run_id)
    with TestClient(create_app(make_store(tmp_path), warm_samples=False)) as c2:
        run = strict(c2.get(f"/api/runs/{run_id}"))
        assert run["status"] == "done" and run["report"]["race"]["task"]["n_examples"] == 12
        assert c2.get("/api/runs/doesnotexist").status_code == 404


def test_problem_without_examples_skips_the_race(client):
    run_id = client.post("/api/runs", json={"problem": "Summarise long legal contracts for a busy founder."}).json()["id"]
    run = wait_done(client, run_id)
    assert run["status"] == "done"
    stages = {s["key"]: s["status"] for s in run["stages"]}
    assert stages["race"] == "skipped" and stages["breed"] == "skipped" and run["report"]["race"] is None
    assert run["race_provenance"] == "untested" and run["report"]["breeding"] is None
    assert run["report"]["candidates"] and run["report"]["verdict"]["tested"] is False


@pytest.mark.parametrize(
    "body, fragment",
    [
        ({"problem": "too short"}, "at least"),
        ({"problem": "x" * 2001}, "too long"),
        ({"problem": "Route tickets to the right queue.", "examples": EXAMPLES[:3]}, "at least 10"),
        ({"problem": "Route tickets to the right queue.", "examples": [{"input": f"t{i}", "expected": "a"} for i in range(201)]}, "too many"),
        ({"problem": "Route tickets to the right queue.", "examples": [{"input": "", "expected": "a"}] * 10}, "non-empty"),
        ({"sample": "../etc"}, "unknown sample"),
        ({"sample": "ticket_triage", "examples": EXAMPLES}, "not both"),
    ],
)
def test_input_limits(client, body, fragment):
    r = client.post("/api/runs", json=body)
    assert r.status_code == 422 and fragment in r.json()["detail"]


def test_rate_limit_counts_fresh_runs_only(client):
    sample = client.post("/api/runs", json={"sample": "ticket_triage"}).json()
    wait_done(client, sample["id"])
    for _ in range(5):  # cached sample hits are free
        assert client.post("/api/runs", json={"sample": "ticket_triage"}).json()["cached"] is True
    for i in range(2):
        assert client.post("/api/runs", json={"problem": f"Classify support emails, variant {i}."}).status_code == 202
    r = client.post("/api/runs", json={"problem": "Classify support emails, one more."})
    assert r.status_code == 429 and "Rate limit" in r.json()["detail"]
    spoofed = client.post("/api/runs", json={"problem": "Classify support emails, spoofed."}, headers={"X-Forwarded-For": "10.0.0.9"})
    assert spoofed.status_code == 429  # forwarded headers are ignored unless the app runs behind a trusted proxy


def test_rate_limit_behind_trusted_proxy_uses_last_hop(tmp_path):
    with TestClient(create_app(make_store(tmp_path), rate_limit=1, warm_samples=False, trust_proxy=True)) as c:
        body = {"problem": "Classify support emails for a shop."}
        assert c.post("/api/runs", json=body, headers={"X-Forwarded-For": "1.1.1.1"}).status_code == 202
        # A forged first hop doesn't help: the proxy-appended last hop is still 1.1.1.1.
        assert c.post("/api/runs", json=body, headers={"X-Forwarded-For": "9.9.9.9, 1.1.1.1"}).status_code == 429
        assert c.post("/api/runs", json=body, headers={"X-Forwarded-For": "2.2.2.2"}).status_code == 202


def test_rate_limiter_window():
    rl = RateLimiter(2, window_s=0.2)
    assert rl.check("a") is None and rl.check("a") is None
    assert rl.check("a") > 0
    time.sleep(0.25)
    assert rl.check("a") is None


def test_frontend_routes_serve_the_app(client):
    for path in ("/", "/explore", "/editor", "/new"):
        r = client.get(path)
        assert r.status_code == 200 and "Wind Tunnel" in r.text
    assert client.get("/static/app.js").status_code == 200
    assert "mermaid@10.9.1" in client.get("/static/app.js").text


# ---------------------------------------------------------------- abuse limits


def test_dockerfile_does_not_trust_forwarded_headers_by_default():
    dockerfile = (jobs.PACKAGE_DIR.parent / "Dockerfile").read_text()
    assert "WINDTUNNEL_TRUST_PROXY=0" in dockerfile and "WINDTUNNEL_TRUST_PROXY=1 \\" not in dockerfile


def test_forged_forwarded_for_cannot_skip_the_rate_limit_without_a_trusted_proxy(tmp_path, monkeypatch):
    # TRUST_PROXY on, but the request does not come from a listed proxy: the header is ignored.
    monkeypatch.setenv("WINDTUNNEL_TRUSTED_PROXIES", "10.0.0.0/8")
    with TestClient(create_app(make_store(tmp_path), rate_limit=2, warm_samples=False, trust_proxy=True)) as c:
        codes = [
            c.post("/api/runs", json={"problem": f"Classify support emails, try {i}."}, headers={"X-Forwarded-For": f"9.9.9.{i}"}).status_code
            for i in range(5)
        ]
    assert codes == [202, 202, 429, 429, 429]


def test_client_ip_hops_and_ipv6_buckets():
    from types import SimpleNamespace

    from windtunnel.api import _networks, client_ip

    def req(peer, xff=None):
        return SimpleNamespace(client=SimpleNamespace(host=peer), headers={"x-forwarded-for": xff} if xff else {})

    assert client_ip(req("1.2.3.4", "6.6.6.6"), False) == "1.2.3.4"  # not behind a proxy: the header is the client's
    assert client_ip(req("10.0.0.2", "6.6.6.6, 5.5.5.5, 10.0.0.1"), True, hops=2) == "5.5.5.5"
    trusted = _networks("10.0.0.0/8")
    assert client_ip(req("10.0.0.2", "5.5.5.5"), True, trusted_proxies=trusted) == "5.5.5.5"
    assert client_ip(req("8.8.8.8", "5.5.5.5"), True, trusted_proxies=trusted) == "8.8.8.8"
    # Every address in one IPv6 /64 shares a bucket, so rotating addresses doesn't help.
    assert client_ip(req("2001:db8::1"), False) == client_ip(req("2001:db8::ffff:1234"), False) == "2001:db8::/64"


def test_rate_limiter_forgets_expired_clients():
    rl = RateLimiter(1, window_s=0.05)
    rl.SWEEP_EVERY_S = 0.0
    for i in range(50):
        rl.check(f"10.0.{i}.1")
    time.sleep(0.1)
    rl.check("fresh")
    assert list(rl._hits) == ["fresh"]


def test_oversized_body_is_refused_before_parsing(client):
    big = {"problem": "Route tickets to the right queue.", "examples": [{"input": "x" * 1000, "expected": "a"}] * 1500}
    assert client.post("/api/runs", json=big).status_code == 413

    def chunks():  # no Content-Length: the streamed body is cut off past the cap
        yield b'{"problem": "'
        for _ in range(200):
            yield b"x" * 10_000
        yield b'"}'

    assert client.post("/api/runs", content=chunks(), headers={"Content-Type": "application/json"}).status_code == 413
    assert client.post("/api/runs", content=b"not json", headers={"Content-Type": "application/json"}).status_code == 400


def gated_store(tmp_path, gate, **kw):
    def slow_arxiv(query, limit, fetcher):
        gate.wait(10)
        return fake_providers()["arxiv"](query, limit, fetcher)

    providers = {**fake_providers(), "arxiv": slow_arxiv}
    return JobStore(env_factory=lambda: Environment(client=None, providers=providers), reports_dir=tmp_path / "reports", **kw)


def test_too_many_runs_in_flight_are_refused_not_queued(tmp_path):
    gate = threading.Event()
    store = gated_store(tmp_path, gate, max_in_flight=1)
    try:
        with TestClient(create_app(store, rate_limit=10, warm_samples=False)) as c:
            first = c.post("/api/runs", json={"problem": "Classify support emails for a shop."})
            second = c.post("/api/runs", json={"problem": "Classify support emails for a bank."})
            assert first.status_code == 202
            assert second.status_code == 503 and "busy" in second.json()["detail"]
            gate.set()
            wait_done(c, first.json()["id"])
            assert c.post("/api/runs", json={"problem": "Classify support emails for a bank."}).status_code == 202
    finally:
        gate.set()


def test_run_interrupted_by_a_restart_is_reported_not_lost(tmp_path):
    gate = threading.Event()
    store = gated_store(tmp_path, gate)
    try:
        run, _ = store.submit(jobs.validate_request({"problem": "Classify support emails for a shop.", "examples": EXAMPLES}))
        # Saved as soon as it is queued, with the user's examples.
        on_disk = json.loads((tmp_path / "reports" / f"{run.id}.json").read_text())
        assert on_disk["status"] in ("queued", "running") and len(on_disk["request"]["examples"]) == len(EXAMPLES)
        restarted = make_store(tmp_path)  # a new process: the run is not in its memory
        got = restarted.get(run.id)
        assert got.status == "error" and "restarted" in got.error
        assert all(s["status"] in ("error", "skipped", "done") for s in got.stages.values())
    finally:
        gate.set()


def test_weak_sample_run_is_kept_but_never_replayed(tmp_path):
    """A sample run whose search found nothing (e.g. arXiv rate limited at warm-up) is not served to everyone."""
    store = JobStore(env_factory=lambda: Environment(client=None, providers={}), reports_dir=tmp_path / "reports")
    with TestClient(create_app(store, warm_samples=False)) as c:
        first = c.post("/api/runs", json={"sample": "ticket_triage"}).json()
        run = wait_done(c, first["id"])
        assert run["status"] == "done" and run["cacheable"] is False and not run["report"]["research"]["cards"]
        again = c.post("/api/runs", json={"sample": "ticket_triage"}).json()
        assert again["cached"] is False and again["id"] == first["id"]  # re-run under the same share link
        wait_done(c, again["id"])

    budget_cut = {"research": {"cards": [1]}, "race": {"aborted": "spending cap reached"}, "verdict": {"aborted": "x"}}
    assert jobs.weak_report(budget_cut) == "the race stopped early"
    plain_cut = {"mode": "nemotron", "research": {"cards": [1]}, "model_receipt": [{"stage": "plain_llm_answer", "detail": "budget exhausted"}]}
    assert "spending cap" in jobs.weak_report(plain_cut)


def test_long_sample_names_keep_their_cache_across_restarts(tmp_path):
    import shutil

    tasks = tmp_path / "tasks"
    name = "customer_support_ticket_triage_long_sample_name"
    shutil.copytree(jobs.TASKS_DIR / "ticket_triage", tasks / name)

    def store():
        return JobStore(env_factory=lambda: Environment(client=None, providers=fake_providers(), fetcher=object()),
                        reports_dir=tmp_path / "reports", tasks_dir=tasks)

    s1 = store()
    task = jobs.load_task(tasks / name)  # validate_request only knows the repo's samples, so build the request here
    req = {"problem": jobs.sample_problem(task), "sample": name, "examples": [], "labels": None, "breed": True}
    run, cached = s1.submit(req)
    assert not cached and name in run.id
    deadline = time.time() + 30
    while s1.get(run.id).status not in ("done", "error") and time.time() < deadline:
        time.sleep(0.05)
    assert s1.get(run.id).status == "done"
    s2 = store()  # a restart
    assert (tmp_path / "reports" / f"{run.id}.json").is_file()
    hit = s2.cached(req)
    assert hit is not None and hit.id == run.id


def test_old_user_reports_are_pruned_but_samples_kept(tmp_path):
    reports = tmp_path / "reports"
    reports.mkdir()
    for i in range(5):
        path = reports / f"userrun{i}.json"
        path.write_text("{}")
        mtime = time.time() - 100 + i
        os.utime(path, (mtime, mtime))
    (reports / "sample-ticket_triage-offline-0123456789.json").write_text("{}")
    store = JobStore(env_factory=lambda: Environment(client=None, providers=fake_providers()), reports_dir=reports, max_reports=2)
    assert store.max_reports == 2
    assert sorted(p.name for p in reports.iterdir()) == ["sample-ticket_triage-offline-0123456789.json", "userrun3.json", "userrun4.json"]


def test_each_web_run_has_its_own_spending_cap(tmp_path):
    from windtunnel.budget import BudgetExceeded

    task = jobs.build_task("x" * 20, EXAMPLES, None, None)
    counted = []
    capped = jobs.CountingClient(CountingMock(task), counted.append, limit_usd=0.001)
    with pytest.raises(BudgetExceeded, match="this run's spending cap"):
        for _ in range(100):
            capped.complete("ultra", [{"role": "user", "content": "ticket 1: refund please"}], max_tokens=1024)
    assert capped.spent_usd <= 0.001

    # End to end: a visitor's run stops at its own cap; the shared ledger keeps the rest for everyone else.
    guard = BudgetGuard(CountingMock(task), limit_usd=5.0, ledger=tmp_path / "spend.json")
    store = JobStore(env_factory=lambda: Environment(client=guard, providers=fake_providers()), reports_dir=tmp_path / "reports",
                     run_budget_usd=0.0001)
    with TestClient(create_app(store, warm_samples=False)) as c:
        run = wait_done(c, c.post("/api/runs", json={"problem": "Route support tickets to refund or bug queues.", "examples": EXAMPLES}).json()["id"])
    assert run["status"] == "done", run["error"]
    assert run["live"]["cost_usd"] <= 0.0001 and guard.spent_usd <= 0.0001
    assert run["cacheable"] is False
