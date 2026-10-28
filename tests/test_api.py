"""Web API tests. Offline: research providers are faked, the race uses MockClient."""

import time

import pytest
from fastapi.testclient import TestClient

from windtunnel.api import RateLimiter, create_app
from windtunnel.jobs import Environment, JobStore
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
    return JobStore(env_factory=lambda: Environment(client=None, providers=fake_providers()), reports_dir=tmp_path / "reports")


@pytest.fixture
def client(tmp_path):
    app = create_app(make_store(tmp_path), rate_limit=3, warm_samples=False)
    with TestClient(app) as c:
        yield c


def wait_done(client, run_id, timeout=30.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        run = client.get(f"/api/runs/{run_id}").json()
        if run["status"] in ("done", "error"):
            return run
        time.sleep(0.05)
    raise AssertionError(f"run {run_id} did not finish")


EXAMPLES = [
    {"input": f"ticket {i}: {'refund please' if i % 2 else 'the app crashes'}", "expected": "refund" if i % 2 else "bug"}
    for i in range(12)
]


def test_health_reports_offline_mode(client):
    h = client.get("/api/health").json()
    assert h["mode"] == "offline" and h["race_provenance"] == "mock" and h["nemotron"] is False
    assert h["providers"] == ["arxiv", "openalex"]
    assert h["limits"]["runs_per_hour"] == 3 and "tavily" in h


def test_samples_and_families(client):
    samples = client.get("/api/samples").json()
    triage = next(s for s in samples if s["name"] == "ticket_triage")
    assert triage["n_examples"] == 40 and "billing" in triage["labels"]
    fams = client.get("/api/families").json()
    assert len(fams) == 5
    assert all(f["mermaid"].startswith("flowchart") and f["helps"] and f["hurts"] for f in fams)


def test_sample_run_end_to_end_is_mock_labelled_persisted_and_cached(client, tmp_path):
    r = client.post("/api/runs", json={"sample": "ticket_triage"})
    assert r.status_code == 202 and r.json()["cached"] is False
    run = wait_done(client, r.json()["id"])
    assert run["status"] == "done", run["error"]
    assert run["mode"] == "offline" and run["race_provenance"] == "mock"
    stages = {s["key"]: s["status"] for s in run["stages"]}
    assert stages == {"plain_llm": "skipped", "research": "done", "design": "done", "race": "done", "advisor": "skipped"}

    res = run["results"]
    assert res["plain_llm"]["available"] is False
    assert res["research"]["cards"] and res["research"]["methods_map"]
    assert all(c["mermaid"].startswith("flowchart") for c in res["candidates"])
    assert {c["role"] for c in res["candidates"]} >= {"baseline", "llm-default"}
    assert res["race"]["provenance"] == "mock" and res["race"]["task"]["n_examples"] == 40
    assert res["verdict"]["tested"] and res["verdict"]["provenance"] == "mock"
    assert res["verdict"]["recommended_id"] in {c["id"] for c in res["candidates"]}
    assert res["advisor"]["available"] is False

    receipt = {e["stage"]: e for e in run["receipt"]}
    assert receipt["race"]["provenance"] == "mock" and receipt["race"]["cost_usd"] == 0.0
    assert sum(e["cost_usd"] for e in run["receipt"]) == 0.0  # nothing real was spent
    assert any("MOCK" in line for line in run["limitations"])

    assert (tmp_path / "reports" / f"{run['id']}.json").is_file()
    again = client.post("/api/runs", json={"sample": "ticket_triage"}).json()
    assert again == {"id": run["id"], "cached": True, "status": "done", "url": f"/r/{run['id']}"}

    page = client.get(f"/r/{run['id']}")
    assert page.status_code == 200 and "text/html" in page.headers["content-type"]


def test_report_survives_restart(tmp_path):
    with TestClient(create_app(make_store(tmp_path), warm_samples=False)) as c1:
        run_id = c1.post("/api/runs", json={"problem": "Route support tickets to refund or bug queues.", "examples": EXAMPLES}).json()["id"]
        wait_done(c1, run_id)
    with TestClient(create_app(make_store(tmp_path), warm_samples=False)) as c2:
        run = c2.get(f"/api/runs/{run_id}").json()
        assert run["status"] == "done" and run["results"]["race"]["task"]["n_examples"] == 12
        assert c2.get("/api/runs/doesnotexist").status_code == 404


def test_problem_without_examples_skips_the_race(client):
    run_id = client.post("/api/runs", json={"problem": "Summarise long legal contracts for a busy founder."}).json()["id"]
    run = wait_done(client, run_id)
    assert run["status"] == "done"
    stages = {s["key"]: s["status"] for s in run["stages"]}
    assert stages["race"] == "skipped" and "race" not in run["results"]
    assert run["results"]["candidates"] and run["results"]["verdict"]["tested"] is False


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
