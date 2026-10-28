"""MCP server tests. No network and no API key: the research providers are replaced with fakes."""

import json
import math
import os
import sys
import time

import anyio
import pytest
from mcp.server.mcpserver.exceptions import ToolError

import windtunnel.mcp_server as server
from windtunnel.budget import BudgetExceeded, BudgetGuard
from windtunnel.engine import STAGES
from windtunnel.llm import MockClient, is_mock
from windtunnel.scout import Source

PROBLEM = "Route support tickets to billing, bug or shipping. Must cost under $2 per 1000."
LABELS = ["billing", "bug", "shipping"]


def fake_providers():
    def arxiv(query, limit, fetcher):
        return [
            Source("Self-Consistency Improves Chain of Thought Reasoning", "https://arxiv.org/abs/2203.11171", "paper", 2022,
                   arxiv_id="2203.11171", abstract="Majority voting over sampled reasoning paths.", found_via=["arxiv"], queries=[query]),
            Source("Self-Refine: Iterative Refinement with Self-Feedback", "https://arxiv.org/abs/2303.17651", "paper", 2023,
                   arxiv_id="2303.17651", abstract="The model critiques and self-corrects its own output.", found_via=["arxiv"], queries=[query]),
        ]

    def openalex(query, limit, fetcher):
        return [Source(f"Voting for fault tolerance: {query}", "https://doi.org/10.1/x", "paper", 1995, doi=f"10.1/{query}",
                       abstract="Redundant voting.", citation_count=12, found_via=["openalex"], queries=[query])]

    def tavily(query, limit, fetcher):
        return [Source("FrugalGPT cascades", "https://example.org/frugal", "web",
                       abstract="An LLM cascade sends easy queries to cheap models.", found_via=["tavily"], queries=[query])]

    return {"arxiv": arxiv, "openalex": openalex, "tavily": tavily}


@pytest.fixture(autouse=True)
def offline(monkeypatch, tmp_path):
    monkeypatch.delenv("NEBIUS_API_KEY", raising=False)
    monkeypatch.setattr(server, "_providers", fake_providers)
    monkeypatch.setattr(server, "JOBS_DIR", tmp_path / "mcp-jobs")
    server._research_cache.clear()
    server._jobs.clear()
    yield


def examples(n=30):
    return [{"input": f"ticket #{i}: issue about {LABELS[i % 3]}", "expected": LABELS[i % 3]} for i in range(n)]


def wait_done(job_id, timeout=30.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        report = server.get_report(job_id)
        if report["status"] in ("done", "failed"):
            return report
        time.sleep(0.05)
    raise AssertionError(f"job {job_id} did not finish: {report}")


def strict_json(obj) -> str:
    """Agent-facing output must be plain, strict JSON: no NaN or Infinity."""
    return json.dumps(obj, allow_nan=False)


# ---------------------------------------------------------------- research and design


def test_find_methods_offline_map_with_citations():
    out = server.find_methods(PROBLEM, max_sources=8)
    assert out["mode"] == "offline" and "OFFLINE" in out["mode_note"] and out["labels"] == ["OFFLINE"]
    assert "classification" in out["problem_spec"]["problem_types"]
    families = {m["family"] for m in out["methods"]}
    assert {"self_consistency", "self_check"} <= families
    card = next(c for m in out["methods"] for c in m["cards"] if c["url"] == "https://arxiv.org/abs/2203.11171")
    assert card["verified"] and card["year"] == 2022 and card["mechanism"]
    assert out["receipt"]["understand"] and out["receipt"]["extract"] == ["offline-heuristic"]
    strict_json(out)


def test_find_methods_passes_provider_warnings_through(monkeypatch):
    def broken(query, limit, fetcher):
        raise RuntimeError("search API down")

    monkeypatch.setattr(server, "_providers", lambda: {**fake_providers(), "openalex": broken})
    out = server.find_methods(PROBLEM, max_sources=8)
    assert out["methods"] and any(w.startswith("openalex failed") for w in out["warnings"])


def test_warnings_are_clipped_and_counted_not_dropped_silently():
    out = server._warnings([f"warning {i} " + "x" * 500 for i in range(15)] + ["", None])
    assert len(out) == server.MAX_WARNINGS + 1
    assert all(len(w) <= 300 for w in out)
    assert out[-1] == "... and 5 more warning(s) not shown"


def test_design_candidates_have_diagrams_tiers_and_citations():
    out = server.design_candidates(PROBLEM)
    roles = {c["role"] for c in out["candidates"]}
    assert {"baseline", "llm-default"} <= roles
    for cand in out["candidates"]:
        assert cand["mermaid"].startswith("flowchart") and cand["model"]
        assert cand["model"] in cand["tiers"]
    voted = next(c for c in out["candidates"] if c["family"] == "self_consistency")
    assert voted["based_on"] and all(b["url"] and b["title"] for b in voted["based_on"])
    # A cascade calls a cheap tier and escalates to a stronger one; both are listed.
    cascade = next(c for c in out["candidates"] if c["family"] == "cascade")
    assert len(cascade["tiers"]) == 2 and cascade["tiers"][0] == cascade["model"]
    crew = next(c for c in out["candidates"] if c["role"] == "llm-default")
    assert crew["tiers"] == ["super", "ultra"]
    strict_json(out)


def test_design_candidates_passes_design_warnings_through(monkeypatch):
    real = server.generate.design_candidates

    def with_warning(spec, cards, client=None, *, warnings=None, **kw):
        out = real(spec, cards, client, warnings=warnings, **kw)
        warnings.append("design slot wildcard: dropped as a near-duplicate of an earlier design")
        return out

    monkeypatch.setattr(server.generate, "design_candidates", with_warning)
    out = server.design_candidates(PROBLEM)
    assert any("near-duplicate" in w for w in out["warnings"])


def test_research_outputs_are_nan_safe(monkeypatch):
    def nan_year(query, limit, fetcher):
        return [Source("Self-consistency voting with NaN metadata", "https://arxiv.org/abs/2401.00001", "paper", float("nan"),
                       arxiv_id="2401.00001", abstract="Majority voting over samples.", citation_count=float("nan"),
                       found_via=["arxiv"], queries=[query])]

    monkeypatch.setattr(server, "_providers", lambda: {"arxiv": nan_year})
    for out in (server.find_methods(PROBLEM, max_sources=5), server.design_candidates(PROBLEM), server.compare_with_plain_llm(PROBLEM)):
        strict_json(out)
    card = next(c for m in server.find_methods(PROBLEM, max_sources=5)["methods"] for c in m["cards"])
    assert card["year"] is None


def test_compare_with_plain_llm_offline_is_honest():
    out = server.compare_with_plain_llm(PROBLEM)
    without = out["without_wind_tunnel"]
    assert without["available"] is False and without["text"] is None
    assert "Needs Nemotron" in without["unavailable_reason"] and "UNAVAILABLE" in without["note"]
    assert without["sources_cited"] == 0
    with_wt = out["with_wind_tunnel"]
    assert with_wt["tested"] is False and with_wt["n_verified_citations"] > 0
    assert any("baseline" in c for c in with_wt["candidates_to_race"])
    assert out["receipt"]["plain_llm"] is None
    strict_json(out)


def test_compare_with_plain_llm_failure_is_unavailable_not_faked(monkeypatch):
    def refuse(problem, client=None, model="ultra"):
        raise BudgetExceeded("cap reached")

    monkeypatch.setattr(server, "plain_llm_answer", refuse)
    without = server.compare_with_plain_llm(PROBLEM)["without_wind_tunnel"]
    assert without["available"] is False and without["text"] is None
    assert "BudgetExceeded" in without["unavailable_reason"]


# ---------------------------------------------------------------- race and report


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"examples": examples(3)}, "at least"),
        ({"examples": [{"input": "x"}] * 10}, "'input' and 'expected'"),
        ({"examples": examples(), "scorer": "fuzzy"}, "unknown scorer"),
    ],
)
def test_start_race_validates_input(kwargs, message):
    with pytest.raises(ToolError, match=message):
        server.start_race(PROBLEM, **kwargs)


def test_race_without_key_runs_the_engine_and_is_labelled_mock(monkeypatch):
    monkeypatch.setenv("WINDTUNNEL_MCP_MAX_EXAMPLES", "24")
    started = server.start_race(PROBLEM, examples(30), labels=LABELS)
    assert started["mode"] == "mock" and "MOCK" in started["mode_note"] and started["breed"] is True
    assert started["labels"] == ["OFFLINE", "MOCK", "SIMULATED"]
    assert started["n_examples"] == 24 and any("Dropped 6" in w for w in started["warnings"])

    report = wait_done(started["job_id"])
    assert report["status"] == "done", report["error"]
    strict_json(report)
    # Progress came from the engine's callback, one entry per stage.
    assert list(report["progress"]["stages"]) == STAGES
    assert set(report["progress"]["stages"].values()) <= {"done", "skipped"}
    assert report["progress"]["stage"] == "done" and report["progress"]["simulated_model_calls"] > 0
    assert "model_calls" not in report["progress"]  # nothing real was called

    result = report["result"]
    assert result["mode"] == "offline" and result["provenance"] == "mock" and result["do_not_quote"] is True
    assert result["labels"] == ["OFFLINE", "MOCK", "SIMULATED"] and report["labels"] == result["labels"]
    assert "OFFLINE" in result["mode_note"] and "MOCK" in result["mode_note"]
    assert result["verdict"]["headline"].startswith("[MOCK")
    assert result["total_cost_usd"] == 0.0 and result["simulated_cost_usd"] > 0
    assert any(line.startswith("MOCK") for line in result["limitations"])
    assert any(line.startswith("OFFLINE") for line in result["limitations"])
    assert "limitations" in result["next"]

    rec = result["recommended"]
    assert rec["id"] == result["verdict"]["recommended_id"] and rec["id"] in result["verdict"]["tied_ids"]
    assert rec["mermaid"].startswith("flowchart") and rec["model"] in rec["tiers"]
    row = result["leaderboard"][0]
    assert {"accuracy", "accuracy_ci95", "cost_per_1k_usd", "p50_latency_s", "n", "status", "tiers", "origin"} <= set(row)
    assert sum(r["status"] == "recommended" for r in result["leaderboard"]) == 1

    # Offline the plain answer and the advisor are unavailable, and say so.
    assert result["plain_llm"]["available"] is False and result["plain_llm"]["text"] is None
    assert result["plain_llm"]["unavailable_reason"]
    assert result["advisor"]["available"] is False and result["advisor"]["typical_default_id"] == "crew-ultra"
    assert [e["stage"] for e in result["model_receipt"]] == STAGES
    assert next(e for e in result["model_receipt"] if e["stage"] == "race")["produced_by"] == "mock"


def test_race_reports_breeding_compactly():
    report = wait_done(server.start_race(PROBLEM, examples(30), labels=LABELS)["job_id"])
    result = report["result"]
    breeding = result["breeding"]
    assert breeding["enabled"] is True and breeding["note"]
    assert breeding["selection_bias"].startswith("present")  # 30 examples is too few for a held-out split
    assert breeding["holdout_examples"] == 0 and breeding["search_examples"] == 30
    assert breeding["diagnoses"] and all(d["failure_modes"] for d in breeding["diagnoses"])
    assert {"mode", "share", "flagged_by"} <= set(breeding["diagnoses"][0]["failure_modes"][0])
    assert breeding["bred"] and len(breeding["mechanisms_applied"]) == len(breeding["bred"])
    for bred, mech in zip(breeding["bred"], breeding["mechanisms_applied"]):
        assert mech["bred_id"] == bred["id"] and mech["applied_to"] == bred["parent_id"]
        assert mech["mechanism"] == bred["mechanism"] and mech["field"] and mech["source"]["url"]
        assert bred["tiers"] and bred["generation"] == 1 and "mermaid" not in bred
    assert [r["role"] for r in breeding["races"]][0] == "initial"
    bred_ids = {b["id"] for b in breeding["bred"]}
    assert {r["id"] for r in result["leaderboard"] if r["origin"] == "bred"} <= bred_ids
    assert any("optimistic" in line for line in result["limitations"])
    if result["recommended"]["origin"] == "bred":
        assert result["recommended"]["lineage"]["parent_id"]


def test_race_with_breed_false_skips_breeding():
    started = server.start_race(PROBLEM, examples(20), labels=LABELS, breed=False)
    assert started["breed"] is False
    report = wait_done(started["job_id"])
    assert report["status"] == "done", report["error"]
    assert report["breed"] is False
    result = report["result"]
    assert result["breeding"] == {"enabled": False, "note": "Breeding was turned off for this run."}
    assert all(r["origin"] != "bred" for r in result["leaderboard"])
    assert result["eliminated_earlier"] == []


def test_counting_client_keeps_budget_guard_and_mock_visible(tmp_path):
    guard = BudgetGuard(MockClient({"x": "billing"}, LABELS), limit_usd=1.0, ledger=tmp_path / "spend.json")
    counter = server.CountingClient(guard)
    assert counter.client is guard and is_mock(counter)  # the engine still sees a mock as a mock
    counter.complete("nano", [{"role": "user", "content": "x"}])
    assert counter.calls == 1


def test_labels_come_from_what_ran_not_from_what_was_asked(monkeypatch, tmp_path):
    # A "real" client that is a mock underneath must still be reported as MOCK by the engine.
    monkeypatch.setattr(server, "_have_key", lambda: True)
    monkeypatch.setattr(server, "_real_client", lambda: BudgetGuard(MockClient({}, LABELS), ledger=tmp_path / "spend.json"))
    started = server.start_race(PROBLEM, examples(12), labels=LABELS, breed=False)
    assert started["mode"] == "nemotron"
    report = wait_done(started["job_id"])
    assert report["status"] == "done", report["error"]
    assert report["result"]["provenance"] == "mock" and "MOCK" in report["labels"] and report["result"]["do_not_quote"]
    assert "spend_cap" in report["result"] and "model_calls" in report["progress"]


def test_compact_report_is_nan_safe():
    report = wait_done(server.start_race(PROBLEM, examples(30), labels=LABELS)["job_id"])
    full = json.loads((server.JOBS_DIR / f"{report['job_id']}.json").read_text())["full"]
    nan = float("nan")
    # An unmeasured design has no accuracy, latency or interval; a broken cost can be infinite.
    row = full["race"]["designs"][0]
    row.update(accuracy=nan, accuracy_ci=[nan, math.inf], p50_latency_s=nan, cost_per_1k_usd=math.inf, n=0)
    for diag in full["breeding"]["diagnoses"][0]:
        diag["accuracy"] = nan
    full["verdict"]["recommended"]["accuracy"] = nan
    full["total_cost_usd"] = nan
    out = server._compact_report(full)
    strict_json(out)
    nan_row = next(r for r in out["leaderboard"] if r["id"] == row["id"])
    assert nan_row["accuracy"] is None and nan_row["accuracy_ci95"] == [None, None] and nan_row["cost_per_1k_usd"] is None
    assert out["recommended"]["accuracy"] is None and out["total_cost_usd"] is None
    assert out["breeding"]["diagnoses"][0]["accuracy"] is None

    # The persisted job file is strict JSON too, even when the engine report carries NaN.
    job = server.Job("nanjob", PROBLEM, 1, "mock")
    job.full = {"x": nan, "y": [math.inf]}
    job.save()
    saved = json.loads((server.JOBS_DIR / "nanjob.json").read_text())
    assert saved["full"] == {"x": None, "y": [None]}


def test_report_survives_restart_from_disk():
    job_id = server.start_race(PROBLEM, examples(10), labels=LABELS)["job_id"]
    assert wait_done(job_id)["status"] == "done"
    saved = json.loads((server.JOBS_DIR / f"{job_id}.json").read_text())
    assert saved["full"]["provenance"] == "mock" and saved["full"]["race"]["provenance"] == "mock"
    server._jobs.clear()  # simulate a server restart
    again = server.get_report(job_id)
    assert again["status"] == "done" and "full" not in again
    assert again["result"]["labels"] == ["OFFLINE", "MOCK", "SIMULATED"]
    with pytest.raises(ToolError, match="no job"):
        server.get_report("nope")


def test_running_job_on_disk_reports_interrupted():
    job = server.Job("halfway", PROBLEM, 10, "mock")
    job.status = "running"
    job.on_progress("research", "running")
    saved = json.loads((server.JOBS_DIR / "halfway.json").read_text())
    assert saved["progress"]["stage"] == "research" and saved["progress"]["stages"]["research"] == "running"
    out = server.get_report("halfway")
    assert out["status"] == "interrupted" and "restarted" in out["error"]


def test_failed_engine_run_reports_failed(monkeypatch):
    def boom(*args, progress=None, **kwargs):
        progress("research", "running")
        progress("research", "failed")
        raise RuntimeError("engine broke")

    monkeypatch.setattr(server, "run_wind_tunnel", boom)
    report = wait_done(server.start_race(PROBLEM, examples(10), labels=LABELS)["job_id"])
    assert report["status"] == "failed" and "engine broke" in report["error"]
    assert report["progress"]["stages"]["research"] == "failed" and report["result"] is None


# ---------------------------------------------------------------- MCP transport


def test_mcp_client_in_memory_lists_and_calls_tools():
    from mcp import Client

    async def run():
        async with Client(server.mcp) as client:
            tools = {t.name: t for t in (await client.list_tools()).tools}
            result = await client.call_tool("find_methods", {"problem": PROBLEM, "max_sources": 5})
            prompt = await client.get_prompt("design", {"problem": PROBLEM})
            return tools, result, prompt

    tools, result, prompt = anyio.run(run)
    assert set(tools) == {"find_methods", "design_candidates", "start_race", "get_report", "compare_with_plain_llm"}
    assert "breed" in tools["start_race"].input_schema["properties"]
    assert not result.is_error and result.structured_content["mode"] == "offline"
    text = prompt.messages[0].content.text
    assert "find_methods" in text and "start_race" in text and "get_report" in text and "limitations" in text


def test_stdio_server_starts_and_lists_tools(tmp_path):
    from mcp import Client
    from mcp.client.stdio import StdioServerParameters

    params = StdioServerParameters(command=sys.executable, args=["-m", "windtunnel.mcp_server"], cwd=str(tmp_path),
                                   env={"PATH": "", "WINDTUNNEL_OFFLINE": "1"})

    async def run():
        with anyio.fail_after(30):
            async with Client(params) as client:
                return {t.name for t in (await client.list_tools()).tools}

    assert "start_race" in anyio.run(run)


# ---------------------------------------------------------------- honest labels, caches and errors


def test_run_whose_research_and_designs_all_fell_back_is_labelled_offline():
    report = {
        "mode": "nemotron",
        "provenance": "measured",
        "research": {"mode": "offline", "spec": {"produced_by": "offline-heuristic"}, "cards": []},
        "candidates": [{"id": "a", "role": "candidate", "generated_by": "offline-heuristic"}, {"id": "b", "role": "baseline", "generated_by": "offline-heuristic"}],
    }
    labels = server._labels(report)
    assert labels == ["OFFLINE", "MEASURED"]
    note = server._mode_note(report, labels)
    assert "fallback" in note and "ran on NVIDIA Nemotron" not in note
    report["candidates"][0]["generated_by"] = "ultra"
    assert server._labels(report) == ["NEMOTRON", "PARTLY-OFFLINE", "MEASURED"]
    report["research"]["mode"] = "nemotron"
    assert server._labels(report) == ["NEMOTRON", "MEASURED"]
    assert server._mode_note(report, ["NEMOTRON", "MEASURED"]) == server.MODE_NOTES["nemotron"]


def test_get_report_normalises_the_job_id_and_never_calls_a_live_job_interrupted():
    job = server.Job("f28fe4e92a17", PROBLEM, 10, "nemotron")
    job.status = "running"
    with server._jobs_lock:
        server._jobs[job.id] = job
    job.save()
    for raw in ("'f28fe4e92a17'", " f28fe4e92a17\n", "F28FE4E92A17"):
        assert server.get_report(raw)["status"] == "running"
    # Another MCP server process owns it: still running, not "interrupted".
    server._jobs.clear()
    path = server.JOBS_DIR / "f28fe4e92a17.json"
    data = json.loads(path.read_text())
    data["pid"] = os.getppid()
    path.write_text(json.dumps(data))
    out = server.get_report("f28fe4e92a17")
    assert out["status"] == "running" and "another" in out["note"] and "pid" not in out


def test_job_without_a_result_never_claims_nemotron_ran(monkeypatch):
    labels, note = server._pending_labels("nemotron", "running")
    assert labels == ["NEMOTRON-REQUESTED"] and "ran on" not in note

    def boom(*args, progress=None, **kwargs):
        raise ConnectionError("Token Factory unreachable")

    monkeypatch.setattr(server, "_have_key", lambda: True)
    monkeypatch.setattr(server, "_real_client", lambda: None)
    monkeypatch.setattr(server, "run_wind_tunnel", boom)
    started = server.start_race(PROBLEM, examples(10), labels=LABELS)
    assert started["labels"] == ["NEMOTRON-REQUESTED"]
    report = wait_done(started["job_id"])
    assert report["status"] == "failed" and report["labels"] == ["NEMOTRON-REQUESTED", "FAILED"]
    assert "FAILED" in report["mode_note"] and "ran on NVIDIA Nemotron" not in report["mode_note"]


class BadPlanClient:
    """A real-looking client whose Ultra plan is unusable; every reply is paid for."""

    def __init__(self):
        self.calls = 0

    def complete(self, model, messages, *, temperature=0.0, max_tokens=1024):
        from windtunnel.llm import Completion

        self.calls += 1
        return Completion('["a list, not a plan"]', model, 500, 200, 0.5)


def test_research_fallback_is_not_cached_and_its_spend_is_counted():
    client = BadPlanClient()
    first = server._research(PROBLEM, 5, client)
    assert first.mode == "offline" and first.cost_usd > 0  # the rejected plan was paid for
    second = server._research(PROBLEM, 5, client)
    # Each attempt is the plan call plus one JSON retry; Nemotron is tried again, not the stale fallback.
    assert second is not first and client.calls == 4
    labels, note = server._research_labels(first, client)
    assert labels == ["OFFLINE"] and note == server.MODE_NOTES["fallback"]


def test_start_race_races_the_candidates_design_candidates_showed(monkeypatch):
    shown = server.design_candidates(PROBLEM)["candidates"]
    calls = []
    real = server.generate.design_candidates
    monkeypatch.setattr(server.generate, "design_candidates", lambda *a, **kw: calls.append(1) or real(*a, **kw))
    again = server.compare_with_plain_llm(PROBLEM)
    assert not calls  # reused, not designed (and paid for) again
    assert again["with_wind_tunnel"]["candidates_to_race"] == [f"{c['id']} ({c['role']}; {'+'.join(c['tiers'])})" for c in shown]
    started = server.start_race(PROBLEM, examples(10), labels=LABELS, breed=False)
    assert started["candidates"] == [c["id"] for c in shown]
    report = wait_done(started["job_id"])
    assert report["status"] == "done", report["error"]
    assert [c["id"] for c in report["result"]["candidates"]] == [c["id"] for c in shown]
    receipt = {e["stage"]: e for e in report["result"]["model_receipt"]}
    assert "reused" in receipt["design_candidates"]["detail"]


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"problem": "x" * 2001}, "problem is too long"),
        ({"examples": [{"input": "", "expected": ""}] * 10}, "non-empty"),
        ({"examples": [{"id": i, "input": "same ticket", "expected": "bug"} for i in range(10)]}, "duplicate inputs"),
        ({"max_cost_per_1k": -1}, "positive"),
        ({"output_instructions": "x" * 1001}, "output_instructions is too long"),
        ({"labels": ["x" * 61]}, "labels must be"),
    ],
)
def test_start_race_rejects_what_the_web_api_rejects(kwargs, message):
    args = {"problem": PROBLEM, "examples": examples(10), **kwargs}
    with pytest.raises(ToolError, match=message):
        server.start_race(**args)


def test_start_race_warns_when_inputs_are_truncated():
    rows = examples(10)
    rows[0]["input"] = "x" * (server.MAX_INPUT_CHARS + 10)
    started = server.start_race(PROBLEM, rows, labels=LABELS)
    assert any("1 example input(s)" in w and "truncated" in w for w in started["warnings"])
    wait_done(started["job_id"])


def test_errors_are_mcp_errors_and_list_jobs_from_disk():
    from mcp import Client

    job_id = server.start_race(PROBLEM, examples(10), labels=LABELS)["job_id"]
    wait_done(job_id)
    server._jobs.clear()  # a restart: the job only exists on disk now

    async def call():
        async with Client(server.mcp) as client:
            return await client.call_tool("get_report", {"job_id": "zzz"})

    result = anyio.run(call)
    assert result.is_error and "no job" in result.content[0].text and job_id in result.content[0].text
