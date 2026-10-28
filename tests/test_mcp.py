"""MCP server tests. No network: the research providers are replaced with fakes."""

import json
import sys
import time

import anyio
import pytest

import windtunnel.mcp_server as server
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

    return {"arxiv": arxiv, "openalex": openalex}


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


def wait_done(job_id, timeout=20.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        report = server.get_report(job_id)
        if report["status"] in ("done", "failed"):
            return report
        time.sleep(0.05)
    raise AssertionError(f"job {job_id} did not finish: {report}")


def test_find_methods_offline_map_with_citations():
    out = server.find_methods(PROBLEM, max_sources=8)
    assert out["mode"] == "offline" and "OFFLINE" in out["mode_note"]
    assert "classification" in out["problem_spec"]["problem_types"]
    families = {m["family"] for m in out["methods"]}
    assert {"self_consistency", "self_check"} <= families
    card = next(c for m in out["methods"] for c in m["cards"] if c["url"] == "https://arxiv.org/abs/2203.11171")
    assert card["verified"] and card["year"] == 2022 and card["mechanism"]
    assert out["receipt"]["understand"] and out["receipt"]["extract"] == ["offline-heuristic"]
    json.dumps(out)  # agent-facing output must be plain JSON


def test_design_candidates_have_diagrams_and_citations():
    out = server.design_candidates(PROBLEM)
    roles = {c["role"] for c in out["candidates"]}
    assert {"baseline", "llm-default"} <= roles
    for cand in out["candidates"]:
        assert cand["mermaid"].startswith("flowchart") and cand["model"]
    voted = next(c for c in out["candidates"] if c["family"] == "self_consistency")
    assert voted["based_on"] and all(b["url"] and b["title"] for b in voted["based_on"])


def test_compare_with_plain_llm_offline_is_honest():
    out = server.compare_with_plain_llm(PROBLEM)
    assert out["without_wind_tunnel"]["available"] is False
    assert out["without_wind_tunnel"]["sources_cited"] == 0
    with_wt = out["with_wind_tunnel"]
    assert with_wt["tested"] is False and with_wt["n_verified_citations"] > 0
    assert any("baseline" in c for c in with_wt["candidates_to_race"])


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"examples": examples(3)}, "at least"),
        ({"examples": [{"input": "x"}] * 10}, "'input' and 'expected'"),
        ({"examples": examples(), "scorer": "fuzzy"}, "unknown scorer"),
    ],
)
def test_start_race_validates_input(kwargs, message):
    out = server.start_race(PROBLEM, **kwargs)
    assert message in out["error"]


def test_race_without_key_is_mock_and_labelled(monkeypatch):
    monkeypatch.setenv("WINDTUNNEL_MCP_MAX_EXAMPLES", "24")
    started = server.start_race(PROBLEM, examples(30), labels=LABELS)
    assert started["mode"] == "mock" and "MOCK" in started["mode_note"]
    assert started["n_examples"] == 24 and any("Dropped 6" in w for w in started["warnings"])

    report = wait_done(started["job_id"])
    assert report["status"] == "done", report["error"]
    result = report["result"]
    assert result["provenance"] == "mock" and result["do_not_quote"] is True
    assert result["verdict"].startswith("MOCK")
    assert any("MOCK" in line for line in result["limitations"])
    assert result["recommended"]["id"] in result["tied_ids"]
    assert result["recommended"]["mermaid"].startswith("flowchart")
    row = result["leaderboard"][0]
    assert {"accuracy", "accuracy_ci95", "cost_per_1k_usd", "p50_latency_s", "n", "status"} <= set(row)
    assert result["advisor"]["model"] == "ultra"
    assert report["progress"]["model_calls"] > 0


def test_report_survives_restart_from_disk():
    job_id = server.start_race(PROBLEM, examples(10), labels=LABELS)["job_id"]
    assert wait_done(job_id)["status"] == "done"
    saved = json.loads((server.JOBS_DIR / f"{job_id}.json").read_text())
    assert saved["full"]["race"]["provenance"] == "mock"
    server._jobs.clear()  # simulate a server restart
    again = server.get_report(job_id)
    assert again["status"] == "done" and "full" not in again
    assert "error" in server.get_report("nope")


def test_mcp_client_in_memory_lists_and_calls_tools():
    from mcp import Client

    async def run():
        async with Client(server.mcp) as client:
            tools = {t.name for t in (await client.list_tools()).tools}
            result = await client.call_tool("find_methods", {"problem": PROBLEM, "max_sources": 5})
            prompt = await client.get_prompt("design", {"problem": PROBLEM})
            return tools, result, prompt

    tools, result, prompt = anyio.run(run)
    assert tools == {"find_methods", "design_candidates", "start_race", "get_report", "compare_with_plain_llm"}
    assert not result.is_error and result.structured_content["mode"] == "offline"
    text = prompt.messages[0].content.text
    assert "find_methods" in text and "start_race" in text and "get_report" in text


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
