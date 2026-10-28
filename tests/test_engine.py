import json
import re

import pytest

from windtunnel import cli
from windtunnel.budget import BudgetGuard
from windtunnel.engine import STAGES, format_report, run_wind_tunnel
from windtunnel.llm import Completion, MockClient
from windtunnel.racer import RaceConfig
from windtunnel.scout import Source
from windtunnel.task import Constraints, Example, TaskSpec, load_task

PROBLEM = "Route support tickets into billing, bug or shipping queues. Must cost under $5 per 1000."
LABELS = ["billing", "bug", "shipping"]


def fake_providers():
    def arxiv(query, limit, fetcher):
        return [
            Source("Self-Consistency Improves Chain of Thought", "https://arxiv.org/abs/2203.11171", "paper", 2022,
                   arxiv_id="2203.11171", abstract="Majority voting over sampled reasoning paths.", found_via=["arxiv"], queries=[query]),
            Source("Self-Refine: iterative self-verification", "https://arxiv.org/abs/2303.17651", "paper", 2023,
                   arxiv_id="2303.17651", abstract="Self-verification and critique improve answers.", found_via=["arxiv"], queries=[query]),
        ]

    def tavily(query, limit, fetcher):
        return [Source("FrugalGPT cascades", "https://example.org/frugal", "web",
                       abstract="An LLM cascade sends easy queries to cheap models.", found_via=["tavily"], queries=[query])]

    return {"arxiv": arxiv, "tavily": tavily}


def make_task(n: int = 30) -> TaskSpec:
    examples = [Example(f"e{i:03d}", f"ticket number {i:03d} about something", LABELS[i % 3]) for i in range(n)]
    return TaskSpec("tickets", "Route the ticket to one queue.", "Answer with the queue only.", "exact", examples,
                    Constraints(max_cost_per_1k=5.0), LABELS, provenance="synthetic test fixture")


def run(task=None, client=None, **kw):
    events = []
    report = run_wind_tunnel(PROBLEM, task, client, providers=fake_providers(), fetcher=object(),
                             progress=lambda stage, status: events.append((stage, status)), **kw)
    json.dumps(report)  # the report must be JSON-serialisable as is
    return report, events


def receipt(report) -> dict:
    return {e["stage"]: e for e in report["model_receipt"]}


def test_offline_run_has_every_section_honestly_labelled():
    report, events = run(make_task())
    assert report["mode"] == "offline" and report["provenance"] == "mock"
    assert [e["stage"] for e in report["model_receipt"]] == STAGES
    assert [s for s, status in events if status == "running"] == STAGES

    # Without: no plain answer offline, and it says so.
    assert report["plain_answer"]["available"] is False
    # Research and candidates come from heuristics, with diagrams.
    assert report["research"]["mode"] == "offline" and report["research"]["cards"]
    assert {c["generated_by"] for c in report["candidates"]} == {"offline-heuristic"}
    assert all(c["mermaid"].startswith("flowchart") for c in report["candidates"])
    roles = [c["role"] for c in report["candidates"]]
    assert roles[0] == "baseline" and roles[-1] == "llm-default"
    assert any(c["family"] == "cascade" for c in report["candidates"])  # from the FrugalGPT source

    # The race is mock, and every place that shows numbers says so.
    assert report["race"]["provenance"] == "mock"
    assert report["verdict"]["provenance"] == "mock" and report["verdict"]["headline"].startswith("[MOCK")
    r = receipt(report)
    assert r["race"]["produced_by"] == "mock" and r["race"]["cost_usd"] == 0 and r["race"]["simulated_cost_usd"] > 0
    assert r["plain_llm_answer"]["status"] == r["llm_pick"]["status"] == "skipped"
    assert r["research"]["produced_by"] == r["design_candidates"]["produced_by"] == "offline-heuristic"
    assert report["total_cost_usd"] == 0

    # Breeding: bred designs, lineage and the selection-bias note (30 examples is too few to hold any out).
    b = report["breeding"]
    assert b["enabled"] and b["bred"] and set(b["lineage"]) == {d["id"] for d in b["bred"]}
    assert b["holdout_ids"] == [] and "optimistic" in b["note"] and "present" in b["selection_bias"]
    assert all(d["mermaid"].startswith("flowchart") for d in b["bred"])

    assert report["advisor"]["available"] is False and report["advisor"]["typical_default_id"] == "crew-ultra"
    text = " ".join(report["limitations"])
    for phrase in ["MOCK race", "OFFLINE mode", "abstracts", "only 30 examples", "optimistic", "synthetic test fixture", "needs Nemotron"]:
        assert phrase in text, phrase
    assert "spent $" not in text


def test_no_task_skips_the_race_and_says_so():
    report, events = run(None)
    assert report["race"] is None and report["breeding"] is None and report["provenance"] == "untested"
    assert report["verdict"]["tested"] is False
    assert receipt(report)["race"]["status"] == "skipped" and ("race", "skipped") in events
    assert any("nothing was raced" in item for item in report["limitations"])


def test_no_breed_runs_a_single_race():
    report, _ = run(make_task(), breed=False)
    assert report["breeding"]["enabled"] is False and report["race"]["role"] == "initial"
    assert not any(item.startswith("Breeding:") for item in report["limitations"])


def test_large_task_confirms_on_held_out_split():
    report, _ = run(make_task(70), config=RaceConfig(seed=1))
    b = report["breeding"]
    assert len(b["holdout_ids"]) == 21 and report["race"]["role"] == "confirmation (held-out split)"
    assert report["race"]["task"]["n_examples"] == 21 and "held-out" in b["selection_bias"]
    assert not any("chosen from failures on the same" in item for item in report["limitations"])


def test_mock_client_passed_in_still_means_offline_and_mock():
    task = make_task()
    report, _ = run(task, BudgetGuard(MockClient.for_task(task), limit_usd=1.0, ledger="/dev/null/never-written"))
    assert report["mode"] == "offline" and report["provenance"] == "mock"
    assert receipt(report)["race"]["produced_by"] == "mock"


class FakeNemotron:
    """Plays every Nemotron role from canned JSON; race and advisor answers come from a MockClient."""

    def __init__(self, task: TaskSpec):
        self.mock = MockClient.for_task(task)

    def complete(self, model, messages, *, temperature=0.0, max_tokens=1024):
        system = messages[0]["content"] if messages[0]["role"] == "system" else ""
        user = next(m["content"] for m in messages if m["role"] == "user")
        if "research engineer" in system:
            data = {"summary": "Route tickets", "problem_types": ["classification"], "search_queries": ["q1", "q2"], "analogy_queries": ["a1"]}
        elif "extract research methods" in system:
            family = "self_consistency" if "Self-Consistency" in user else "self_check"
            data = {"method_name": "M", "family": family, "mechanism": "how", "relevance": 3}
        elif "Design slot:" in user:
            slot = re.search(r"^Design slot: (\S+)$", user, re.MULTILINE).group(1)
            data = {"family": "cascade" if slot == "wildcard" else slot, "model": "super", "params": {"n": 3},
                    "instructions": "Watch for refunds.", "rationale": "fits", "based_on": [], "expected_tradeoff": "more calls"}
        elif "reliability engineer" in system:
            data = {"failure_modes": [{"mode": "label_confusion", "evidence": "billing read as bug"}]}
        elif user.startswith("I'm building this"):
            return Completion("Use a multi-agent crew on Ultra.", model, 300, 100, 1.0)
        else:
            return self.mock.complete(model, messages, temperature=temperature, max_tokens=max_tokens)
        return Completion(json.dumps(data), model, 500, 200, 0.5)


def test_nemotron_mode_labels_each_stage_with_its_model():
    task = make_task()
    report, _ = run(task, FakeNemotron(task))
    assert report["mode"] == "nemotron" and report["provenance"] == "measured"
    r = receipt(report)
    assert r["plain_llm_answer"]["produced_by"] == "ultra" and r["plain_llm_answer"]["cost_usd"] > 0
    assert r["research"]["produced_by"] == "ultra+super"
    assert r["design_candidates"]["produced_by"] == "ultra+offline-heuristic"  # reference points are templates
    assert r["llm_pick"]["produced_by"] == "ultra" and report["advisor"]["available"]
    assert r["race"]["produced_by"] != "mock" and "simulated_cost_usd" not in r["race"]
    assert "ultra" in r["race"]["detail"]  # failures diagnosed by the model
    assert report["plain_answer"]["available"] and "advisor_pick" in report["verdict"]
    assert report["total_cost_usd"] == pytest.approx(sum(e["cost_usd"] for e in report["model_receipt"]))
    text = " ".join(report["limitations"])
    assert "MOCK" not in text and "OFFLINE" not in text and "spent $" in text
    assert "WITHOUT" in format_report(report)


def test_failed_stage_is_reported_to_progress():
    class Broken:
        def complete(self, *a, **kw):
            raise RuntimeError("down")

    events = []
    with pytest.raises(RuntimeError):
        run_wind_tunnel(PROBLEM, None, Broken(), providers=fake_providers(), fetcher=object(),
                        progress=lambda s, st: events.append((s, st)))
    assert events[-1] == ("plain_llm_answer", "failed")


def test_cli_run_offline_smoke(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)  # no .env here, and nothing written into the repo
    monkeypatch.setenv("NEBIUS_API_KEY", "set-but-offline")
    monkeypatch.setattr(cli, "PROVIDERS", fake_providers())

    def no_real_client(*a, **kw):
        raise AssertionError("--offline must not build a Token Factory client")

    monkeypatch.setattr(cli, "TokenFactoryClient", no_real_client)
    assert cli.main(["run", PROBLEM, "--sample", "ticket_triage", "--offline", "--providers", "arxiv,tavily", "--out", "out"]) == 0
    printed = capsys.readouterr().out
    for section in ["WITHOUT", "WITH WIND TUNNEL", "TOP METHODS FOUND", "CANDIDATES", "LEADERBOARD", "VERDICT", "LIMITATIONS", "MODEL RECEIPT"]:
        assert section in printed, section
    [path] = (tmp_path / "out").glob("run-ticket_triage-offline-mock-*.json")
    report = json.loads(path.read_text())
    assert report["provenance"] == "mock" and report["task"]["n_examples"] == len(load_task(cli.TASKS_DIR / "ticket_triage").examples)
    records = [json.loads(line) for line in (tmp_path / "atlas" / "outcomes.jsonl").read_text().splitlines()]
    assert records and {r["provenance"] for r in records} == {"mock"}
    assert len({r["design"]["id"] for r in records}) == len(records) == report["atlas"]["records"]  # each design once


def test_cli_run_without_examples(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("NEBIUS_API_KEY", raising=False)
    monkeypatch.setattr(cli, "PROVIDERS", fake_providers())
    assert cli.main(["run", PROBLEM, "--providers", "arxiv", "--no-breed", "--out", "out"]) == 0
    assert "untested" in capsys.readouterr().out
    assert not (tmp_path / "atlas").exists()  # nothing raced, nothing logged
    assert list((tmp_path / "out").glob("run-untested-offline-untested-*.json"))


class Metered:
    """Records every completion that reached the API, so a test can compare real spend with the report."""

    def __init__(self, client):
        self.client = client
        self.spent = 0.0

    def complete(self, model, messages, **kw):
        completion = self.client.complete(model, messages, **kw)
        self.spent += completion.cost_usd
        return completion


class BadPlan(FakeNemotron):
    def complete(self, model, messages, **kw):
        if "research engineer" in messages[0]["content"]:
            return Completion('["a list, not a plan"]', model, 500, 200, 0.5)
        return super().complete(model, messages, **kw)


def test_unusable_plan_falls_back_and_its_spend_is_on_the_receipt():
    task = make_task()
    client = Metered(BadPlan(task))
    report, _ = run(task, client)
    r = receipt(report)
    assert report["research"]["spec"]["produced_by"] == "offline-heuristic"
    assert "ultra(rejected)" in r["research"]["produced_by"] and r["research"]["cost_usd"] > 0
    assert report["total_cost_usd"] == pytest.approx(client.spent, abs=1e-6)  # every paid call is counted
    assert any("plan fell back" in item for item in report["limitations"])


def test_receipt_total_matches_real_spend():
    task = make_task()
    client = Metered(FakeNemotron(task))
    report, _ = run(task, client)
    assert report["total_cost_usd"] == pytest.approx(client.spent, abs=1e-6)


def test_task_with_no_examples_is_untested_not_a_nan_winner():
    report, _ = run(TaskSpec("empty", "Route.", "", "exact", []))
    assert report["verdict"]["tested"] is False and report["race"] is None
    json.dumps(report, allow_nan=False)


class CapAfter(FakeNemotron):
    """Refuses every call, like BudgetGuard, once `limit` calls have been made."""

    def __init__(self, task, limit):
        super().__init__(task)
        self.limit, self.calls = limit, 0

    def complete(self, model, messages, **kw):
        from windtunnel.budget import BudgetExceeded

        self.calls += 1
        if self.calls > self.limit:
            raise BudgetExceeded("spending cap reached")
        return super().complete(model, messages, **kw)


def test_budget_running_out_mid_race_still_returns_an_honest_report():
    task = make_task()
    report, _ = run(task, CapAfter(task, 60))
    verdict = report["verdict"]
    assert verdict["headline"].startswith("[INCOMPLETE]") and verdict["recommended_id"] is None
    assert verdict["aborted"] and report["advisor"]["available"] is False
    text = " ".join(report["limitations"])
    assert "INCOMPLETE" in text and "No design met" not in text
    assert all(not d["errors"] for d in report["race"]["designs"])  # refusals are not "wrong answers"
    json.dumps(report, allow_nan=False)


def test_verdict_counts_only_the_examples_the_recommendation_saw():
    task = make_task(100)
    client = MockClient.for_task(task, skill={"nano": 0.0, "lightning": 0.0, "super": 0.0, "ultra": 1.0})
    report, _ = run(task, client, breed=False, config=RaceConfig(schedule=(20, 100)))
    verdict = report["verdict"]
    assert verdict["n_examples"] == verdict["recommended"]["n"]
    if verdict["n_examples"] < verdict["examples_in_race"]:
        assert any("measured on only" in item for item in report["limitations"])
    assert any(f"only {verdict['n_examples']} examples" in item for item in report["limitations"])


def test_advisor_comparison_says_when_numbers_come_from_another_race():
    task = make_task(70)
    report, _ = run(task, FakeNemotron(task), config=RaceConfig(seed=1))
    verdict = report["verdict"]
    pick = verdict["advisor_pick"]["measured"]
    assert pick["id"] != verdict["recommended_id"]
    assert pick["comparable"] == (pick["measured_in"] == verdict["decided_in"])
    # A number from an earlier race (other split, other n) is never set side by side as "vs".
    pick.update(comparable=False, measured_in="initial (search split)", n=10)
    text = format_report(report)
    assert "measured in initial (search split): accuracy" in text and "not comparable" in text


# ---------------------------------------------------------------- honest comparisons and labels


def test_unmeasured_llm_default_has_no_cost_not_zero():
    class FlakyUltra(MockClient):
        def complete(self, model, messages, **kw):
            if model == "ultra":
                raise ConnectionError("Token Factory unreachable")
            return super().complete(model, messages, **kw)

    task = make_task()
    report, _ = run(task, FlakyUltra.for_task(task))
    vs = report["verdict"]["vs_llm_default"]
    assert vs["n"] == 0 and vs["accuracy"] is None and vs["cost_per_1k_usd"] is None
    format_report(report)  # the text report copes with a missing cost


def test_plain_answer_lost_to_the_cap_does_not_claim_offline_mode():
    from windtunnel.budget import BudgetExceeded

    class CapOnPlain(FakeNemotron):
        def complete(self, model, messages, **kw):
            if messages[-1]["content"].startswith("I'm building this"):
                raise BudgetExceeded("spending cap reached")
            return super().complete(model, messages, **kw)

    task = make_task()
    report, _ = run(task, CapOnPlain(task))
    plain = report["plain_answer"]
    assert report["mode"] == "nemotron" and plain["available"] is False
    assert "spending cap" in plain["text"] and "offline" not in plain["text"].lower()
    assert receipt(report)["plain_llm_answer"]["detail"] == "budget exhausted"


def test_selection_bias_is_present_when_the_deciding_race_never_ran_on_the_holdout(monkeypatch):
    from windtunnel import breed

    real_race, calls = breed.race, []

    def race_stopped_in_generation_1(*args, **kwargs):
        result = real_race(*args, **kwargs)
        calls.append(result)
        if len(calls) == 2:  # the generation-1 race on the search split hits the spending cap
            result.aborted = "spending cap reached"
        return result

    monkeypatch.setattr(breed, "race", race_stopped_in_generation_1)
    report, _ = run(make_task(70), config=RaceConfig(seed=1), generations=2)
    b = report["breeding"]
    assert b["holdout_ids"] and b["bred"] and report["race"]["role"] == "generation 1"
    assert b["final_on_holdout"] is False and b["selection_bias"].startswith("present")
    assert any("optimistic until confirmed" in item for item in report["limitations"])


def test_atlas_marks_bred_designs_and_selection_bias(tmp_path):
    report, _ = run(make_task(), atlas_path=tmp_path / "outcomes.jsonl")
    records = [json.loads(line) for line in (tmp_path / "outcomes.jsonl").read_text().splitlines()]
    bred_ids = {b["id"] for b in report["breeding"]["bred"]}
    bred = [r for r in records if r["design"]["id"] in bred_ids]
    assert bred and all(r["origin"] == "bred" and r["selection_biased"] and r["race_role"].startswith("generation") for r in bred)
    assert all(r["lineage"]["parent_id"] for r in bred)
    first = [r for r in records if r["design"]["id"] not in bred_ids]
    assert first and all(r["origin"] == "candidate" and not r["selection_biased"] and r["race_role"] == "initial" for r in first)


def test_held_out_bred_records_are_not_marked_biased(tmp_path):
    report, _ = run(make_task(70), config=RaceConfig(seed=1), atlas_path=tmp_path / "outcomes.jsonl")
    records = [json.loads(line) for line in (tmp_path / "outcomes.jsonl").read_text().splitlines()]
    confirmed = [r for r in records if r["race_role"] == "confirmation (held-out split)"]
    assert report["breeding"]["final_on_holdout"] is True
    assert all(not r["selection_biased"] for r in confirmed)


def test_precomputed_research_and_candidates_are_raced_as_shown():
    from windtunnel.generate import design_candidates
    from windtunnel.research import research

    found = research(PROBLEM, None, providers=fake_providers(), fetcher=object())
    shown = design_candidates(found.spec, found.cards, None)

    def no_search(*a, **kw):
        raise AssertionError("reused research must not search again")

    report = run_wind_tunnel(PROBLEM, make_task(), None, providers={"arxiv": no_search}, fetcher=object(), breed=False,
                             research_result=found, candidates=shown)
    assert [c["id"] for c in report["candidates"]] == [c.design.id for c in shown]
    assert {d["id"] for d in report["race"]["designs"]} == {c.design.id for c in shown}
    r = receipt(report)
    assert "reused" in r["research"]["detail"] and "reused" in r["design_candidates"]["detail"]
