import json
import re

import pytest

from windtunnel.breed import (
    FAILURE_MODES,
    MUTATIONS,
    Diagnosis,
    FailureMode,
    apply_mutation,
    breed,
    diagnose,
    load_mechanisms,
    race_with_breeding,
    signature,
    split_holdout,
)
from windtunnel.designs import Design, run_design
from windtunnel.graph import graph_for
from windtunnel.llm import Completion, MockClient
from windtunnel.racer import DesignResult, RaceConfig, RaceResult
from windtunnel.task import Constraints, Example, TaskSpec

LABELS = ["red", "green", "blue"]


def make_task(n: int = 30, constraints: Constraints | None = None) -> TaskSpec:
    examples = [Example(f"e{i}", f"item number {i:03d}", LABELS[i % 3]) for i in range(n)]
    return TaskSpec("colors", "Name the color.", "Answer with one word.", "exact", examples, constraints or Constraints(), LABELS)


class ScriptedClient:
    def __init__(self, replies):
        self.replies = list(replies)
        self.prompts = []

    def complete(self, model, messages, *, temperature=0.0, max_tokens=1024):
        self.prompts.append((model, messages))
        return Completion(self.replies.pop(0), model, 10, 5, 0.1)


def scripted_result(task: TaskSpec, design: Design, answers: dict[str, str], errors: dict[str, str] | None = None) -> DesignResult:
    """A DesignResult whose outputs are given per example id; unlisted examples are answered correctly."""
    r = DesignResult(design)
    for ex in task.examples:
        if errors and ex.id in errors:
            r.scores[ex.id], r.outputs[ex.id], r.errors[ex.id] = 0.0, "", errors[ex.id]
        else:
            out = answers.get(ex.id, ex.expected)
            r.outputs[ex.id] = out
            r.scores[ex.id] = float(out == ex.expected)
        r.latencies.append(0.5)
    r.cost_usd = 0.01
    return r


def make_race(task: TaskSpec, results: list[DesignResult], tied: list[str]) -> RaceResult:
    return RaceResult(task, RaceConfig(), results, [len(task.examples)], "t0", "t1", tied[0], tied[0], tied, "verdict")


def confused_race(task: TaskSpec) -> RaceResult:
    # e0, e3, e6, e9 are "red" -> answered "blue"; e12, e15 -> invalid "purple".
    bad = scripted_result(task, Design("bad", "single_call", "nano"), {"e0": "blue", "e3": "blue", "e6": "blue", "e9": "blue", "e12": "purple", "e15": "Purple!"})
    good = scripted_result(task, Design("good", "single_call", "ultra"), {})
    return make_race(task, [good, bad], ["good"])


def test_diagnose_finds_confusion_pairs_and_invalid_labels():
    task = make_task(30)
    diags = diagnose(confused_race(task), task)
    assert [d.design_id for d in diags] == ["bad"]  # the perfect design has nothing to diagnose
    d = diags[0]
    assert d.produced_by == "offline-heuristic"
    assert d.confusions == [("red", "blue", 4)]
    modes = {m.mode: m for m in d.failure_modes}
    assert set(modes) == {"label_confusion", "format_violation"}
    assert "expected red, answered blue: 4 times" in modes["label_confusion"].evidence
    assert modes["format_violation"].evidence[0].startswith("2 of 30 answers were not one of the task's labels")
    assert modes["label_confusion"].share == pytest.approx(4 / 30, abs=1e-3)
    assert d.failure_modes[0].mode == "label_confusion"  # ranked by share
    assert 1 <= len(d.sample_failures) <= 5
    assert {"example_id", "input", "expected", "got", "error"} <= set(d.sample_failures[0])
    json.dumps(d.to_dict())


def test_diagnose_flags_errors_constraints_and_scattered_errors():
    task = make_task(30, Constraints(max_cost_per_1k=0.1))
    slow = scripted_result(task, Design("pricey", "single_call", "ultra"), {}, errors={"e1": "TimeoutError: took too long", "e2": "TimeoutError: again"})
    slow.eliminated_round, slow.elimination_reason = 1, "over budget: $0.33/1k > $0.10/1k"
    noisy = scripted_result(task, Design("noisy", "single_call", "nano"), {"e0": "blue", "e1": "red", "e2": "green", "e3": "green"})
    noisy.cost_usd = 0.0
    good = scripted_result(task, Design("good", "single_call", "nano"), {})
    good.cost_usd = 0.0
    diags = {d.design_id: d for d in diagnose(make_race(task, [good, noisy, slow], ["good", "noisy"]), task)}
    assert list(diags) == ["pricey", "noisy"]  # eliminated losers first, then tied designs with failures
    pricey = {m.mode: m for m in diags["pricey"].failure_modes}
    assert set(pricey) == {"over_budget", "pipeline_error"}
    assert "TimeoutError x2" in pricey["pipeline_error"].evidence[0]
    assert pricey["over_budget"].evidence == ["over budget: $0.33/1k > $0.10/1k"]
    noisy_modes = {m.mode: m for m in diags["noisy"].failure_modes}
    assert set(noisy_modes) == {"inconsistent"}  # 4 one-off confusions, none repeated
    assert "proxy" in noisy_modes["inconsistent"].evidence[0]


def test_diagnose_merges_model_modes_from_taxonomy_only():
    task = make_task(30)
    reply = json.dumps(
        {
            "failure_modes": [
                {"mode": "label_confusion", "evidence": "Crimson items read as blue."},
                {"mode": "inconsistent", "evidence": "Answers flip between runs."},
                {"mode": "made_up_mode", "evidence": "ignored"},
            ]
        }
    )
    client = ScriptedClient([reply])
    (d,) = diagnose(confused_race(task), task, client)
    assert client.prompts[0][0] == "ultra"
    prompt = client.prompts[0][1][1]["content"]
    assert "expected red, answered blue" in prompt and "format_violation" in prompt
    assert d.produced_by == "offline-heuristic+ultra"
    modes = {m.mode: m for m in d.failure_modes}
    assert "made_up_mode" not in modes
    assert modes["label_confusion"].flagged_by == ["offline-heuristic", "ultra"]
    assert modes["label_confusion"].evidence[-1] == "(ultra) Crimson items read as blue."
    assert modes["inconsistent"].flagged_by == ["ultra"] and modes["inconsistent"].share == 0.0
    assert d.cost_usd > 0


def test_diagnose_unparseable_or_mock_reply_stays_offline():
    task = make_task(30)
    (d,) = diagnose(confused_race(task), task, ScriptedClient(["no json here", "still none"]))
    assert d.produced_by == "offline-heuristic (ultra reply was not valid JSON)"
    assert all(m.flagged_by == ["offline-heuristic"] for m in d.failure_modes)
    # A mock client is never labelled as a real model.
    (d,) = diagnose(confused_race(task), task, MockClient.for_task(task))
    assert "mock-ultra" in d.produced_by and not d.produced_by.endswith("+ultra")


def test_mechanisms_load_and_every_mutation_is_applicable():
    mechs = load_mechanisms()
    assert 10 <= len(mechs) <= 14
    assert {m.op for m in mechs} == set(MUTATIONS)
    assert set().union(*(m.failure_modes for m in mechs)) == set(FAILURE_MODES)  # every mode has a fix
    assert len({m.field for m in mechs}) >= 6  # genuinely cross-domain
    bases = [
        Design("single-nano", "single_call", "nano"),
        Design("single-ultra", "single_call", "ultra"),
        Design("vote3", "self_consistency", "super", {"n": 3}),
        Design("cascade", "cascade", "nano", {"cheap_model": "nano", "escalate_model": "super"}),
    ]
    task = make_task(3)
    for m in mechs:
        assert m.source is None or m.source["url"].startswith("https://")
        children = [
            apply_mutation(b, m.mutation, new_id=f"{b.id}.{m.id}", labels=LABELS, confusions=[("red", "blue", 3)])
            for b in bases
        ]
        applied = [c for c in children if c is not None]
        assert applied, f"{m.id} applies to none of the base designs"
        for child in applied:
            graph_for(child)
            assert not run_design(child, task, task.examples[0].input, MockClient.for_task(task)).error


def test_load_mechanisms_rejects_unknown_mutations(tmp_path):
    row = {"id": "x", "name": "X", "field": "f", "failure_modes": ["inconsistent"], "idea": "i", "source": None}
    path = tmp_path / "m.json"
    path.write_text(json.dumps([{**row, "mutation": {"op": "teleport"}}]))
    with pytest.raises(ValueError, match="unknown mutation"):
        load_mechanisms(path)
    path.write_text(json.dumps([{**row, "mutation": {"op": "add_voting"}}]))
    with pytest.raises(ValueError, match="takes"):
        load_mechanisms(path)
    path.write_text(json.dumps([{**row, "failure_modes": ["gremlins"], "mutation": {"op": "add_plan"}}]))
    with pytest.raises(ValueError, match="unknown failure modes"):
        load_mechanisms(path)


def test_apply_mutation_skips_no_ops():
    assert apply_mutation(Design("a", "self_check", "nano"), {"op": "add_self_check"}, new_id="b") is None
    assert apply_mutation(Design("a", "single_call", "ultra"), {"op": "upgrade_tier"}, new_id="b") is None
    assert apply_mutation(Design("a", "single_call", "nano"), {"op": "downgrade_tier"}, new_id="b") is None
    assert apply_mutation(Design("a", "self_consistency", "nano", {"n": 7}), {"op": "add_voting", "n": 5}, new_id="b") is None
    # The {confusions} placeholder needs diagnosed confusions.
    assert apply_mutation(Design("a", "single_call", "nano"), {"op": "add_instructions", "text": "Watch {confusions}."}, new_id="b") is None
    up = apply_mutation(Design("a", "cascade", "nano", {"cheap_model": "nano", "escalate_model": "super"}), {"op": "upgrade_tier"}, new_id="b")
    assert up.params["escalate_model"] == "ultra" and up.params["cheap_model"] == "nano"
    down = apply_mutation(Design("a", "orchestrator_workers", "super", {"worker_model": "super"}), {"op": "downgrade_tier"}, new_id="b")
    assert down.model == "nano" and down.params["worker_model"] == "nano"
    kept = apply_mutation(Design("a", "single_call", "nano", {"instructions": "Be terse."}), {"op": "add_plan"}, new_id="b")
    assert kept.family == "plan_execute" and kept.instructions == "Be terse."


def label_diagnosis(design_id: str = "single-nano") -> Diagnosis:
    return Diagnosis(
        design_id=design_id,
        accuracy=0.6,
        n=30,
        failure_modes=[
            FailureMode("label_confusion", ["expected red, answered blue: 4 times"], ["offline-heuristic"], 0.13),
            FailureMode("format_violation", ["2 of 30 answers were not one of the task's labels"], ["offline-heuristic"], 0.07),
        ],
        sample_failures=[],
        produced_by="offline-heuristic",
        confusions=[("red", "blue", 4)],
        labels=LABELS,
    )


def test_breed_produces_valid_unique_designs():
    parent = Design("single-nano", "single_call", "nano")
    kids = breed(parent, label_diagnosis(), limit=2)
    assert len(kids) == 2
    # Round-robin: one fix per diagnosed mode before a second fix for any mode.
    assert [k.failure_mode for k in kids] == ["label_confusion", "format_violation"]
    for k in kids:
        assert k.parent_id == "single-nano"
        assert k.design.id == f"single-nano.{k.mechanism_id}"
        assert re.fullmatch(r"[\w.-]+", k.design.id)  # readable by the advisor's PICK parser
        assert k.mechanism_name in k.design.rationale and k.mechanism_field in k.design.rationale
        assert k.graph.design_id == k.design.id and k.graph.to_dict()["nodes"]
        assert k.generated_by == "offline-heuristic"
        json.dumps(k.to_dict())

    everything = breed(parent, label_diagnosis(), limit=20)
    sigs = [signature(k.design) for k in everything]
    assert len(sigs) == len(set(sigs)) and signature(parent) not in sigs
    assert len({k.design.id for k in everything}) == len(everything)
    by_mech = {k.mechanism_id: k for k in everything}
    assert "red vs blue" in by_mech["differential-diagnosis"].design.instructions
    assert "red, green, blue" in by_mech["poka-yoke"].design.instructions

    taken = breed(parent, label_diagnosis(), limit=1, taken_ids={kids[0].design.id})
    assert taken[0].design.id == kids[0].design.id + "-2"


def test_breed_cost_failures_yield_cheaper_designs():
    parent = Design("single-ultra", "single_call", "ultra")
    diag = Diagnosis("single-ultra", 0.9, 30, [FailureMode("over_budget", ["over budget"], ["offline-heuristic"], 1.0)], [], "offline-heuristic")
    kids = breed(parent, diag, limit=5)
    assert {k.mechanism_id for k in kids} == {"triage", "load-shedding"}
    by_mech = {k.mechanism_id: k.design for k in kids}
    assert by_mech["triage"].family == "cascade" and by_mech["triage"].params["cheap_model"] == "nano"
    assert by_mech["load-shedding"].model == "super"


def test_split_holdout():
    search, held = split_holdout(make_task(40), fraction=0.3, min_holdout=20)
    assert held is None and len(search.examples) == 40
    search, held = split_holdout(make_task(100), fraction=0.3, min_holdout=20)
    assert len(held.examples) == 30 and len(search.examples) == 70
    assert not {e.id for e in held.examples} & {e.id for e in search.examples}


def test_race_with_breeding_end_to_end_without_holdout():
    task = make_task(30)
    client = MockClient.for_task(task, skill={"nano": 0.4, "ultra": 1.0})
    designs = [Design("single-nano", "single_call", "nano"), Design("single-ultra", "single_call", "ultra")]
    res = race_with_breeding(task, designs, client, RaceConfig(schedule=(10, 20), seed=3), generations=1, breed_per_generation=3)
    assert res.provenance == "mock"
    assert res.race_roles == ["initial", "generation 1"]
    assert res.holdout_ids == [] and len(res.search_ids) == 30
    assert "selection bias" in res.note and "optimistic" in res.note and "Mock run" in res.note
    assert res.diagnoses[0] and res.diagnoses[0][0].design_id == "single-nano"
    assert res.diagnoses[0][0].produced_by == "offline-heuristic"  # the mock never poses as Ultra
    assert 1 <= len(res.bred) <= 3
    raced = {r.design.id for r in res.final.results}
    for kid in res.bred:
        assert kid.design.id in raced
        assert res.lineage[kid.design.id]["parent_id"] == kid.parent_id
        assert res.lineage[kid.design.id]["generation"] == 1
    assert res.final.config.seed != res.races[0].config.seed
    assert res.recommended_id in raced
    json.dumps(res.to_dict())


def test_race_with_breeding_uses_held_out_split_when_task_is_big_enough():
    task = make_task(90)
    client = MockClient.for_task(task, skill={"nano": 0.4, "ultra": 1.0})
    designs = [Design("single-nano", "single_call", "nano"), Design("single-ultra", "single_call", "ultra")]
    res = race_with_breeding(task, designs, client, RaceConfig(schedule=(10, 30)), generations=2, min_holdout=20)
    assert len(res.holdout_ids) == 27 and len(res.search_ids) == 63
    assert res.race_roles[0] == "initial (search split)" and res.race_roles[-1] == "confirmation (held-out split)"
    assert {e.id for e in res.races[0].task.examples} == set(res.search_ids)
    assert {e.id for e in res.final.task.examples} == set(res.holdout_ids)
    assert "held-out" in res.note and "selection bias" not in res.note
    assert res.bred
    gens = {v["generation"] for v in res.lineage.values()}
    assert gens <= {1, 2} and 1 in gens


@pytest.mark.parametrize(
    "reply",
    [
        {"failure_modes": [{"mode": ["label_confusion"], "evidence": "x"}, {"mode": {"k": 1}}]},
        {"failure_modes": {"mode": "label_confusion", "evidence": "a single object, not a list"}},
        {"failure_modes": 3},
        ["label_confusion"],
    ],
)
def test_diagnose_survives_wrong_shaped_replies(reply):
    task = make_task(30)
    (d,) = diagnose(confused_race(task), task, ScriptedClient([json.dumps(reply)]))
    assert d.produced_by.startswith("offline-heuristic")
    assert all(isinstance(m.mode, str) for m in d.failure_modes)


def test_diagnose_falls_back_on_budget_and_api_errors_and_counts_failed_spend():
    from windtunnel.budget import BudgetExceeded

    class Raises:
        def __init__(self, exc):
            self.exc = exc

        def complete(self, *a, **kw):
            raise self.exc

    task = make_task(30)
    (d,) = diagnose(confused_race(task), task, Raises(BudgetExceeded("cap")))
    assert "budget exhausted" in d.produced_by
    (d,) = diagnose(confused_race(task), task, Raises(TimeoutError("slow")))
    assert "call failed: TimeoutError" in d.produced_by
    # Two unparseable replies were paid for even though they were unusable.
    (d,) = diagnose(confused_race(task), task, ScriptedClient(["no json", "still none"]))
    assert d.cost_usd == pytest.approx(2 * Completion("", "ultra", 10, 5, 0.1).cost_usd)


def test_wrapped_mock_is_still_labelled_mock():
    class Counting:  # the shape of the MCP server's counting wrapper
        def __init__(self, client):
            self.client = client

        def complete(self, *a, **kw):
            return self.client.complete(*a, **kw)

    task = make_task(30)
    designs = [Design("nano", "single_call", "nano"), Design("super", "single_call", "super")]
    out = race_with_breeding(task, designs, Counting(MockClient.for_task(task)), RaceConfig(schedule=(10, 30)))
    assert out.provenance == "mock" and "Mock run" in out.note
    assert all("+ultra" not in d.produced_by for gen in out.diagnoses for d in gen)


def test_no_confirmation_race_when_nothing_survived_the_constraints():
    task = make_task(90, Constraints(max_cost_per_1k=1e-9))
    designs = [Design("nano", "single_call", "nano"), Design("ultra", "single_call", "ultra")]
    out = race_with_breeding(task, designs, MockClient.for_task(task), RaceConfig(), mechanisms=[])
    assert len(out.races) == 1 and out.recommended_id is None
    assert "no confirmation race" in out.note


def test_single_survivor_confirmation_measures_every_held_out_example():
    task = make_task(90)
    designs = [Design("good", "single_call", "ultra"), Design("bad", "single_call", "super")]
    client = MockClient.for_task(task, skill={"ultra": 1.0, "super": 0.0})
    out = race_with_breeding(task, designs, client, RaceConfig(), mechanisms=[])
    assert out.race_roles[-1] == "confirmation (held-out split)"
    assert out.final.get("good").n == len(out.holdout_ids) == 27
    assert "measured on" not in out.note  # no gap between what the note claims and what was measured
