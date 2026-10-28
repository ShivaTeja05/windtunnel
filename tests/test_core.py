import json

import pytest

from windtunnel.advisor import llm_pick
from windtunnel.cli import main
from windtunnel.designs import Design, extract_answer, run_design
from windtunnel.llm import Completion, MockClient
from windtunnel.racer import RaceConfig, race
from windtunnel.scoring import exact, numeric
from windtunnel.stats import bootstrap_ci, paired_diff_ci
from windtunnel.task import Constraints, Example, TaskSpec


def make_task(n: int = 40, constraints: Constraints | None = None) -> TaskSpec:
    labels = ["red", "green", "blue"]
    examples = [Example(f"e{i}", f"item number {i:03d}", labels[i % 3]) for i in range(n)]
    return TaskSpec("colors", "Name the color.", "Answer with one word.", "exact", examples, constraints or Constraints(), labels)


def test_scorers():
    assert exact("Account Access.", "account_access") == 1.0
    assert exact("billing", "bug") == 0.0
    assert numeric("The total is 1,234.50 dollars", "1234.5") == 1.0
    assert numeric("no number", "3") == 0.0


def test_extract_answer_takes_last_answer_line():
    assert extract_answer("thinking...\nANSWER: red\nmore\nanswer:  blue ") == "blue"
    assert extract_answer("just this line") == "just this line"


def test_bootstrap_ci_brackets_mean():
    lo, hi = bootstrap_ci([0, 1] * 50)
    assert lo < 0.5 < hi
    lo, hi = paired_diff_ci([0.0] * 30, [1.0] * 30)
    assert hi < 0


class ScriptedClient:
    def __init__(self, replies):
        self.replies = list(replies)

    def complete(self, model, messages, *, temperature=0.0, max_tokens=1024):
        return Completion(self.replies.pop(0), model, 10, 5, 0.1)


def test_self_consistency_majority_vote():
    task = make_task(3)
    design = Design("v", "self_consistency", "nano", {"n": 3})
    trace = run_design(design, task, "x", ScriptedClient(["ANSWER: red", "ANSWER: Blue", "ANSWER: blue"]))
    assert trace.output.lower() == "blue"
    assert len(trace.calls) == 3


def test_failed_pipeline_is_recorded_not_raised():
    trace = run_design(Design("s", "single_call", "nano"), make_task(3), "x", ScriptedClient([]))
    assert trace.error and trace.output == ""


def test_race_eliminates_losers_and_prefers_cheaper_tie():
    task = make_task(60)
    client = MockClient.for_task(task, skill={"nano": 1.0, "ultra": 1.0, "super": 0.0})
    designs = [
        Design("nano", "single_call", "nano"),
        Design("ultra", "single_call", "ultra"),
        Design("broken", "single_call", "super"),
    ]
    result = race(task, designs, client, RaceConfig(schedule=(10, 30), max_workers=4))
    assert not result.get("broken").alive
    assert result.get("broken").n < 60  # eliminated before seeing every example
    assert set(result.tied_ids) == {"nano", "ultra"}
    assert result.recommended_id == "nano"  # same accuracy, cheaper


def test_race_spends_less_than_grid():
    task = make_task(60)
    designs = [Design("good", "single_call", "ultra"), Design("bad", "single_call", "super")]
    client = MockClient.for_task(task, skill={"ultra": 1.0, "super": 0.1})
    raced = race(task, designs, client, RaceConfig(schedule=(10, 30)))
    grid = race(task, designs, MockClient.for_task(task, skill={"ultra": 1.0, "super": 0.1}), RaceConfig(mode="grid"))
    assert raced.recommended_id == grid.recommended_id == "good"
    assert raced.total_calls < grid.total_calls


def test_constraints_eliminate_over_budget_designs():
    task = make_task(20, Constraints(max_cost_per_1k=0.0001))
    result = race(task, [Design("u", "single_call", "ultra")], MockClient.for_task(task), RaceConfig())
    assert result.recommended_id is None
    assert "over budget" in result.get("u").elimination_reason


def test_advisor_parses_pick():
    task = make_task(3)
    designs = [Design("a", "single_call", "nano"), Design("b", "plan_execute", "super")]
    pick = llm_pick(task, designs, MockClient.for_task(task))
    assert pick.design_id == "a"


@pytest.mark.parametrize("mode", ["race", "grid"])
def test_cli_mock_end_to_end(tmp_path, mode):
    task_dir = tmp_path / "task"
    task_dir.mkdir()
    (task_dir / "task.json").write_text(json.dumps({"name": "t", "description": "Name the color.", "labels": ["red", "blue"]}))
    (task_dir / "examples.jsonl").write_text(
        "\n".join(json.dumps({"input": f"thing {i:02d}", "expected": "red" if i % 2 else "blue"}) for i in range(30))
    )
    atlas = tmp_path / "atlas.jsonl"
    assert main(["race", str(task_dir), "--mock", "--mode", mode, "--out", str(tmp_path / "runs"), "--atlas", str(atlas)]) == 0
    run = json.loads(next((tmp_path / "runs").glob("*.json")).read_text())
    assert run["provenance"] == "mock" and run["recommended_id"]
    assert len(atlas.read_text().splitlines()) == len(run["designs"])
