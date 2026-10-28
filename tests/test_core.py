import json

import pytest

from windtunnel.advisor import llm_pick
from windtunnel.cli import main
from windtunnel.designs import Design, extract_answer, run_design
from windtunnel.llm import Completion, MockClient
from windtunnel.racer import RaceConfig, race
from windtunnel.scoring import contains, exact, numeric
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


# ---------------------------------------------------------------- scoring and statistics fixes


def test_scorers_parse_formatted_numbers_and_keep_signs():
    assert numeric(".5", "0.5") == numeric("1e3", "1000") == numeric("1000", "1,000") == numeric("$12", "$12") == 1.0
    assert numeric("12", "12%") == 1.0
    with pytest.raises(ValueError):
        numeric("3", "three")  # a label that states no number is an error, not a silent 0
    assert exact("-1", "1") == 0.0 and exact("C++", "C#") == 0.0 and exact("-1", "-1") == 1.0
    assert exact("account-access", "account_access") == 1.0
    assert contains("anything at all", "") == 0.0 and exact("", "") == 0.0  # an empty label matches nothing


class OracleClient:
    """Right on every example except the ones listed per model; counts calls."""

    def __init__(self, task: TaskSpec, wrong: dict[str, set[str]] | None = None, fail: dict[str, Exception] | None = None):
        self.by_input = {e.input: e for e in task.examples}
        self.wrong, self.fail, self.calls = wrong or {}, fail or {}, 0

    def complete(self, model, messages, *, temperature=0.0, max_tokens=1024):
        self.calls += 1
        if model in self.fail:
            raise self.fail[model]
        ex = next(e for inp, e in self.by_input.items() if inp in messages[-1]["content"])
        answer = "nonsense" if ex.id in self.wrong.get(model, set()) else ex.expected
        return Completion(f"ANSWER: {answer}", model, 10, 5, 0.1)


def test_race_rejects_unusable_tasks_before_spending():
    task = make_task(5)
    client = OracleClient(task)
    bad = TaskSpec("n", "Count.", "", "numeric", [Example("a", "x 1", "1,234"), Example("b", "x 2", "three")])
    with pytest.raises(ValueError, match="'b'"):
        race(bad, [Design("s", "single_call", "nano")], client)
    with pytest.raises(ValueError, match="no examples"):
        race(TaskSpec("e", "d", "", "exact", []), [Design("s", "single_call", "nano")], client)
    assert client.calls == 0


def test_accuracy_ci_is_never_zero_width():
    from windtunnel.stats import mean_ci

    lo, hi = mean_ci([1.0] * 20)
    assert 0.8 < lo < 0.86 and hi == 1.0  # Wilson 20/20, not (1.0, 1.0)
    lo, hi = mean_ci([1.0])
    assert lo < 0.5


def test_single_design_race_measures_every_example():
    task = make_task(60)
    result = race(task, [Design("only", "single_call", "nano")], OracleClient(task), RaceConfig(schedule=(10, 25)))
    assert result.get("only").n == 60


def test_small_samples_are_not_called_clear_winners():
    # 10 examples, 3 discordant pairs all one way: an exact test gives p=0.25, so nothing is eliminated.
    task = make_task(10)
    ids = [e.id for e in task.examples]
    client = OracleClient(task, wrong={"nano": set(ids[:3])})
    designs = [Design("nano", "single_call", "nano"), Design("ultra", "single_call", "ultra")]
    result = race(task, designs, client, RaceConfig(schedule=(10,)))
    assert result.get("nano").alive
    assert result.recommended_id == "ultra" and result.tied_ids == ["ultra"] and result.inconclusive_ids == ["nano"]
    assert "Inconclusive" in result.verdict


def test_tie_needs_non_inferiority_not_just_no_difference():
    # 10/10 vs 8/10: no significant difference, but a 20-point gap is not "tied".
    task = make_task(10)
    client = OracleClient(task, wrong={"nano": {task.examples[0].id, task.examples[1].id}})
    designs = [Design("nano", "single_call", "nano"), Design("ultra", "single_call", "ultra")]
    result = race(task, designs, client, RaceConfig(mode="grid"))
    assert result.recommended_id == "ultra" and "nano" not in result.tied_ids
    # With plenty of identical answers the cheap design is shown to be within the margin and wins.
    big = make_task(200)
    result = race(big, designs, OracleClient(big), RaceConfig(mode="grid"))
    assert result.recommended_id == "nano" and set(result.tied_ids) == {"nano", "ultra"}
    assert "tie margin" in result.verdict


def test_elimination_is_corrected_for_repeated_checks():
    # 9/10 vs 6/10 eliminated at round 1 under the old uncorrected bootstrap; now it needs more evidence.
    task = make_task(10)
    ids = [e.id for e in task.examples]
    client = OracleClient(task, wrong={"nano": set(ids[:4]), "ultra": {ids[0]}})
    designs = [Design("nano", "single_call", "nano"), Design("ultra", "single_call", "ultra")]
    result = race(task, designs, client, RaceConfig(schedule=(10,)))
    assert result.get("nano").alive


def test_budget_refusal_stops_the_race_instead_of_scoring_zero():
    from windtunnel.budget import BudgetExceeded

    task = make_task(30)
    client = OracleClient(task, fail={"ultra": BudgetExceeded("cap")})
    designs = [Design("nano", "single_call", "nano"), Design("ultra", "single_call", "ultra")]
    result = race(task, designs, client, RaceConfig(schedule=(10, 30)))
    ultra = result.get("ultra")
    assert result.aborted and "spending cap" in result.aborted
    assert result.recommended_id is None and "no recommendation" in result.verdict
    assert ultra.n == 0 and len(ultra.unmeasured) == 10 and not ultra.errors and ultra.alive
    assert result.get("nano").n == 10  # the race stopped after the round that hit the cap


def test_provider_errors_are_left_out_not_scored_wrong():
    class RateLimitError(Exception):  # same name as openai's, raised once retries are exhausted
        pass

    task = make_task(30)
    client = OracleClient(task, wrong={"nano": {e.id for e in task.examples}}, fail={"ultra": RateLimitError("429")})
    designs = [Design("nano", "single_call", "nano"), Design("ultra", "single_call", "ultra")]
    result = race(task, designs, client, RaceConfig(mode="grid"))
    ultra = result.get("ultra")
    assert not result.aborted and ultra.n == 0 and len(ultra.unmeasured) == 30
    assert result.recommended_id == "nano"  # the only design that was measured
    assert "worse" not in (ultra.elimination_reason or "")


def test_cli_mock_race_has_no_advisor_and_is_labelled(tmp_path, capsys):
    task_dir = tmp_path / "task"
    task_dir.mkdir()
    (task_dir / "task.json").write_text(json.dumps({"name": "t", "description": "Name the color.", "labels": ["red", "blue"]}))
    (task_dir / "examples.jsonl").write_text(
        "\n".join(json.dumps({"input": f"thing {i:02d}", "expected": "red" if i % 2 else "blue"}) for i in range(30))
    )
    assert main(["race", str(task_dir), "--mock", "--out", str(tmp_path / "runs"), "--atlas", str(tmp_path / "a.jsonl")]) == 0
    out = capsys.readouterr().out
    assert "LLM advisor" not in out and "measured recommendation" not in out and out.startswith("[MOCK")
    run = json.loads(next((tmp_path / "runs").glob("*.json")).read_text())
    assert run["advisor"] is None


def test_tied_verdict_reports_the_gap_and_its_interval():
    task = make_task(200)
    client = OracleClient(task, wrong={"nano": {task.examples[0].id}})
    designs = [Design("nano", "single_call", "nano"), Design("ultra", "single_call", "ultra")]
    result = race(task, designs, client, RaceConfig(mode="grid"))
    assert result.best_id == "ultra" and result.recommended_id == "nano"
    assert "accuracy 0.99 vs 1.00, difference" in result.verdict and "200 shared examples" in result.verdict
