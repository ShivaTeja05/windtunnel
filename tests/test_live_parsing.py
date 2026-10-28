"""Regressions for reply shapes real models produce but the mock never does."""

from windtunnel.advisor import llm_pick
from windtunnel.designs import Design, extract_answer, infra_kind, run_design
from windtunnel.llm import Completion
from windtunnel.racer import RaceConfig, race
from windtunnel.structured import ask_json, extract_json
from windtunnel.task import Example, TaskSpec


class Scripted:
    def __init__(self, replies):
        self.replies = list(replies)

    def complete(self, model, messages, *, temperature=0.0, max_tokens=1024):
        return Completion(self.replies.pop(0), model, 10, 5, 0.1)


def test_markdown_answer_lines_parse():
    assert extract_answer("Reasoning...\n**ANSWER:** billing") == "billing"
    assert extract_answer("**ANSWER: account_access**") == "account_access"
    assert extract_answer("> ANSWER: `shipping`") == "shipping"
    assert extract_answer("ANSWER: C#") == "C#"
    assert extract_answer("**billing**") == "billing"


def test_markdown_pick_line_parses():
    task = TaskSpec("t", "d", "", "exact", [Example("1", "x", "a")])
    designs = [Design("a", "single_call", "nano"), Design("b.quorum", "single_call", "super")]
    pick = llm_pick(task, designs, Scripted(["I'd go with b.\n**PICK:** b.quorum"]))
    assert pick.design_id == "b.quorum"


def test_citation_marker_is_not_taken_for_the_json_reply():
    assert extract_json('As shown in [1], the plan is {"summary": "s"}', dict) == {"summary": "s"}
    data, calls = ask_json(Scripted(['See [1].', '{"ok": true}']), "ultra", "sys", "user")
    assert data == {"ok": True} and len(calls) == 2  # the retry fired


def test_missing_model_and_bad_key_are_infrastructure_failures():
    class NotFoundError(Exception):
        pass

    class AuthenticationError(Exception):
        pass

    assert infra_kind(NotFoundError("model not served")) == "provider"
    assert infra_kind(AuthenticationError("bad key")) == "provider"

    class Missing:
        def complete(self, *a, **k):
            raise NotFoundError("model not served")

    task = TaskSpec("t", "d", "", "exact", [Example("1", "x", "a")])
    trace = run_design(Design("s", "single_call", "nano"), task, "x", Missing())
    assert trace.infra == "provider"


def test_exact_accuracy_tie_goes_to_the_cheaper_design():
    examples = [Example(str(i), f"item {i}", "a") for i in range(12)]
    task = TaskSpec("t", "d", "", "exact", examples)

    class AlwaysRight:
        def complete(self, model, messages, **kw):
            return Completion("ANSWER: a", model, 100, 10, 0.1)

    designs = [Design("pricey", "single_call", "ultra"), Design("cheap", "single_call", "nano")]
    result = race(task, designs, AlwaysRight(), RaceConfig(mode="grid"))
    assert result.best_id == "cheap" and result.recommended_id == "cheap"
