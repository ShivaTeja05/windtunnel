import pytest

from windtunnel.designs import MAX_INSTRUCTIONS_CHARS, Design, default_designs, run_design
from windtunnel.graph import graph_for, to_mermaid
from windtunnel.llm import Completion, MockClient
from windtunnel.task import Constraints, Example, TaskSpec


def make_task(n: int = 6) -> TaskSpec:
    labels = ["red", "green", "blue"]
    examples = [Example(f"e{i}", f"item number {i:03d}", labels[i % 3]) for i in range(n)]
    return TaskSpec("colors", "Name the color.", "Answer with one word.", "exact", examples, Constraints(), labels)


class RecordingClient:
    """Scripted replies; records (model, messages, temperature) for every call."""

    def __init__(self, replies, latency: float = 0.1):
        self.replies = list(replies)
        self.latency = latency
        self.seen: list[tuple[str, list[dict], float]] = []

    def complete(self, model, messages, *, temperature=0.0, max_tokens=1024):
        self.seen.append((model, messages, temperature))
        return Completion(self.replies.pop(0), model, 10, 5, self.latency)


def _system_prompts(client: RecordingClient) -> list[str]:
    return [m[0]["content"] for _, m, _ in client.seen]


# --- instructions -----------------------------------------------------------------


def test_instructions_reach_every_answering_prompt():
    tip = "Treat any mention of a refund as billing."
    client = RecordingClient(["ANSWER: red", "ANSWER: red"])
    run_design(Design("sc", "self_check", "nano", {"instructions": tip}), make_task(), "x", client)
    assert all(tip in s for s in _system_prompts(client))


def test_instructions_absent_keeps_prompt_identical():
    plain, empty = RecordingClient(["ANSWER: red"]), RecordingClient(["ANSWER: red"])
    run_design(Design("a", "single_call", "nano"), make_task(), "x", plain)
    run_design(Design("b", "single_call", "nano", {"instructions": "   "}), make_task(), "x", empty)
    assert _system_prompts(plain) == _system_prompts(empty)
    assert "Guidance" not in _system_prompts(plain)[0]


def test_instructions_are_capped_and_validated():
    d = Design("long", "single_call", "nano", {"instructions": "x" * 5000})
    assert len(d.instructions) == MAX_INSTRUCTIONS_CHARS
    client = RecordingClient(["ANSWER: red"])
    run_design(d, make_task(), "x", client)
    assert "x" * (MAX_INSTRUCTIONS_CHARS + 1) not in _system_prompts(client)[0]
    with pytest.raises(ValueError):
        Design("bad", "single_call", "nano", {"instructions": ["not", "a", "string"]})


def test_instructions_reach_cascade_and_workers():
    tip = "Prefer the most specific queue."
    cascade = RecordingClient(["ANSWER: red", "ANSWER: blue", "ANSWER: blue"])
    run_design(Design("c", "cascade", "nano", {"instructions": tip}), make_task(), "x", cascade)
    assert len(cascade.seen) == 3 and all(tip in s for s in _system_prompts(cascade))
    crew = RecordingClient(["q1\nq2", "f1", "f2", "ANSWER: red"])
    run_design(Design("w", "orchestrator_workers", "ultra", {"instructions": tip, "max_subtasks": 2}), make_task(), "x", crew)
    prompts = _system_prompts(crew)
    assert tip not in prompts[0]  # the split step plans, it does not answer
    assert all(tip in s for s in prompts[1:])


# --- cascade ----------------------------------------------------------------------


def test_cascade_returns_cheap_answer_when_samples_agree():
    client = RecordingClient(["ANSWER: Blue", "ANSWER: blue."], latency=0.2)
    trace = run_design(Design("c", "cascade", "nano"), make_task(), "x", client)
    assert trace.error is None
    assert trace.output == "Blue"
    assert [c.model for c in trace.calls] == ["nano", "nano"]
    assert all(t == 0.7 for _, _, t in client.seen)
    assert trace.latency_s == pytest.approx(0.2)  # parallel samples count once


def test_cascade_escalates_on_disagreement():
    client = RecordingClient(["ANSWER: red", "ANSWER: green", "ANSWER: blue"], latency=0.2)
    design = Design("c", "cascade", "nano", {"cheap_model": "lightning", "escalate_model": "ultra", "samples": 2})
    trace = run_design(design, make_task(), "x", client)
    assert trace.output == "blue"
    assert [c.model for c in trace.calls] == ["lightning", "lightning", "ultra"]
    assert client.seen[-1][2] == 0.0  # one deterministic escalation call
    assert trace.latency_s == pytest.approx(0.4)


def test_cascade_escalates_when_cheap_answer_is_empty():
    trace = run_design(Design("c", "cascade", "nano"), make_task(), "x", RecordingClient(["", "", "ANSWER: red"]))
    assert trace.output == "red" and trace.calls[-1].model == "super"


def test_cascade_validates_models_and_samples():
    with pytest.raises(KeyError):
        Design("c", "cascade", "nano", {"escalate_model": "gpt-9"})
    with pytest.raises(KeyError):
        Design("c", "cascade", "nano", {"cheap_model": "tiny"})
    with pytest.raises(ValueError):
        Design("c", "cascade", "nano", {"samples": 1})


def test_cascade_runs_on_mock_client():
    task = make_task(30)
    client = MockClient.for_task(task, skill={"nano": 0.5, "super": 1.0})
    design = next(d for d in default_designs() if d.id == "cascade-nano-super")
    traces = [run_design(design, task, e.input, client) for e in task.examples]
    assert all(t.error is None and t.output for t in traces)
    escalated = [t for t in traces if len(t.calls) == 3]
    assert escalated and len(escalated) < len(traces)  # some inputs escalate, some do not
    for t, e in zip(traces, task.examples):
        if len(t.calls) == 3:
            assert t.calls[-1].model == "super" and t.output == e.expected  # super is perfect here


# --- graph ------------------------------------------------------------------------


def test_cascade_graph_is_connected_and_tier_coloured():
    design = Design("c", "cascade", "nano", {"cheap_model": "nano", "escalate_model": "ultra", "samples": 3})
    g = graph_for(design)
    ids = {n.id for n in g.nodes}
    reachable, frontier = {"in"}, ["in"]
    while frontier:
        cur = frontier.pop()
        for e in g.edges:
            if e.src == cur and e.dst not in reachable:
                reachable.add(e.dst)
                frontier.append(e.dst)
    assert reachable == ids
    tiers = {n.id: n.model for n in g.nodes}
    assert tiers["samples"] == "nano" and tiers["escalate"] == "ultra" and tiers["agree"] is None
    assert g.models() == ["nano", "ultra"]
    branch = next(e for e in g.edges if e.dst == "escalate")
    assert branch.dashed and branch.label
    assert {"dashed"} <= set(g.to_dict()["edges"][0])


def test_cascade_mermaid_renders_dashed_branch():
    text = to_mermaid(graph_for(Design("c", "cascade", "nano")))
    assert text.startswith("flowchart LR")
    assert 'samples[["2 cheap answers · nano"]]' in text
    assert 'agree{{"Answers agree?"}}' in text
    assert 'agree -.->|"disagree"| escalate' in text
    assert 'agree -->|"agree"| out' in text
    assert "class samples nano" in text and "class escalate super" in text and "class agree io" in text


def test_default_designs_include_cascade_and_unique_ids():
    designs = default_designs()
    assert "cascade-nano-super" in {d.id for d in designs}
    assert len({d.id for d in designs}) == len(designs)
