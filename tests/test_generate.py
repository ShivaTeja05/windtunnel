import json
import re
import threading

import pytest

from windtunnel.designs import MAX_INSTRUCTIONS_CHARS
from windtunnel.extract import MethodCard
from windtunnel.generate import OFFLINE, WILDCARD, design_candidates, design_candidates_offline
from windtunnel.llm import Completion
from windtunnel.understand import understand_offline

SPEC = understand_offline("Route support tickets into billing or bug queues. Must cost under $2 per 1000.")
SLOT_RE = re.compile(r"^Design slot: (\S+)$", re.MULTILINE)


def card(key: str, family: str, relevance: int = 2, flags: list[str] | None = None, name: str = "") -> MethodCard:
    return MethodCard(
        source_key=key,
        method_name=name or f"Method {key}",
        family=family,
        mechanism=f"mechanism of {key}",
        works_when="when inputs are noisy",
        breaks_when="when labels are ambiguous",
        reported_result="",
        relevance=relevance,
        relevance_reason="",
        citation={"title": f"Paper {key}", "url": f"https://example.org/{key}"},
        verified=True,
        flags=flags or [],
        extracted_by="offline-heuristic",
    )


CARDS = [
    card("arxiv:sc", "self_consistency", 3),
    card("arxiv:chk", "self_check", 2),
    card("arxiv:crew", "orchestrator_workers", 1),
    card("doi:fault", "other", 1, ["other-field"]),
]


class SlotClient:
    """Answers each slot's call from `replies[slot]` (a list is consumed in order); records every call."""

    def __init__(self, replies: dict):
        self.replies = {k: list(v) if isinstance(v, list) else [v] for k, v in replies.items()}
        self.seen: list[tuple[str, str, list[dict]]] = []
        self._lock = threading.Lock()

    def complete(self, model, messages, *, temperature=0.0, max_tokens=1024):
        slot = SLOT_RE.search(messages[1]["content"]).group(1)
        with self._lock:
            self.seen.append((model, slot, messages))
            queue = self.replies.get(slot) or ["not json at all"]
            text = queue.pop(0) if len(queue) > 1 else queue[0]
        reply = text if isinstance(text, str) else json.dumps(text)
        return Completion(reply, model, 1000, 200, 0.1)


def reply(model, params=None, **kw) -> dict:
    return {
        "model": model,
        "params": params or {},
        "instructions": kw.pop("instructions", "Prefer the queue named in the ticket subject."),
        "rationale": "Fits noisy short tickets.",
        "based_on": kw.pop("based_on", []),
        "expected_tradeoff": "More calls for steadier answers.",
        **kw,
    }


GOOD = {
    "self_consistency": reply("nano", {"n": 7, "temperature": 0.5}, based_on=["arxiv:sc"]),
    "self_check": reply("lightning", based_on=["arxiv:chk"]),
    "orchestrator_workers": reply("super", {"worker_model": "nano", "max_subtasks": 2}, based_on=["arxiv:crew"]),
    WILDCARD: reply("nano", {"cheap_model": "nano", "samples": 3, "escalate_model": "ultra"}, family="cascade", based_on=["doi:fault"]),
}


def by_slot(cands):
    return {c.slot: c for c in cands}


def test_valid_json_path_builds_ultra_candidates():
    warnings: list[str] = []
    client = SlotClient(GOOD)
    cands = design_candidates(SPEC, CARDS, client, max_designs=8, warnings=warnings)
    assert warnings == []
    assert cands[0].role == "baseline" and cands[-1].role == "llm-default"
    slots = by_slot(cands)
    sc = slots["self_consistency"]
    assert sc.generated_by == "ultra" and sc.design.params["n"] == 7 and sc.design.params["temperature"] == 0.5
    assert sc.design.instructions == "Prefer the queue named in the ticket subject."
    assert sc.based_on == ["arxiv:sc"] and sc.cost_usd > 0 and sc.expected_tradeoff
    assert slots["self_check"].design.model == "lightning"
    crew = slots["orchestrator_workers"]
    assert crew.design.model == "super" and crew.design.params["worker_model"] == "nano"
    wild = slots[WILDCARD]
    assert wild.design.family == "cascade" and wild.design.params["escalate_model"] == "ultra"
    assert {c.generated_by for c in cands if c.role != "candidate"} == {OFFLINE}
    # references carry no generated instructions and cost nothing
    assert all(not c.design.instructions and c.cost_usd == 0 for c in cands if c.role != "candidate")
    assert len({c.design.id for c in cands}) == len(cands)
    for c in cands:
        d = c.to_dict()
        assert {"id", "family", "model", "params", "description", "graph", "based_on", "rationale", "role", "generated_by"} <= set(d)
        assert d["graph"]["design_id"] == d["id"]
        json.dumps(d)
    assert all(model == "ultra" for model, _, _ in client.seen)


def test_invalid_json_falls_back_offline_with_warning():
    warnings: list[str] = []
    client = SlotClient({**GOOD, "self_check": ["nope", "still nope"]})
    cands = design_candidates(SPEC, CARDS, client, max_designs=8, warnings=warnings)
    chk = by_slot(cands)["self_check"]
    assert chk.generated_by == OFFLINE and chk.based_on == ["arxiv:chk"]
    assert chk.cost_usd > 0  # the two rejected replies were still paid for
    assert any("self_check" in w and "offline template" in w for w in warnings)
    assert by_slot(cands)["self_consistency"].generated_by == "ultra"


@pytest.mark.parametrize(
    "bad",
    [
        reply("gpt-4"),  # unknown tier
        reply("nano", instructions=["not", "a", "string"]),
        ["a", "list"],
        reply("nano", params="n=5"),
    ],
)
def test_invalid_fields_fall_back(bad):
    warnings: list[str] = []
    cands = design_candidates(SPEC, CARDS, SlotClient({**GOOD, "self_check": bad}), max_designs=8, warnings=warnings)
    assert by_slot(cands)["self_check"].generated_by == OFFLINE
    assert any(w.startswith("design slot self_check") for w in warnings)


def test_bad_wildcard_is_dropped_with_warning():
    warnings: list[str] = []
    bad = reply("nano", family="retrieval")
    cands = design_candidates(SPEC, CARDS, SlotClient({**GOOD, WILDCARD: bad}), max_designs=8, warnings=warnings)
    assert WILDCARD not in by_slot(cands)
    assert any("wildcard" in w and "dropped" in w for w in warnings)


def test_invented_citations_are_filtered():
    sc = reply("nano", {"n": 5}, based_on=["arxiv:sc", "arxiv:made-up", "arxiv:chk", 42])
    invented_only = reply("lightning", based_on=["doi:fake"])
    cands = design_candidates(SPEC, CARDS, SlotClient({**GOOD, "self_consistency": sc, "self_check": invented_only}), max_designs=8)
    slots = by_slot(cands)
    # arxiv:chk exists but was not given to the self_consistency slot, so it is not a valid citation there
    assert slots["self_consistency"].based_on == ["arxiv:sc"]
    # nothing valid cited: credit the cards the slot was actually shown, never the invented key
    assert slots["self_check"].based_on == ["arxiv:chk"]
    provided = {c.source_key for c in CARDS}
    assert all(set(c.based_on) <= provided for c in cands)


def test_numbers_are_clamped_and_instructions_capped():
    sc = reply("nano", {"n": 50, "temperature": 3.5, "bogus": 1}, instructions="x" * 5000)
    crew = reply("super", {"worker_model": "nano", "max_subtasks": 99})
    wild = reply("nano", {"samples": 12, "temperature": -1, "escalate_model": "super"}, family="cascade")
    client = SlotClient({**GOOD, "self_consistency": sc, "orchestrator_workers": crew, WILDCARD: wild})
    slots = by_slot(design_candidates(SPEC, CARDS, client, max_designs=8))
    p = slots["self_consistency"].design.params
    assert p["n"] == 9 and p["temperature"] == 1.0 and "bogus" not in p
    assert len(p["instructions"]) <= MAX_INSTRUCTIONS_CHARS
    assert slots["orchestrator_workers"].design.params["max_subtasks"] == 4
    w = slots[WILDCARD].design.params
    assert w["samples"] == 3 and w["temperature"] == 0.0 and w["cheap_model"] == "nano"
    low = reply("nano", {"n": 1})
    sc_low = by_slot(design_candidates(SPEC, CARDS, SlotClient({**GOOD, "self_consistency": low}), max_designs=8))
    assert sc_low["self_consistency"].design.params["n"] == 3


def test_cascade_must_escalate_to_another_tier():
    warnings: list[str] = []
    same = reply("nano", {"escalate_model": "nano"}, family="cascade")
    design_candidates(SPEC, CARDS, SlotClient({**GOOD, WILDCARD: same}), max_designs=8, warnings=warnings)
    assert any("wildcard" in w for w in warnings)


def test_near_duplicates_are_dropped():
    warnings: list[str] = []
    # Same pipeline as the llm-default (instructions differ, which does not count) and as the baseline.
    crew = reply("ultra", {"worker_model": "super", "max_subtasks": 3}, instructions="Different words.")
    chk_dup = reply("nano", family="single_call")  # family is fixed per slot, so this stays self_check
    wild_dup = reply("nano", family="single_call")
    client = SlotClient({**GOOD, "orchestrator_workers": crew, "self_check": chk_dup, WILDCARD: wild_dup})
    cands = design_candidates(SPEC, CARDS, client, max_designs=8, warnings=warnings)
    sigs = [(c.design.family, c.design.model, tuple(sorted((k, v) for k, v in c.design.params.items() if k != "instructions"))) for c in cands]
    assert len(sigs) == len(set(sigs))
    assert "orchestrator_workers" not in by_slot(cands) and WILDCARD not in by_slot(cands)
    assert by_slot(cands)["self_check"].design.family == "self_check"
    assert sum("near-duplicate" in w for w in warnings) == 2


def test_same_family_and_model_get_unique_ids():
    cards = [card("a", "self_consistency", 3), card("b", "other", 1, ["long-tail"])]
    sc = reply("nano", {"n": 5})
    wild = reply("nano", {"n": 7}, family="self_consistency")
    cands = design_candidates(SPEC, cards, SlotClient({"self_consistency": sc, WILDCARD: wild}), max_designs=8)
    ids = [c.design.id for c in cands]
    assert len(ids) == len(set(ids)) and "self-consistency-nano" in ids and "self-consistency-nano-2" in ids


@pytest.mark.parametrize("max_designs", [0, 2, 3, 4, 6])
def test_reference_points_always_present_and_max_respected(max_designs):
    cards = CARDS + [card("p", "plan_execute", 2), card("s", "single_call", 2), card("c", "cascade", 2)]
    for client in (None, SlotClient(GOOD)):
        cands = design_candidates(SPEC, cards, client, max_designs=max_designs)
        roles = [c.role for c in cands]
        assert roles[0] == "baseline" and roles[-1] == "llm-default"
        assert len(cands) == max(2, max_designs)
    llm_default = design_candidates(SPEC, cards, SlotClient(GOOD), max_designs=max_designs)[-1].design
    assert llm_default.model == "ultra" and llm_default.params["worker_model"] == "super"


def test_wildcard_keeps_a_place_when_room_is_tight():
    cands = design_candidates(SPEC, CARDS, SlotClient(GOOD), max_designs=3)
    assert [c.slot for c in cands] == ["baseline", WILDCARD, "llm-default"]


def test_each_slot_is_an_isolated_call():
    markers = {slot: f"MARKER-{slot}-{i}" for i, slot in enumerate(GOOD)}
    replies = {slot: {**r, "rationale": markers[slot], "instructions": markers[slot]} for slot, r in GOOD.items()}
    client = SlotClient(replies)
    design_candidates(SPEC, CARDS, client, max_designs=8)
    assert sorted(slot for _, slot, _ in client.seen) == sorted(GOOD)
    for _, slot, messages in client.seen:
        assert [m["role"] for m in messages] == ["system", "user"]  # fresh conversation per slot
        text = "\n".join(m["content"] for m in messages)
        assert not any(marker in text for marker in markers.values())
        # each slot sees only its own cards
        others = {"self_consistency": "arxiv:sc", "self_check": "arxiv:chk", "orchestrator_workers": "arxiv:crew", WILDCARD: "doi:fault"}
        assert others[slot] in text
        assert not any(key in text for s, key in others.items() if s != slot)


def test_prompt_carries_spec_and_card_fields():
    client = SlotClient(GOOD)
    design_candidates(SPEC, CARDS, client, max_designs=8)
    _, _, messages = next(s for s in client.seen if s[1] == "self_consistency")
    user = messages[1]["content"]
    assert SPEC.summary in user and SPEC.constraints[0] in user and SPEC.success_metric in user
    for key in ("mechanism", "works_when", "breaks_when", "reported_result"):
        assert f'"{key}"' in user


def test_offline_knows_about_cascade_and_keeps_behaviour():
    cards = [card("sc", "self_consistency", 3), card("fg", "other", 2, name="FrugalGPT: cascading LLM calls"), card("crew", "orchestrator_workers", 1)]
    warnings: list[str] = []
    cands = design_candidates(SPEC, cards, None, warnings=warnings)
    assert design_candidates_offline(SPEC, cards)[1].design.id == cands[1].design.id
    slots = by_slot(cands)
    assert slots["cascade"].design.params["escalate_model"] == "super" and slots["cascade"].based_on == ["fg"]
    assert all(c.generated_by == OFFLINE and c.cost_usd == 0 for c in cands)
    assert cands[-1].based_on == ["crew"] and warnings == []
    assert "orchestrator_workers" not in slots and WILDCARD not in slots
    assert [c.design.id for c in cands] == ["baseline-nano", "self-consistency-nano", "cascade-nano", "crew-ultra"]


def test_no_supported_families_gives_only_reference_points():
    client = SlotClient({})
    cands = design_candidates(SPEC, [card("r", "retrieval", 3)], client)
    assert [c.role for c in cands] == ["baseline", "llm-default"] and client.seen == []
