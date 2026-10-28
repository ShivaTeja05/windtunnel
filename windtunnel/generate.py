"""L4 Design: turn method cards into candidate designs that can race.

Every candidate records which method cards it is based on, and comes with the
architecture graph that the diagrams are drawn from.

With a client, Nemotron Ultra designs one candidate per runnable family that the
research supports, plus one wildcard slot fed the other-field and unusual cards. Each
slot is a separate, isolated call (no shared conversation) so earlier designs can't
anchor later ones. Ultra's JSON is validated strictly: tiers must exist, numbers are
clamped, and `based_on` keeps only card keys the slot was actually given. A slot whose
reply fails validation falls back to the offline template and adds a warning.

Offline mode maps the families found in the research onto runnable templates. Both
modes always add two reference points: the cheapest baseline (single Nano call) and
the design a plain LLM usually recommends (an Ultra orchestrator with Super workers).
"""

from __future__ import annotations

import json
import math
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace

from .designs import FAMILY_SUMMARIES, MAX_INSTRUCTIONS_CHARS, Design
from .extract import MethodCard
from .graph import DesignGraph, graph_for
from .llm import Completion, LLMClient
from .models import CATALOG
from .structured import ask_json
from .understand import ProblemSpec, has_keyword

OFFLINE = "offline-heuristic"
WILDCARD = "wildcard"

# Card families that map onto a runnable template, with the tier each starts on.
RUNNABLE = {
    "single_call": ("single_call", "super", {}),
    "self_check": ("self_check", "nano", {}),
    "self_consistency": ("self_consistency", "nano", {"n": 5, "temperature": 0.7}),
    "plan_execute": ("plan_execute", "super", {}),
    "orchestrator_workers": ("orchestrator_workers", "ultra", {"worker_model": "super", "max_subtasks": 3}),
    "cascade": ("cascade", "nano", {"cheap_model": "nano", "samples": 2, "temperature": 0.7, "escalate_model": "super"}),
}

# Extract does not label cascade cards yet, so recognise them from the card text too.
CASCADE_KEYWORDS = ["cascad", "frugalgpt", "model rout", "llm rout", "learning to defer", "model escalat"]

# Tunable params per family: ("tier",) or (kind, low, high). Numbers outside the range are clamped.
PARAM_RULES: dict[str, dict[str, tuple]] = {
    "self_consistency": {"n": ("int", 3, 9), "temperature": ("float", 0.0, 1.0)},
    "orchestrator_workers": {"worker_model": ("tier",), "max_subtasks": ("int", 2, 4)},
    "cascade": {
        "cheap_model": ("tier",),
        "samples": ("int", 2, 3),
        "temperature": ("float", 0.0, 1.0),
        "escalate_model": ("tier",),
    },
}

PARAM_HINTS = {
    "single_call": "{} (no tunable params)",
    "self_check": "{} (no tunable params)",
    "self_consistency": '{"n": number of samples 3-9, "temperature": 0-1}',
    "plan_execute": "{} (no tunable params)",
    "orchestrator_workers": '{"worker_model": tier for the workers, "max_subtasks": 2-4}',
    "cascade": '{"cheap_model": tier sampled first, "samples": 2-3, "temperature": 0-1, '
    '"escalate_model": a different, stronger tier used only when the cheap answers disagree}',
}

# Shown when a candidate comes from a template rather than from Ultra.
OFFLINE_TRADEOFFS = {
    "single_call": "One call per input; cheapest and fastest, with no second look at mistakes.",
    "self_check": "Two calls per input; can catch slips, but a model often repeats its own mistake.",
    "self_consistency": "n sampled calls per input; reduces noisy answers at n times the cost.",
    "plan_execute": "Two sequential calls; helps multi-step inputs, adds latency on simple ones.",
    "orchestrator_workers": "Several calls including the most expensive tier; slow and costly per input.",
    "cascade": "Two cheap calls on most inputs, plus a stronger call only when they disagree.",
}

_FIELD_CHARS = 400  # per card field in the prompt
_TEXT_CHARS = 400  # rationale / trade-off kept from the reply
_CARDS_PER_SLOT = 4


@dataclass
class CandidateDesign:
    design: Design
    graph: DesignGraph
    based_on: list[str] = field(default_factory=list)  # MethodCard.source_key values
    rationale: str = ""
    role: str = "candidate"  # "candidate", "baseline" or "llm-default"
    generated_by: str = OFFLINE  # a model key such as "ultra", or "offline-heuristic"
    cost_usd: float = 0.0  # model spend to produce this candidate (0 for templates)
    expected_tradeoff: str = ""
    slot: str = ""  # the family slot (or "wildcard", "baseline", "llm-default") it filled

    def to_dict(self) -> dict:
        d = self.design
        return {
            "id": d.id,
            "family": d.family,
            "model": d.model,
            "params": d.params,
            "description": d.describe(),
            "graph": self.graph.to_dict(),
            "based_on": self.based_on,
            "rationale": self.rationale,
            "role": self.role,
            "generated_by": self.generated_by,
            "cost_usd": round(self.cost_usd, 6),
            "expected_tradeoff": self.expected_tradeoff,
            "slot": self.slot,
        }


def _candidate(design: Design, **kw) -> CandidateDesign:
    return CandidateDesign(design=design, graph=graph_for(design), **kw)


# --- reference points and slots ----------------------------------------------------


def _baseline() -> CandidateDesign:
    return _candidate(
        Design("baseline-nano", "single_call", "nano"),
        rationale="Cheapest possible design: every other candidate must beat it to be worth its cost.",
        role="baseline",
        expected_tradeoff=OFFLINE_TRADEOFFS["single_call"],
        slot="baseline",
    )


def _llm_default(cards: list[MethodCard]) -> CandidateDesign:
    return _candidate(
        Design("crew-ultra", "orchestrator_workers", "ultra", {"worker_model": "super", "max_subtasks": 3}),
        based_on=[c.source_key for c in cards[:3]],
        rationale="The multi-agent design a plain LLM typically recommends; included so the race can test it.",
        role="llm-default",
        expected_tradeoff=OFFLINE_TRADEOFFS["orchestrator_workers"],
        slot="llm-default",
    )


def _is_cascade_card(card: MethodCard) -> bool:
    text = f"{card.method_name} {card.mechanism}".lower()
    return card.family == "cascade" or any(has_keyword(text, w) for w in CASCADE_KEYWORDS)


def _is_unusual(card: MethodCard) -> bool:
    return card.family == "other" or "other-field" in card.flags or "long-tail" in card.flags


def _family_slots(cards: list[MethodCard]) -> dict[str, list[MethodCard]]:
    """Runnable families the research supports, best-supported first, each with its cards by relevance."""
    slots: dict[str, list[MethodCard]] = {}
    for card in sorted(cards, key=lambda c: -c.relevance):
        if card.family in RUNNABLE:
            slots.setdefault(card.family, []).append(card)
        if card.family != "cascade" and _is_cascade_card(card):
            slots.setdefault("cascade", []).append(card)
    return slots


def _offline_candidate(family: str, family_cards: list[MethodCard]) -> CandidateDesign:
    template, model, params = RUNNABLE[family]
    top = family_cards[:3]
    return _candidate(
        Design(f"{template.replace('_', '-')}-{model}", template, model, dict(params)),
        based_on=[c.source_key for c in top],
        rationale=f"Suggested by {len(family_cards)} source(s), e.g. {top[0].method_name[:80]}.",
        expected_tradeoff=OFFLINE_TRADEOFFS[family],
        slot=family,
    )


# --- diversity and ids ---------------------------------------------------------------


def _signature(design: Design) -> tuple:
    """Family + model + effective params, ignoring instructions: equal signatures race the same pipeline."""
    defaults = {**RUNNABLE.get(design.family, ("", "", {}))[2]}
    if design.family == "orchestrator_workers":
        defaults["worker_model"] = design.model  # the runner's default
    params = {**defaults, **design.params}
    params.pop("instructions", None)
    items = ((k, v if isinstance(v, (int, float, str)) else repr(v)) for k, v in params.items())
    return design.family, design.model, tuple(sorted(items))


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def _finalize(
    baseline: CandidateDesign,
    llm_default: CandidateDesign,
    candidates: list[CandidateDesign],
    max_designs: int,
    warnings: list[str],
) -> list[CandidateDesign]:
    """Drop near-duplicates, keep within max_designs (one place reserved for the wildcard), assign unique ids."""
    seen = {_signature(baseline.design), _signature(llm_default.design)}
    kept = []
    for cand in candidates:
        sig = _signature(cand.design)
        if sig in seen:
            spent = f" (generation cost ${cand.cost_usd:.4f})" if cand.cost_usd else ""
            warnings.append(f"design slot {cand.slot}: dropped as a near-duplicate of an earlier design{spent}")
            continue
        seen.add(sig)
        kept.append(cand)
    room = max(0, max_designs - 2)  # the two reference points are always kept
    wild = [c for c in kept if c.slot == WILDCARD][: min(1, room)]
    chosen = [c for c in kept if c.slot != WILDCARD][: room - len(wild)] + wild
    for cand in kept:
        if all(cand is not c for c in chosen) and cand.generated_by != OFFLINE:
            warnings.append(f"design slot {cand.slot}: generated but not raced (max_designs={max_designs})")

    used = {baseline.design.id, llm_default.design.id}
    out = [baseline]
    for cand in chosen:
        base = _slug(f"{cand.design.family}-{cand.design.model}")
        new_id, i = base, 2
        while new_id in used:
            new_id, i = f"{base}-{i}", i + 1
        used.add(new_id)
        cand.design = replace(cand.design, id=new_id)
        cand.graph = graph_for(cand.design)
        out.append(cand)
    return out + [llm_default]


# --- offline -------------------------------------------------------------------------


def design_candidates_offline(
    spec: ProblemSpec, cards: list[MethodCard], *, max_designs: int = 6, warnings: list[str] | None = None
) -> list[CandidateDesign]:
    slots = _family_slots(cards)
    candidates = [_offline_candidate(f, fc) for f, fc in slots.items() if f != "orchestrator_workers"]
    crew = _llm_default(slots.get("orchestrator_workers", []))
    return _finalize(_baseline(), crew, candidates, max_designs, warnings if warnings is not None else [])


# --- Nemotron Ultra ------------------------------------------------------------------

SYSTEM = (
    "You are a senior AI systems architect. You design ONE candidate LLM pipeline, grounded only in the "
    "research cards you are given. The design will be measured on labelled examples, so do not predict "
    "that it wins; state the trade-off it makes. Never cite a card key you were not given. Reply with JSON only."
)

PROMPT = """The developer's problem: {summary}
Constraints: {constraints}
Success metric: {metric}

Design slot: {slot}
{family_block}

Model tiers (NVIDIA Nemotron on Nebius Token Factory, USD per 1M input / output tokens):
{tiers}

Research cards:
{cards}

Return JSON with exactly these keys:
- "family": {family_hint}
- "model": the tier that runs the main step, one of {tier_keys}.
- "params": {params_hint}
- "instructions": at most two sentences of task-specific guidance added to every answering prompt, drawn from the
  cards' mechanisms and failure cases. Do not restate the answer format; the pipeline adds that.
- "rationale": one or two sentences on why this design fits the problem.
- "based_on": list of the card "key" values you actually used.
- "expected_tradeoff": one sentence on what it costs (calls, latency) versus what it may gain.
"""


def _clip(text: str, n: int) -> str:
    text = " ".join(str(text or "").split())
    return text if len(text) <= n else text[: n - 1].rstrip() + "…"


def _card_payload(cards: list[MethodCard]) -> str:
    rows = [
        {
            "key": c.source_key,
            "name": _clip(c.method_name, 160),
            "mechanism": _clip(c.mechanism, _FIELD_CHARS),
            "works_when": _clip(c.works_when, _FIELD_CHARS),
            "breaks_when": _clip(c.breaks_when, _FIELD_CHARS),
            "reported_result": _clip(c.reported_result, _FIELD_CHARS),
        }
        for c in cards
    ]
    return json.dumps(rows, indent=1, ensure_ascii=False)


def _slot_prompt(spec: ProblemSpec, slot: str, cards: list[MethodCard]) -> str:
    """The whole context for one slot: the problem and this slot's cards, nothing from other slots."""
    if slot == WILDCARD:
        family_block = (
            "This is the wildcard slot. The cards below come from other fields or are unusual. Transfer their core "
            "mechanism into whichever runnable family fits it best:\n"
            + "\n".join(f"- {f}: {FAMILY_SUMMARIES[f]}; params {PARAM_HINTS[f]}" for f in RUNNABLE)
        )
        family_hint, params_hint = f"one of {list(RUNNABLE)}.", "the params for the family you chose, as listed above."
    else:
        family_block = f"Architecture family (fixed for this slot): {slot}: {FAMILY_SUMMARIES[slot]}."
        family_hint, params_hint = f'"{slot}" (fixed).', PARAM_HINTS[slot] + "."
        if slot == "cascade":
            family_hint += ' "model" is the cheap tier.'
    tiers = "\n".join(f"- {m.key}: ${m.usd_per_m_input:.2f} / ${m.usd_per_m_output:.2f}" for m in CATALOG.values())
    return PROMPT.format(
        summary=spec.summary or spec.problem[:300],
        constraints="; ".join(spec.constraints) or "none stated",
        metric=spec.success_metric or "accuracy against the labelled examples",
        slot=slot,
        family_block=family_block,
        tiers=tiers,
        cards=_card_payload(cards),
        family_hint=family_hint,
        tier_keys=list(CATALOG),
        params_hint=params_hint,
    )


def _tier(value) -> str:
    key = str(value).strip().lower() if isinstance(value, str) else ""
    if key not in CATALOG:
        raise ValueError(f"unknown model tier {value!r}")
    return key


def _number(value, kind: str, low: float, high: float) -> int | float:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise ValueError(f"not a number: {value!r}")
    x = float(value)  # raises ValueError for non-numeric strings
    if not math.isfinite(x):
        raise ValueError(f"not a finite number: {value!r}")
    x = min(high, max(low, x))
    return int(round(x)) if kind == "int" else round(x, 2)


def _params(family: str, model: str, raw: dict) -> dict:
    """Keep only the family's known params, validated and clamped; fill the rest from the template."""
    if family == "cascade":
        raw = {"cheap_model": model, **raw}
    defaults = RUNNABLE[family][2]
    out = {}
    for key, rule in PARAM_RULES.get(family, {}).items():
        value = raw.get(key, defaults.get(key))
        out[key] = _tier(value) if rule[0] == "tier" else _number(value, *rule)
    if family == "cascade" and out["cheap_model"] == out["escalate_model"]:
        raise ValueError("cascade must escalate to a different tier")
    return out


class _Recorder:
    """Wraps the caller's client for one slot so the slot's spend is known even when it fails."""

    def __init__(self, client: LLMClient):
        self.client = client
        self.calls: list[Completion] = []

    def complete(self, model: str, messages: list[dict], **kw) -> Completion:
        completion = self.client.complete(model, messages, **kw)
        self.calls.append(completion)
        return completion

    @property
    def cost_usd(self) -> float:
        return sum(c.cost_usd for c in self.calls)


def _from_reply(slot: str, data, cards: list[MethodCard], model: str) -> CandidateDesign:
    """Validate one slot's JSON strictly; raise ValueError (or KeyError) on anything unusable."""
    if not isinstance(data, dict):
        raise ValueError("reply is not a JSON object")
    family = slot
    if slot == WILDCARD:
        family = str(data.get("family", "")).strip()
        if family not in RUNNABLE:
            raise ValueError(f"unknown family {family!r}")
    tier = _tier(data.get("model"))
    raw_params = data.get("params") or {}
    if not isinstance(raw_params, dict):
        raise ValueError("params is not an object")
    params = _params(family, tier, raw_params)
    if family == "cascade":
        tier = params["cheap_model"]  # the cascade's model is its cheap tier
    instructions = data.get("instructions") or ""
    if not isinstance(instructions, str):
        raise ValueError("instructions is not a string")
    instructions = _clip(instructions, MAX_INSTRUCTIONS_CHARS)
    if instructions:
        params["instructions"] = instructions

    # Citations: only keys this slot was given. If none survive, credit the cards it was shown.
    given = [c.source_key for c in cards]
    cited = data.get("based_on")
    cited = [cited] if isinstance(cited, str) else cited if isinstance(cited, list) else []
    based_on = list(dict.fromkeys(k.strip() for k in cited if isinstance(k, str) and k.strip() in given)) or given

    design = Design(f"{family}-{tier}", family, tier, params)  # id is reassigned in _finalize
    return _candidate(
        design,
        based_on=based_on,
        rationale=_clip(data.get("rationale", ""), _TEXT_CHARS) if isinstance(data.get("rationale"), str) else "",
        generated_by=model,
        expected_tradeoff=_clip(data.get("expected_tradeoff", ""), _TEXT_CHARS)
        if isinstance(data.get("expected_tradeoff"), str)
        else "",
        slot=slot,
    )


def _design_slot(
    spec: ProblemSpec, slot: str, cards: list[MethodCard], client: LLMClient, model: str
) -> tuple[CandidateDesign | None, str | None]:
    """One isolated Ultra call for one slot. Falls back to the offline template (or nothing, for the wildcard)."""
    recorder = _Recorder(client)
    try:
        data, _ = ask_json(recorder, model, SYSTEM, _slot_prompt(spec, slot, cards), max_tokens=900)
        cand = _from_reply(slot, data, cards, model)
        cand.cost_usd = recorder.cost_usd
        return cand, None
    except Exception as exc:  # one bad slot never sinks the others
        reason = f"{type(exc).__name__}: {_clip(str(exc), 120)}"
        if slot == WILDCARD:
            return None, f"design slot wildcard: {model} output rejected ({reason}); slot dropped"
        cand = _offline_candidate(slot, cards)
        cand.cost_usd = recorder.cost_usd  # paid for, even though unusable
        return cand, f"design slot {slot}: {model} output rejected ({reason}); used the offline template"


def design_candidates(
    spec: ProblemSpec,
    cards: list[MethodCard],
    client: LLMClient | None = None,
    *,
    max_designs: int = 6,
    warnings: list[str] | None = None,
    model: str = "ultra",
    max_workers: int = 4,
) -> list[CandidateDesign]:
    """Candidate designs for the race: baseline first, llm-default last, generated candidates between.

    `client` must already be wrapped by the caller (e.g. in BudgetGuard). Without one, the offline
    mapping is used. Warnings about rejected or dropped slots are appended to `warnings`.
    """
    warnings = warnings if warnings is not None else []
    if client is None:
        return design_candidates_offline(spec, cards, max_designs=max_designs, warnings=warnings)

    slots = _family_slots(cards)
    room = max(0, max_designs - 2)
    jobs = [(family, family_cards[:_CARDS_PER_SLOT]) for family, family_cards in slots.items()][:room]
    unusual = sorted((c for c in cards if _is_unusual(c)), key=lambda c: -c.relevance)[:_CARDS_PER_SLOT]
    if unusual and room:
        jobs.append((WILDCARD, unusual))  # may displace the last family slot; one spare call absorbs a duplicate

    with ThreadPoolExecutor(max_workers=max(1, max_workers)) as pool:
        results = list(pool.map(lambda job: _design_slot(spec, job[0], job[1], client, model), jobs))
    candidates = [cand for cand, _ in results if cand is not None]
    warnings.extend(w for _, w in results if w)
    return _finalize(_baseline(), _llm_default(slots.get("orchestrator_workers", [])), candidates, max_designs, warnings)
