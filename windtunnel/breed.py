"""Diagnose and breed: explain why losing designs failed, then breed new ones.

After a race, `diagnose` looks at the weakest designs and names their failure modes
from a small fixed taxonomy. The evidence is computed offline from the recorded
outputs (confusion pairs, invalid answers, errors, constraint eliminations). With a
client, Nemotron Ultra also picks failure modes from a few failure examples, and the
two are merged with each mode recording who flagged it.

`breed` looks up mechanisms from other fields (mechanisms.json) that address those
failure modes and applies each one's mutation to the design, giving a new runnable
design. `race_with_breeding` puts this in a loop and races the bred designs. Bred
designs are chosen from failures on the race examples, so their scores on those same
examples are optimistic. When the task has enough examples, a held-out split is kept
back for the final race.
"""

from __future__ import annotations

import json
import math
import random
import re
from collections import Counter
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

from .designs import MAX_INSTRUCTIONS_CHARS, Design
from .graph import DesignGraph, graph_for
from .budget import BudgetExceeded
from .llm import LLMClient, Recorder, is_mock
from .racer import DesignResult, RaceConfig, RaceResult, race
from .report import to_dict as race_to_dict
from .scoring import normalize
from .structured import ask_json
from .task import TaskSpec

FAILURE_MODES = {
    "format_violation": "the answer is not a valid label or not in the required format",
    "label_confusion": "systematic confusions: the same expected answer is repeatedly given as the same wrong one",
    "inconsistent": "unstable answers: wrong answers are scattered or sampled answers disagree",
    "pipeline_error": "the pipeline raised exceptions (timeouts, bad replies, budget refusals)",
    "over_budget": "cost per 1k requests exceeds the task's limit",
    "too_slow": "median latency exceeds the task's limit",
    "reasoning_error": "valid answers that are wrong for no more specific reason (catch-all)",
}

# Mutations the code knows how to apply, with the arguments each one takes.
MUTATIONS = {
    "add_voting": {"n"},
    "add_self_check": set(),
    "switch_to_cascade": {"cheap_model", "escalate_model"},
    "upgrade_tier": set(),
    "downgrade_tier": set(),
    "add_plan": set(),
    "add_instructions": {"text"},
}

# Cost ladder for tier moves. Lightning costs the same as Nano, so it sits on Nano's rung.
TIER_RANK = {"nano": 0, "lightning": 0, "super": 1, "ultra": 2}
LADDER = ["nano", "super", "ultra"]

MECHANISMS_PATH = Path(__file__).with_name("mechanisms.json")
OFFLINE = "offline-heuristic"
MAX_SAMPLE_FAILURES = 5


# ---------------------------------------------------------------- diagnosis


@dataclass
class FailureMode:
    mode: str
    evidence: list[str] = field(default_factory=list)
    flagged_by: list[str] = field(default_factory=list)  # "offline-heuristic" and/or a model key
    share: float = 0.0  # fraction of examples affected, from offline evidence (0 if model-only)


@dataclass
class Diagnosis:
    design_id: str
    accuracy: float
    n: int
    failure_modes: list[FailureMode]
    sample_failures: list[dict]
    produced_by: str
    confusions: list[tuple[str, str, int]] = field(default_factory=list)  # (expected, got, count), count >= 2
    labels: list[str] | None = None
    cost_usd: float = 0.0

    @property
    def modes(self) -> list[str]:
        return [f.mode for f in self.failure_modes]

    def to_dict(self) -> dict:
        d = asdict(self)
        d["confusions"] = [list(c) for c in self.confusions]
        return d


def _labels_for(task: TaskSpec) -> tuple[list[str] | None, bool]:
    """The allowed answers, and whether they were declared (True) or inferred from the examples."""
    if task.labels:
        return list(task.labels), True
    expected = sorted({e.expected for e in task.examples})
    if task.scorer == "exact" and len(expected) <= 20 and len(expected) * 2 < len(task.examples):
        return expected, False
    return None, False


def _clip(text: str, n: int = 200) -> str:
    text = " ".join(text.split())
    return text if len(text) <= n else text[: n - 1] + "…"


def _offline_modes(r: DesignResult, task: TaskSpec) -> tuple[list[FailureMode], list[tuple[str, str, int]], list[str]]:
    """Failure modes from the recorded outputs alone; returns (modes, confusions, failed example ids)."""
    expected = {e.id: e.expected for e in task.examples}
    n = max(r.n, 1)
    failed = sorted(eid for eid, s in r.scores.items() if s < 1.0)
    modes: list[FailureMode] = []

    def add(mode: str, count: float, *evidence: str) -> None:
        modes.append(FailureMode(mode, list(evidence), [OFFLINE], round(count / n, 4)))

    limits = task.constraints
    reason = r.elimination_reason or ""
    if reason.startswith("over budget") or (limits.max_cost_per_1k is not None and r.cost_per_1k > limits.max_cost_per_1k):
        add("over_budget", n, reason or f"${r.cost_per_1k:.2f}/1k exceeds the ${limits.max_cost_per_1k:.2f}/1k limit")
    if reason.startswith("too slow") or (limits.max_p50_latency_s is not None and r.p50_latency_s > limits.max_p50_latency_s):
        add("too_slow", n, reason or f"p50 {r.p50_latency_s:.1f}s exceeds the {limits.max_p50_latency_s:.1f}s limit")

    if r.errors:
        kinds = Counter(err.split(":", 1)[0] for err in r.errors.values())
        example = next(iter(r.errors.values()))
        add(
            "pipeline_error",
            len(r.errors),
            f"{len(r.errors)} of {r.n} examples raised errors ({', '.join(f'{k} x{c}' for k, c in kinds.most_common(3))})",
            f"e.g. {_clip(example, 160)}",
        )

    labels, declared = _labels_for(task)
    answered = [eid for eid in failed if eid not in r.errors]
    if labels is not None:
        canon = {normalize(lab): lab for lab in labels}
        invalid = [eid for eid in answered if normalize(r.outputs.get(eid, "")) not in canon]
        if invalid:
            shown = ", ".join(repr(_clip(r.outputs.get(eid, ""), 40)) for eid in invalid[:3])
            where = "the task's labels" if declared else "the answers seen in the examples"
            add("format_violation", len(invalid), f"{len(invalid)} of {r.n} answers were not one of {where}, e.g. {shown}")
        valid_wrong = [eid for eid in answered if eid not in invalid]
        pairs = Counter((expected[eid], canon[normalize(r.outputs[eid])]) for eid in valid_wrong)
        confusions = [(exp, got, c) for (exp, got), c in pairs.most_common() if c >= 2]
        systematic = sum(c for _, _, c in confusions)
        if confusions:
            add(
                "label_confusion",
                systematic,
                *(f"expected {exp}, answered {got}: {c} times" for exp, got, c in confusions[:4]),
            )
        scattered = len(valid_wrong) - systematic
        if scattered >= 3 and not confusions:
            # Per-sample answers are not recorded, so scattered one-off errors are our proxy for instability.
            add(
                "inconsistent",
                scattered,
                f"{scattered} wrong answers spread over {len(pairs)} different confusions with no repeats, "
                "so the errors look unstable rather than systematic (proxy: per-sample answers are not recorded)",
            )
        elif scattered > 0:
            add("reasoning_error", scattered, f"{scattered} valid but wrong answers with no repeated pattern")
    else:
        confusions = []
        empty = [eid for eid in answered if not r.outputs.get(eid, "").strip()]
        if task.scorer == "numeric":
            empty += [eid for eid in answered if eid not in empty and not re.search(r"-?\d", r.outputs.get(eid, ""))]
        if empty:
            add("format_violation", len(empty), f"{len(empty)} of {r.n} answers were empty or had no usable value")
        wrong = len(answered) - len(empty)
        if wrong > 0:
            add("reasoning_error", wrong, f"{wrong} of {r.n} answers were well-formed but wrong")

    modes.sort(key=lambda m: -m.share)
    return modes, confusions, failed


def _sample_failures(r: DesignResult, task: TaskSpec, failed: list[str]) -> list[dict]:
    by_id = {e.id: e for e in task.examples}
    out = []
    for eid in failed[:MAX_SAMPLE_FAILURES]:
        ex = by_id[eid]
        out.append(
            {
                "example_id": eid,
                "input": _clip(ex.input, 300),
                "expected": ex.expected,
                "got": r.outputs.get(eid, ""),
                "error": r.errors.get(eid),
            }
        )
    return out


DIAGNOSE_SYSTEM = "You are a reliability engineer. You diagnose why an AI pipeline fails, using only the evidence given."

DIAGNOSE_PROMPT = """Task: {task}
Design under test ({design_id}): {design}
Measured accuracy: {accuracy:.2f} on {n} examples.

Evidence computed from the recorded outputs:
{evidence}

Failure examples:
{examples}

Failure taxonomy (use these keys only):
{taxonomy}

Return JSON: {{"failure_modes": [{{"mode": "<taxonomy key>", "evidence": "<one sentence citing the examples>"}}]}}
List at most 3 modes, most important first. Do not invent modes outside the taxonomy."""


def _model_label(client: LLMClient, model: str) -> str:
    return f"mock-{model}" if is_mock(client) else model


def _ask_model(diag: Diagnosis, r: DesignResult, task: TaskSpec, client: LLMClient, model: str) -> None:
    """Merge the model's picks into the diagnosis in place.

    Any failure (unparseable or wrong-shaped reply, the spending cap, an API error) leaves
    the diagnosis offline-only and says why; what the failed calls cost is still counted.
    """
    evidence = "\n".join(f"- {m.mode}: {'; '.join(m.evidence)}" for m in diag.failure_modes) or "- none found"
    examples = "\n".join(
        f"- input: {f['input']}\n  expected: {f['expected']}\n  got: {f['got'] or '(empty)'}"
        + (f"\n  error: {f['error']}" if f["error"] else "")
        for f in diag.sample_failures
    ) or "- none"
    prompt = DIAGNOSE_PROMPT.format(
        task=task.description,
        design_id=r.design.id,
        design=r.design.describe(),
        accuracy=r.accuracy,
        n=r.n,
        evidence=evidence,
        examples=examples,
        taxonomy="\n".join(f"- {k}: {v}" for k, v in FAILURE_MODES.items()),
    )
    label = _model_label(client, model)
    recorder = Recorder(client)
    try:
        data, _ = ask_json(recorder, model, DIAGNOSE_SYSTEM, prompt, max_tokens=800)
    except ValueError:
        diag.produced_by = f"{OFFLINE} ({label} reply was not valid JSON)"
        return
    except BudgetExceeded:
        diag.produced_by = f"{OFFLINE} (budget exhausted before the {label} diagnosis)"
        return
    except Exception as exc:  # a diagnosis is never worth losing the race it explains
        diag.produced_by = f"{OFFLINE} ({label} call failed: {type(exc).__name__})"
        return
    finally:
        diag.cost_usd += recorder.cost_usd
    picks = data.get("failure_modes", []) if isinstance(data, dict) else data
    if isinstance(picks, dict):
        picks = [picks]  # a single mode given as an object instead of a list
    if not isinstance(picks, list):
        picks = []
    merged = False
    for pick in picks[:3]:
        mode = pick.get("mode") if isinstance(pick, dict) else None
        if not isinstance(mode, str) or mode not in FAILURE_MODES:
            continue
        text = _clip(str(pick.get("evidence", "")).strip(), 300)
        existing = next((m for m in diag.failure_modes if m.mode == mode), None)
        if existing is None:
            existing = FailureMode(mode)
            diag.failure_modes.append(existing)
        if label not in existing.flagged_by:
            existing.flagged_by.append(label)
        if text:
            existing.evidence.append(f"({label}) {text}")
        merged = True
    diag.produced_by = f"{OFFLINE}+{label}" if merged else f"{OFFLINE} ({label} named no taxonomy modes)"


def diagnose(
    result: RaceResult, task: TaskSpec, client: LLMClient | None = None, *, max_designs: int = 3, model: str = "ultra"
) -> list[Diagnosis]:
    """Explain the weakest designs' failures: losers (not tied with the best) first, then tied designs with failures."""
    def weakest_first(rs: list[DesignResult]) -> list[DesignResult]:
        return sorted(rs, key=lambda r: (r.alive, r.accuracy))

    losers = [r for r in result.results if r.design.id not in result.tied_ids]
    tied = [r for r in result.results if r.design.id in result.tied_ids]
    out: list[Diagnosis] = []
    for r in weakest_first(losers) + weakest_first(tied):
        if len(out) >= max_designs:
            break
        modes, confusions, failed = _offline_modes(r, task)
        if not modes:
            continue  # nothing went wrong for this design
        labels, _ = _labels_for(task)
        diag = Diagnosis(
            design_id=r.design.id,
            accuracy=r.accuracy,
            n=r.n,
            failure_modes=modes,
            sample_failures=_sample_failures(r, task, failed),
            produced_by=OFFLINE,
            confusions=confusions,
            labels=labels,
        )
        if client is not None:
            _ask_model(diag, r, task, client, model)
        out.append(diag)
    return out


# ---------------------------------------------------------------- mechanisms


@dataclass(frozen=True)
class Mechanism:
    id: str
    name: str
    field: str
    failure_modes: tuple[str, ...]
    idea: str
    mutation: dict
    source: dict | None = None  # {"title", "url"} only when verified; never invented

    @property
    def op(self) -> str:
        return self.mutation["op"]


def load_mechanisms(path: str | Path | None = None) -> list[Mechanism]:
    rows = json.loads(Path(path or MECHANISMS_PATH).read_text())
    out = []
    for row in rows:
        m = Mechanism(
            id=row["id"],
            name=row["name"],
            field=row["field"],
            failure_modes=tuple(row["failure_modes"]),
            idea=row["idea"],
            mutation=dict(row["mutation"]),
            source=row.get("source"),
        )
        unknown = set(m.failure_modes) - set(FAILURE_MODES)
        if unknown:
            raise ValueError(f"mechanism {m.id!r}: unknown failure modes {sorted(unknown)}")
        if m.op not in MUTATIONS:
            raise ValueError(f"mechanism {m.id!r}: unknown mutation {m.op!r}; known: {', '.join(MUTATIONS)}")
        args = set(m.mutation) - {"op"}
        if args != MUTATIONS[m.op]:
            raise ValueError(f"mechanism {m.id!r}: {m.op} takes {sorted(MUTATIONS[m.op])}, got {sorted(args)}")
        if m.source is not None and not (m.source.get("title") and m.source.get("url")):
            raise ValueError(f"mechanism {m.id!r}: source needs a title and url, or null")
        out.append(m)
    if len({m.id for m in out}) != len(out):
        raise ValueError("duplicate mechanism ids")
    return out


# ---------------------------------------------------------------- mutations


def _keep_instructions(d: Design) -> dict:
    return {"instructions": d.params["instructions"]} if d.params.get("instructions") else {}


def _shift(model: str, step: int) -> str | None:
    i = TIER_RANK[model] + step
    return LADDER[i] if 0 <= i < len(LADDER) else None


def _fill(text: str, labels: list[str] | None, confusions: list[tuple[str, str, int]]) -> str | None:
    """Fill {labels} / {confusions} placeholders; None when the mechanism needs data we don't have."""
    if "{confusions}" in text and not confusions:
        return None
    return text.replace("{labels}", ", ".join(labels) if labels else "the exact format the task asks for").replace(
        "{confusions}", "; ".join(f"{exp} vs {got}" for exp, got, _ in confusions[:4])
    )


def apply_mutation(
    design: Design,
    mutation: dict,
    *,
    new_id: str,
    labels: list[str] | None = None,
    confusions: list[tuple[str, str, int]] | None = None,
) -> Design | None:
    """The mutated design, or None when the mutation does not change this design (or lacks the data it needs)."""
    op, d, keep = mutation["op"], design, _keep_instructions(design)
    if op == "add_voting":
        n = int(mutation["n"])
        if d.family == "self_consistency" and int(d.params.get("n", 5)) >= n:
            return None
        return Design(new_id, "self_consistency", d.model, {"n": n, "temperature": 0.7, **keep})
    if op == "add_self_check":
        return None if d.family == "self_check" else Design(new_id, "self_check", d.model, keep)
    if op == "add_plan":
        return None if d.family == "plan_execute" else Design(new_id, "plan_execute", d.model, keep)
    if op == "switch_to_cascade":
        cheap, strong = mutation["cheap_model"], mutation["escalate_model"]
        if d.family == "cascade" and d.params.get("cheap_model", "nano") == cheap and d.params.get("escalate_model", "super") == strong:
            return None
        params = {"cheap_model": cheap, "samples": 2, "temperature": 0.7, "escalate_model": strong, **keep}
        return Design(new_id, "cascade", cheap, params)
    if op in ("upgrade_tier", "downgrade_tier"):
        step = 1 if op == "upgrade_tier" else -1
        if d.family == "cascade":  # the escalation tier is what a cascade's quality and cost hinge on
            cheap = d.params.get("cheap_model", "nano")
            strong = _shift(d.params.get("escalate_model", "super"), step)
            if strong is None or TIER_RANK[strong] <= TIER_RANK[cheap]:
                return None
            return Design(new_id, "cascade", d.model, {**d.params, "escalate_model": strong})
        model = _shift(d.model, step)
        if model is None:
            return None
        params = dict(d.params)
        if "worker_model" in params and step < 0 and TIER_RANK[params["worker_model"]] > TIER_RANK[model]:
            params["worker_model"] = model  # workers should not outrank the orchestrator after a downgrade
        return Design(new_id, d.family, model, params)
    if op == "add_instructions":
        text = _fill(str(mutation["text"]), labels, confusions or [])
        current = d.params.get("instructions", "").strip()
        if text is None or text in current:
            return None
        combined = f"{current} {text}".strip()[:MAX_INSTRUCTIONS_CHARS]
        return Design(new_id, d.family, d.model, {**d.params, "instructions": combined})
    raise ValueError(f"unknown mutation {op!r}")


def signature(design: Design) -> tuple:
    """What a design does, ignoring its id, so duplicates can be spotted."""
    return design.family, design.model, json.dumps(design.params, sort_keys=True)


# ---------------------------------------------------------------- breeding


@dataclass
class BredDesign:
    design: Design
    parent_id: str
    mechanism_id: str
    mechanism_name: str
    mechanism_field: str
    failure_mode: str  # the diagnosed mode this mechanism targets
    graph: DesignGraph
    source: dict | None = None
    generated_by: str = OFFLINE  # the mutation is rule-based, whoever wrote the diagnosis

    def to_dict(self) -> dict:
        d = self.design
        return {
            "id": d.id,
            "family": d.family,
            "model": d.model,
            "params": d.params,
            "description": d.describe(),
            "rationale": d.rationale,
            "parent_id": self.parent_id,
            "mechanism": {"id": self.mechanism_id, "name": self.mechanism_name, "field": self.mechanism_field, "source": self.source},
            "failure_mode": self.failure_mode,
            "graph": self.graph.to_dict(),
            "generated_by": self.generated_by,
        }


def _unique_id(base: str, taken: set[str]) -> str:
    candidate, i = base, 2
    while candidate in taken:
        candidate, i = f"{base}-{i}", i + 1
    return candidate


def breed(
    design: Design,
    diagnosis: Diagnosis,
    *,
    limit: int = 2,
    mechanisms: list[Mechanism] | None = None,
    taken_ids: set[str] | frozenset[str] = frozenset(),
) -> list[BredDesign]:
    """Apply mechanisms that target the diagnosed failure modes. Picks round-robin across modes for variety.

    Child ids are "<parent>.<mechanism>" (a dot, not "+", so the advisor's PICK parser still reads them).
    """
    mechs = mechanisms if mechanisms is not None else load_mechanisms()
    queues = [[(f.mode, m) for m in mechs if f.mode in m.failure_modes] for f in diagnosis.failure_modes]
    order = [q[rank] for rank in range(max(map(len, queues), default=0)) for q in queues if rank < len(q)]
    taken = set(taken_ids) | {design.id}
    seen = {signature(design)}
    out: list[BredDesign] = []
    used: set[str] = set()
    for mode, mech in order:
        if len(out) >= limit:
            break
        if mech.id in used:
            continue
        child_id = _unique_id(f"{design.id}.{mech.id}", taken)
        child = apply_mutation(design, mech.mutation, new_id=child_id, labels=diagnosis.labels, confusions=diagnosis.confusions)
        if child is None or signature(child) in seen:
            continue
        evidence = next((m.evidence[0] for m in diagnosis.failure_modes if m.mode == mode and m.evidence), "")
        rationale = f"{mech.name} ({mech.field}): {mech.idea} Bred from {design.id} to fix {mode}" + (f" ({evidence})." if evidence else ".")
        child = replace(child, rationale=rationale)
        out.append(BredDesign(child, design.id, mech.id, mech.name, mech.field, mode, graph_for(child), mech.source))
        taken.add(child_id)
        seen.add(signature(child))
        used.add(mech.id)
    return out


# ---------------------------------------------------------------- the loop


@dataclass
class BreedingResult:
    races: list[RaceResult]
    race_roles: list[str]  # what each race was for, e.g. "initial (search split)"
    diagnoses: list[list[Diagnosis]]  # one list per generation
    bred: list[BredDesign]
    lineage: dict[str, dict]  # bred design id -> parent, mechanism, generation
    search_ids: list[str]
    holdout_ids: list[str]  # empty when there were too few examples to hold any out
    provenance: str  # "mock" or "measured"
    note: str
    final_on_holdout: bool = False  # the deciding race ran on the held-out split, so selection bias is controlled

    @property
    def final(self) -> RaceResult:
        return self.races[-1]

    @property
    def recommended_id(self) -> str | None:
        return self.final.recommended_id

    def to_dict(self) -> dict:
        return {
            "provenance": self.provenance,
            "note": self.note,
            "races": [{"role": role, **race_to_dict(r, None, self.provenance)} for role, r in zip(self.race_roles, self.races)],
            "diagnoses": [[d.to_dict() for d in gen] for gen in self.diagnoses],
            "bred": [b.to_dict() for b in self.bred],
            "lineage": self.lineage,
            "search_ids": self.search_ids,
            "holdout_ids": self.holdout_ids,
            "final_on_holdout": self.final_on_holdout,
            "recommended_id": self.recommended_id,
        }


def split_holdout(
    task: TaskSpec, *, fraction: float = 0.3, min_holdout: int = 20, seed: int = 0
) -> tuple[TaskSpec, TaskSpec | None]:
    """(search task, held-out task), or (task, None) when the held-out part would be under min_holdout examples."""
    k = int(len(task.examples) * fraction)
    if k < min_holdout or len(task.examples) - k < min_holdout:
        return task, None
    examples = list(task.examples)
    random.Random(f"holdout-{seed}").shuffle(examples)
    held, search = examples[:k], examples[k:]
    return replace(task, examples=search), replace(task, examples=held)


def _pick_bred(per_parent: list[list[BredDesign]], pool: list[Design], k: int) -> list[BredDesign]:
    """Round-robin over parents (weakest first), skipping designs the pool already has."""
    seen = {signature(d) for d in pool}
    out: list[BredDesign] = []
    for rank in range(max(map(len, per_parent), default=0)):
        for kids in per_parent:
            if rank < len(kids) and len(out) < k and signature(kids[rank].design) not in seen:
                out.append(kids[rank])
                seen.add(signature(kids[rank].design))
    return out


def race_with_breeding(
    task: TaskSpec,
    designs: list[Design],
    client: LLMClient,
    config: RaceConfig | None = None,
    *,
    generations: int = 1,
    breed_per_generation: int = 3,
    holdout_fraction: float = 0.3,
    min_holdout: int = 20,
    diagnose_with_model: bool = True,
    mechanisms: list[Mechanism] | None = None,
) -> BreedingResult:
    """Race, diagnose the losers, breed fixes, race survivors plus bred designs again.

    The last race runs on a held-out split when the task has enough examples, otherwise
    on the same examples with a fresh seed (and the note says its scores are optimistic).
    The client must already be wrapped in BudgetGuard if it is a real one.
    """
    if generations < 1:
        raise ValueError("generations must be at least 1")
    cfg = config or RaceConfig()
    mechs = mechanisms if mechanisms is not None else load_mechanisms()
    mock = is_mock(client)
    # The mock cannot write diagnoses, so mock runs stay on the offline heuristic.
    diag_client = client if diagnose_with_model and not mock else None
    search, holdout = split_holdout(task, fraction=holdout_fraction, min_holdout=min_holdout, seed=cfg.seed)

    races = [race(search, designs, client, cfg)]
    roles = ["initial (search split)" if holdout else "initial"]
    diagnoses: list[list[Diagnosis]] = []
    bred_all: list[BredDesign] = []
    lineage: dict[str, dict] = {}
    stopped_early = False
    for gen in range(1, generations + 1):
        prev = races[-1]
        if prev.aborted:
            break  # the spending cap stopped the race; breeding would only be refused too
        diags = diagnose(prev, search, diag_client, max_designs=max(3, breed_per_generation))
        diagnoses.append(diags)
        survivors = [r.design for r in prev.results if r.alive]
        taken = {d.id for d in designs} | set(lineage)
        per_parent = [breed(prev.get(dg.design_id).design, dg, limit=2, mechanisms=mechs, taken_ids=taken) for dg in diags]
        kids = _pick_bred(per_parent, survivors, breed_per_generation)
        for kid in kids:
            lineage[kid.design.id] = {
                "parent_id": kid.parent_id,
                "mechanism_id": kid.mechanism_id,
                "mechanism": kid.mechanism_name,
                "field": kid.mechanism_field,
                "failure_mode": kid.failure_mode,
                "generation": gen,
            }
        bred_all += kids
        pool = survivors + [k.design for k in kids]
        if not kids or not pool:
            stopped_early = True
            break
        last = gen == generations
        target = holdout if last and holdout is not None else search
        races.append(race(target, pool, client, replace(cfg, seed=cfg.seed + gen)))
        roles.append("confirmation (held-out split)" if target is holdout else f"generation {gen}")

    pool = [r.design for r in races[-1].results if r.alive]
    if stopped_early and holdout is not None and pool and not races[-1].aborted:
        # Still confirm whatever survived on data no selection step has seen. With no survivors
        # (all over the task's limits) there is nothing to confirm, so no extra race is paid for.
        races.append(race(holdout, pool, client, replace(cfg, seed=cfg.seed + generations + 1)))
        roles.append("confirmation (held-out split)")

    n_total = len(task.examples)
    final = races[-1]
    if holdout is not None and final.task is holdout:
        rec_n = final.get(final.recommended_id).n if final.recommended_id else None
        measured = f" (the recommended design was measured on {rec_n} of them)" if rec_n is not None and rec_n < len(holdout.examples) else ""
        note = (
            f"Survivors and bred designs were chosen using failures on {len(search.examples)} search examples. "
            f"The final race ran on {len(holdout.examples)} held-out examples{measured} that no selection or breeding "
            "step saw, so its scores are not inflated by that selection (they are still small-sample estimates)."
        )
    elif holdout is not None:
        note = (
            f"A held-out split of {len(holdout.examples)} examples was kept back, but no confirmation race ran on it "
            f"(no surviving design, or the race stopped), so the final scores come from the {len(search.examples)} "
            "search examples."
        )
    else:
        note = (
            f"Bred designs were chosen using failures on the same {n_total} examples they were then raced on "
            "(selection bias), so their scores are optimistic until confirmed on held-out examples. "
            f"A held-out split ({holdout_fraction:.0%} of examples, at least {min_holdout}) needs about "
            f"{math.ceil(min_holdout / holdout_fraction) if holdout_fraction > 0 else 'more'} examples; this task has {n_total}."
        )
    if stopped_early:
        note += " Breeding stopped early: no mechanism produced a new design."
    aborted = next((r.aborted for r in races if r.aborted), None)
    if aborted:
        note += f" A race stopped before finishing: {aborted}."
    if mock:
        note += " Mock run: answers are simulated and test the plumbing only; never quote these numbers."
    return BreedingResult(
        races=races,
        race_roles=roles,
        diagnoses=diagnoses,
        bred=bred_all,
        lineage=lineage,
        search_ids=[e.id for e in search.examples],
        holdout_ids=[e.id for e in holdout.examples] if holdout else [],
        provenance="mock" if mock else "measured",
        note=note,
        final_on_holdout=holdout is not None and final.task is holdout,
    )
