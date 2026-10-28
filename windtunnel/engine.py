"""The whole Wind Tunnel in one call: problem in, report out.

Stages, in order: plain LLM answer -> research (L1-L3) -> candidate designs (L4) ->
race on the task's examples, with diagnose-and-breed (L5) -> LLM advisor pick ->
limitations (L6). Each stage adds a model-receipt entry naming who produced it (a
model key, "offline-heuristic" or "mock"), what it cost and how long it took.

Nemotron generates; measurement judges. Without a real client every model step falls
back to its offline heuristic, the race runs on MockClient and is labelled "mock",
and the report says so. The caller must wrap a real client in BudgetGuard.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from .advisor import AdvisorPick, llm_pick
from .atlas import outcome_records
from .baseline import plain_llm_answer
from .breed import BreedingResult, race_with_breeding
from .budget import BudgetExceeded
from .generate import CandidateDesign, design_candidates
from .graph import DesignGraph, graph_for, to_mermaid
from .llm import LLMClient, MockClient, Recorder, is_mock, unwrap
from .models import CATALOG
from .racer import DesignResult, RaceConfig, RaceResult, race
from .report import finite
from .report import to_dict as race_to_dict
from .research import ResearchResult, research
from .scout import Fetcher, Provider
from .task import TaskSpec

STAGES = ["plain_llm_answer", "research", "design_candidates", "race", "llm_pick", "limitations"]
OFFLINE = "offline-heuristic"
MOCK = "mock"

Progress = Callable[[str, str], None]  # (stage, status) with status "running", "done", "skipped" or "failed"


@dataclass
class ReceiptEntry:
    stage: str
    produced_by: str  # model key(s) joined by "+", "offline-heuristic", "mock" or "none"
    cost_usd: float = 0.0  # real model spend; mock spend is never counted here
    seconds: float = 0.0
    status: str = "done"  # "done" or "skipped"
    detail: str = ""
    simulated_cost_usd: float | None = None  # what a mock race would have cost at list prices

    def to_dict(self) -> dict:
        d = asdict(self)
        d["cost_usd"] = round(self.cost_usd, 6)
        d["seconds"] = round(self.seconds, 3)
        if self.simulated_cost_usd is None:
            del d["simulated_cost_usd"]  # only mock races have one
        else:
            d["simulated_cost_usd"] = round(self.simulated_cost_usd, 6)
        return d


_unwrap = unwrap  # kept for callers that import it from here; is_mock is re-exported from llm too


def _join(labels) -> str:
    """Distinct producers, model keys first, e.g. "ultra+super+offline-heuristic"."""
    distinct = list(dict.fromkeys(label for label in labels if label))
    return "+".join(sorted(distinct, key=lambda label: label == OFFLINE)) or OFFLINE


def mermaid_for(graph: DesignGraph, family: str) -> str:
    # Fan-out designs read better top-down; chains read better left to right.
    return to_mermaid(graph, "TD" if family == "orchestrator_workers" else "LR")


def _latest(races: list[RaceResult], roles: list[str], design_id: str) -> dict | None:
    """A design's measurements from the most recent race it ran in."""
    for result, role in zip(reversed(races), reversed(roles)):
        found = next((r for r in result.results if r.design.id == design_id), None)
        if found is not None:
            return _measured(found, result, role)
    return None


def _measured(r: DesignResult, result: RaceResult, role: str) -> dict:
    lo, hi = r.accuracy_ci(result.config.confidence, result.config.seed)
    return {
        "id": r.design.id,
        "family": r.design.family,
        "model": r.design.model,
        "accuracy": finite(r.accuracy),
        "accuracy_ci": [finite(lo), finite(hi)],
        "cost_per_1k_usd": r.cost_per_1k,
        "p50_latency_s": finite(r.p50_latency_s),
        "n": r.n,
        "unmeasured": len(r.unmeasured),
        "eliminated": not r.alive,
        "elimination_reason": r.elimination_reason,
        "measured_in": role,
    }


def _strip_outputs(d: dict) -> dict:
    """A race dict without per-example outputs, for the history of earlier races."""
    return {**d, "designs": [{k: v for k, v in row.items() if k != "outputs"} for row in d["designs"]]}


def append_to_atlas(races: list[RaceResult], provenance: str, path: str | Path) -> int:
    """Log each design once: every candidate from the first race, then designs first seen in later races (bred ones)."""
    seen: set[str] = set()
    records = []
    for result in races:
        # A race the spending cap stopped is labelled so it is never read back as a measured outcome.
        for record in outcome_records(result, f"{provenance}-incomplete" if result.aborted else provenance):
            if record["design"]["id"] not in seen:
                seen.add(record["design"]["id"])
                records.append(record)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")
    return len(records)


# ---------------------------------------------------------------- verdict and limitations


def _verdict(
    races: list[RaceResult],
    roles: list[str],
    provenance: str,
    candidates: list[CandidateDesign],
    breeding: BreedingResult | None,
    advisor: AdvisorPick | None,
) -> dict:
    final = races[-1]
    by_id = {c.design.id: c for c in candidates}
    headline = final.verdict if provenance == "measured" else f"[MOCK, not a result] {final.verdict}"
    if final.aborted:
        headline = f"[INCOMPLETE] {headline}"
    in_race = len(final.task.examples)
    out: dict = {
        "tested": True,
        "provenance": provenance,
        "headline": headline,
        "text": final.verdict,
        "recommended_id": final.recommended_id,
        "best_id": final.best_id,
        "tied_ids": final.tied_ids,
        "inconclusive_ids": final.inconclusive_ids,
        "aborted": final.aborted,
        # The examples the recommendation was actually measured on; a race stops once one design is left.
        "n_examples": final.get(final.recommended_id).n if final.recommended_id else in_race,
        "examples_in_race": in_race,
        "decided_in": roles[-1],
    }
    if final.recommended_id:
        rec = _measured(final.get(final.recommended_id), final, roles[-1])
        lineage = (breeding.lineage if breeding else {}).get(final.recommended_id)
        rec["origin"] = "bred" if lineage else by_id[final.recommended_id].role if final.recommended_id in by_id else "candidate"
        rec["based_on"] = by_id[final.recommended_id].based_on if final.recommended_id in by_id else []
        rec["lineage"] = lineage
        out["recommended"] = rec
    # Numbers from another race (other split, other n, cut early for looking low) are not
    # comparable with the recommendation's, so each comparison says whether it is.
    default = next((c for c in candidates if c.role == "llm-default"), None)
    if default and default.design.id != final.recommended_id:
        found = _latest(races, roles, default.design.id)
        if found:
            found["comparable"] = found["measured_in"] == roles[-1]
            out["vs_llm_default"] = found
    if advisor is not None:
        pick: dict = {"model": advisor.model, "design_id": advisor.design_id, "matches_recommendation": advisor.design_id == final.recommended_id}
        found = _latest(races, roles, advisor.design_id) if advisor.design_id else None
        if found:
            found["comparable"] = found["measured_in"] == roles[-1]
            pick["measured"] = found
        out["advisor_pick"] = pick
    return out


def _overlaps(a: tuple[float, float], b: tuple[float, float]) -> bool:
    return a[0] <= b[1] and b[0] <= a[1]


def build_limitations(
    *,
    mode: str,
    provenance: str | None,
    research_result: ResearchResult,
    candidates: list[CandidateDesign],
    design_warnings: list[str],
    task: TaskSpec | None,
    races: list[RaceResult],
    breeding: BreedingResult | None,
    plain_available: bool,
    advisor_available: bool,
    real_cost_usd: float,
    infra_notes: list[str] | None = None,
) -> list[str]:
    """'When not to trust this report', computed from what actually happened in this run."""
    out: list[str] = []
    if provenance == MOCK:
        out.append(
            "MOCK race: answers were simulated from your labels at fixed per-tier accuracy rates. The numbers test "
            "the plumbing only and say nothing about how Nemotron performs on your task. Never quote them."
        )
    if mode == "offline":
        out.append(
            "OFFLINE mode: real paper search, but the search plan, method cards and candidate designs come from "
            "keyword heuristics and templates, not Nemotron. Family labels can be wrong."
        )
    cards = research_result.cards
    if not cards:
        out.append("The search returned no sources, so the candidates are generic templates, not grounded in research.")
    else:
        out.append(
            "Method cards summarise abstracts and search snippets only; full papers were not read. "
            "Reported results are the authors' claims, not re-measured here."
        )
        unverified = sum(not c.verified for c in cards)
        if unverified:
            out.append(f"{unverified} of {len(cards)} sources have no resolvable arXiv ID or DOI (web pages or unindexed); treat them as leads.")
        if mode == "nemotron":
            fallback = sum(c.extracted_by == OFFLINE for c in cards)
            if fallback:
                out.append(f"{fallback} of {len(cards)} method cards were written by the offline heuristic instead of Nemotron (see warnings).")
            if research_result.spec.produced_by == OFFLINE:
                out.append("The search plan fell back to the offline heuristic because Nemotron's plan could not be parsed.")
    generated = [c for c in candidates if c.role == "candidate"]
    if mode == "nemotron" and any(c.generated_by == OFFLINE for c in generated):
        n_off = sum(c.generated_by == OFFLINE for c in generated)
        out.append(f"{n_off} of {len(generated)} generated candidates fell back to built-in templates (see warnings).")
    if design_warnings:
        out.append(f"{len(design_warnings)} design slot(s) were dropped or replaced; see warnings.")

    if task is None or not races:
        out.append("No labelled examples were given, so nothing was raced. The candidates are untested suggestions.")
    else:
        final = races[-1]
        cfg = final.config
        in_race = len(final.task.examples)
        n = final.get(final.recommended_id).n if final.recommended_id else in_race
        if n < in_race:
            out.append(
                f"The recommendation was measured on only {n} of the deciding race's {in_race} examples: the race stops "
                "spending once one design is left, so its accuracy is the estimate from those examples only."
            )
        if n < 100:
            out.append(
                f"The deciding race used only {n} examples. Confidence intervals are wide, so small accuracy gaps are "
                "noise. Use 200+ real examples before you commit."
            )
        aborted = [result.aborted for result in races if result.aborted]
        if aborted:
            out.append(
                f"INCOMPLETE: a race stopped early because {aborted[0]}. Refused runs were not scored, and there is no "
                "measured recommendation from that race. Raise the cap or race fewer designs."
            )
        if final.recommended_id is None:
            if not final.aborted:
                out.append("No design met the task's cost and latency constraints, so there is no recommendation.")
        else:
            if len(final.tied_ids) > 1:
                others = [i for i in final.tied_ids if i != final.recommended_id]
                out.append(
                    f"{len(final.tied_ids)} designs are statistically tied ({', '.join(final.tied_ids)}). The pick is the "
                    f"cheapest of them, not a proven accuracy winner over {', '.join(others)}."
                )
            if final.inconclusive_ids:
                out.append(
                    f"Inconclusive: {', '.join(final.inconclusive_ids)} could be neither shown worse than the top scorer "
                    f"nor shown within {cfg.tie_margin:.2f} accuracy of it. More examples could change the recommendation."
                )
            rec = final.get(final.recommended_id)
            rec_ci = rec.accuracy_ci(cfg.confidence, cfg.seed)
            overlapping = [
                r.design.id
                for r in final.results
                if r is not rec and r.design.id not in final.tied_ids and r.n and _overlaps(rec_ci, r.accuracy_ci(cfg.confidence, cfg.seed))
            ]
            if overlapping:
                out.append(
                    f"The 95% accuracy intervals of {', '.join(overlapping)} overlap the recommendation's; they were "
                    "separated by a paired test or a constraint, and the ranking could shift with more data."
                )
        early = [(r, len(result.task.examples)) for result in races for r in result.results if not r.alive and 0 < r.n < len(result.task.examples)]
        if early:
            smallest, total = min(early, key=lambda pair: pair[0].n)
            out.append(
                f"Some designs were eliminated early, {smallest.design.id} after only {smallest.n} of {total} examples. "
                "Each elimination is a test corrected for the number of comparisons and checks, which keeps but does not "
                "remove the chance of cutting a slow starter."
            )
        errored = list(dict.fromkeys(r.design.id for result in races for r in result.results if r.errors))
        if errored:
            out.append(f"Some runs raised errors and were scored as wrong: {', '.join(errored)}.")
        unmeasured = list(dict.fromkeys(r.design.id for result in races for r in result.results if r.unmeasured))
        if unmeasured:
            out.append(
                f"Some runs failed for infrastructure reasons (spending cap, API errors) and were left out of the scores, "
                f"so these designs were measured on fewer examples: {', '.join(unmeasured)}."
            )
        if breeding is not None:
            if breeding.bred and not breeding.holdout_ids:
                out.append(
                    f"Bred designs ({', '.join(b.design.id for b in breeding.bred)}) were chosen from failures on the same "
                    "examples they were raced on, so their scores are optimistic until confirmed on held-out data."
                )
            out.append(f"Breeding: {breeding.note}")
        if task.provenance:
            out.append(f"Data: {task.provenance}")
        out.append(f"Scored with the '{task.scorer}' scorer; it measures only what that scorer checks.")
        out.append("Measured on these examples only. If live traffic looks different, re-race on a sample of it.")
        if provenance == "measured":
            out.append("Costs use Token Factory list prices and measured token counts. Latency depends on load at run time.")
    out += infra_notes or []
    if not plain_available:
        out.append("The plain-LLM answer ('without') needs Nemotron and was not produced in this run.")
    if not advisor_available:
        out.append("The 'what would an LLM pick' baseline needs Nemotron and was not run.")
    out.append(
        "Only the built-in architecture families are raced. A design outside them (fine-tuning, retrieval over "
        "your own data, tools) may beat all of these."
    )
    if real_cost_usd:
        out.append(f"This run spent ${real_cost_usd:.4f} on Nemotron calls.")
    return out


# ---------------------------------------------------------------- the engine


class _Stages:
    """Times each stage, reports it to the progress callback and keeps the model receipt."""

    def __init__(self, progress: Progress | None):
        self.progress = progress
        self.receipt: list[ReceiptEntry] = []
        self._t0 = 0.0
        self._current = ""

    def start(self, stage: str) -> None:
        self._current, self._t0 = stage, time.perf_counter()
        if self.progress:
            self.progress(stage, "running")

    def done(self, produced_by: str, cost_usd: float = 0.0, *, status: str = "done", detail: str = "", **extra) -> None:
        entry = ReceiptEntry(self._current, produced_by, cost_usd, time.perf_counter() - self._t0, status, detail, **extra)
        self.receipt.append(entry)
        if self.progress:
            self.progress(self._current, status)

    def skip(self, stage: str, detail: str) -> None:
        self.start(stage)
        self.done("none", status="skipped", detail=detail)

    def failed(self) -> None:
        if self.progress and self._current:
            self.progress(self._current, "failed")


def run_wind_tunnel(
    problem: str,
    task: TaskSpec | None,
    client: LLMClient | None = None,
    *,
    providers: dict[str, Provider] | None = None,
    fetcher: Fetcher | None = None,
    breed: bool = True,
    config: RaceConfig | None = None,
    progress: Progress | None = None,
    max_sources: int = 15,
    max_designs: int = 6,
    generations: int = 1,
    breed_per_generation: int = 3,
    atlas_path: str | Path | None = None,
) -> dict:
    """Run every stage and return the report dict.

    `client` must already be wrapped in BudgetGuard if it is real. None (or a MockClient)
    means offline: heuristics for the model stages and a mock race labelled "mock".
    With `atlas_path`, raced designs are appended to the Outcome Atlas with their provenance.
    """
    stages = _Stages(progress)
    try:
        return _run(
            problem, task, client, stages, providers, fetcher, breed, config, max_sources, max_designs, generations,
            breed_per_generation, atlas_path,
        )
    except BaseException:
        stages.failed()
        raise


def _run(
    problem, task, client, stages, providers, fetcher, breed, config, max_sources, max_designs, generations,
    breed_per_generation, atlas_path,
) -> dict:
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    cfg = config or RaceConfig()
    # A mock can simulate race answers but cannot write plans, cards or designs, so it means offline.
    model_client = None if client is None or is_mock(client) else client
    mode = "offline" if model_client is None else "nemotron"
    warnings: list[str] = []
    infra_notes: list[str] = []  # stages the spending cap or an API failure cut short, for the limitations
    if task is not None and not task.examples:
        warnings.append(f"task {task.name!r} has no examples, so nothing was raced")
        task = None

    # Without: a plain LLM answering from memory.
    stages.start("plain_llm_answer")
    try:
        plain = plain_llm_answer(problem, model_client)
    except BudgetExceeded:
        plain = plain_llm_answer(problem, None)
        infra_notes.append("The spending cap was reached before the plain-LLM answer, so it was not produced.")
    if plain.available:
        stages.done(plain.model, plain.cost_usd, detail="no search, no tests")
    else:
        stages.done("none", status="skipped", detail="needs Nemotron" if model_client is None else "budget exhausted")

    # L1-L3: understand, scout, extract. A recorder keeps the spend on replies that were rejected.
    stages.start("research")
    recorder = Recorder(model_client) if model_client is not None else None
    rejected = ""
    try:
        found = research(problem, recorder, providers=providers, max_sources=max_sources, fetcher=fetcher)
    except (ValueError, TypeError, AttributeError, KeyError, BudgetExceeded) as exc:  # Ultra's plan was unusable
        warnings.append(
            f"research plan from Nemotron was unusable ({type(exc).__name__}: {exc}); fell back to the offline heuristic"
        )
        found = research(problem, None, providers=providers, max_sources=max_sources, fetcher=fetcher)
        found.spec.cost_usd = recorder.cost_usd  # paid for, even though unusable
        rejected = "+".join(f"{m}(rejected)" for m in recorder.models)
    warnings += found.warnings
    verified = sum(c.verified for c in found.cards)
    stages.done(
        _join([rejected, found.spec.produced_by, *(c.extracted_by for c in found.cards)]),
        recorder.cost_usd if recorder is not None else found.cost_usd,
        detail=f"{len(found.scout.sources)} sources, {len(found.cards)} method cards, {verified} with verified IDs",
    )

    # L4: candidate designs, each with the graph its diagram is drawn from.
    stages.start("design_candidates")
    design_warnings: list[str] = []
    # Metered here too: a dropped wildcard slot or near-duplicate was still paid for.
    recorder = Recorder(model_client) if model_client is not None else None
    candidates = design_candidates(found.spec, found.cards, recorder, max_designs=max_designs, warnings=design_warnings)
    warnings += design_warnings
    stages.done(
        _join(c.generated_by for c in candidates),
        recorder.cost_usd if recorder is not None else 0.0,
        detail=f"{len(candidates)} candidates",
    )

    # L5: the race, with diagnose-and-breed.
    races: list[RaceResult] = []
    roles: list[str] = []
    breeding: BreedingResult | None = None
    provenance: str | None = None
    race_cost = 0.0
    if task is None:
        stages.skip("race", "no labelled examples")
    else:
        stages.start("race")
        if model_client is None:
            race_client = unwrap(client) if is_mock(client) else MockClient.for_task(task, seed=cfg.seed)
        else:
            race_client = model_client
        provenance = MOCK if is_mock(race_client) else "measured"
        designs = [c.design for c in candidates]
        diag_cost = 0.0
        diagnosed_by = None
        if breed:
            breeding = race_with_breeding(
                task,
                designs,
                race_client,
                cfg,
                generations=generations,
                breed_per_generation=breed_per_generation,
                diagnose_with_model=model_client is not None,
            )
            races, roles = list(breeding.races), list(breeding.race_roles)
            diags = [d for gen in breeding.diagnoses for d in gen]
            diag_cost = sum(d.cost_usd for d in diags)
            diagnosed_by = _join(p.split(" (")[0] for d in diags for p in d.produced_by.split("+")) if diags else None
        else:
            races, roles = [race(task, designs, race_client, cfg)], ["initial"]
        simulated = sum(r.total_cost_usd for r in races)
        race_cost = 0.0 if provenance == MOCK else simulated
        tiers = sorted({m for r in races for res in r.results for m in graph_for(res.design).models()}, key=list(CATALOG).index)
        stages.done(
            MOCK if provenance == MOCK else "+".join(tiers),
            race_cost + diag_cost,
            detail=f"{len(races)} race(s), {sum(r.total_calls for r in races)} calls"
            + (f", {len(breeding.bred)} bred design(s)" if breeding else "")
            + (f", failures diagnosed by {diagnosed_by}" if diagnosed_by else ""),
            simulated_cost_usd=simulated if provenance == MOCK else None,
        )

    # Baseline: which design would an LLM have picked without measuring?
    advisor: AdvisorPick | None = None
    advisor_failed = ""
    if model_client is None:
        stages.skip("llm_pick", "needs Nemotron")
    else:
        stages.start("llm_pick")
        advisor_task = task or TaskSpec("custom", problem, "", "exact", [])
        try:
            advisor = llm_pick(advisor_task, [c.design for c in candidates], model_client)
        except Exception as exc:  # the race is already paid for; never lose the report over the baseline
            advisor_failed = "budget exhausted" if isinstance(exc, BudgetExceeded) else f"{type(exc).__name__}"
            warnings.append(f"LLM advisor call failed ({advisor_failed}); the advisor baseline was skipped")
            infra_notes.append(f"The 'what would an LLM pick' baseline failed ({advisor_failed}) and is missing.")
            stages.done("none", status="skipped", detail=advisor_failed)
        else:
            stages.done(advisor.model, advisor.cost_usd, detail=f"picked {advisor.design_id or 'no valid design'}")
    if breeding is not None:
        for d in (d for gen in breeding.diagnoses for d in gen):
            if "budget exhausted" in d.produced_by or "call failed" in d.produced_by:
                infra_notes.append(f"Failure diagnosis for {d.design_id} fell back to the offline heuristic: {d.produced_by}.")

    # L6: limitations, computed from this run.
    stages.start("limitations")
    real_cost = sum(e.cost_usd for e in stages.receipt)
    limits = build_limitations(
        mode=mode,
        provenance=provenance,
        research_result=found,
        candidates=candidates,
        design_warnings=design_warnings,
        task=task,
        races=races,
        breeding=breeding,
        plain_available=plain.available,
        advisor_available=advisor is not None or bool(advisor_failed),
        real_cost_usd=real_cost,
        infra_notes=infra_notes,
    )
    stages.done(OFFLINE, detail=f"{len(limits)} limitations")

    raced_ids = {r.design.id for result in races for r in result.results}
    cand_dicts = []
    for c in candidates:
        d = c.to_dict()
        d["mermaid"] = mermaid_for(c.graph, c.design.family)
        d["tiers"] = c.graph.models()
        d["raced"] = c.design.id in raced_ids
        cand_dicts.append(d)

    race_d = None
    if races:
        race_d = {"role": roles[-1], **race_to_dict(races[-1], advisor, provenance)}
    breeding_d = None
    if breeding is not None:
        bred = []
        for b in breeding.bred:
            d = b.to_dict()
            d["mermaid"] = mermaid_for(b.graph, b.design.family)
            bred.append(d)
        breeding_d = {
            "enabled": True,
            "note": breeding.note,
            "selection_bias": "controlled by a held-out split" if breeding.holdout_ids else "present: bred designs' scores are optimistic",
            "bred": bred,
            "lineage": breeding.lineage,
            "diagnoses": [[d.to_dict() for d in gen] for gen in breeding.diagnoses],
            "search_ids": breeding.search_ids,
            "holdout_ids": breeding.holdout_ids,
            "races": [{"role": role, **_strip_outputs(race_to_dict(r, None, provenance))} for role, r in zip(roles, races)],
        }
    elif task is not None:
        breeding_d = {"enabled": False, "note": "Breeding was turned off for this run.", "bred": [], "lineage": {}}

    verdict = (
        _verdict(races, roles, provenance, candidates, breeding, advisor)
        if races
        else {"tested": False, "provenance": "untested", "headline": "No labelled examples: the candidates are untested suggestions."}
    )
    if model_client is None:
        default = next((c for c in candidates if c.role == "llm-default"), None)
        advisor_d = {
            "available": False,
            "model": "ultra",
            "design_id": None,
            "typical_default_id": default.design.id if default else None,
            "reply": "Needs Nemotron (offline mode).",
            "cost_usd": 0.0,
        }
    elif advisor is None:
        advisor_d = {"available": False, "model": "ultra", "design_id": None, "reply": f"Advisor call failed ({advisor_failed}).", "cost_usd": 0.0}
    else:
        advisor_d = {"available": True, "model": advisor.model, "design_id": advisor.design_id, "reply": advisor.reply, "cost_usd": advisor.cost_usd}

    atlas_d = None
    if atlas_path and races:
        atlas_d = {"path": str(atlas_path), "records": append_to_atlas(races, provenance, atlas_path), "provenance": provenance}

    return {
        "problem": problem,
        "mode": mode,
        "provenance": provenance or "untested",
        "started_at": started,
        "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "task": None
        if task is None
        else {"name": task.name, "n_examples": len(task.examples), "labels": task.labels, "scorer": task.scorer, "provenance": task.provenance},
        "plain_answer": plain.to_dict(),
        "research": found.to_dict(),
        "candidates": cand_dicts,
        "race": race_d,
        "breeding": breeding_d,
        "advisor": advisor_d,
        "verdict": verdict,
        "limitations": limits,
        "warnings": warnings,
        "model_receipt": [e.to_dict() for e in stages.receipt],
        "total_cost_usd": round(real_cost, 6),
        "atlas": atlas_d,
    }


# ---------------------------------------------------------------- text report


def _num(x: float | None) -> str:
    return "n/a" if x is None else f"{x:.2f}"


def _clip(text: str, n: int) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= n else text[: n - 1].rstrip() + "…"


def format_report(report: dict) -> str:
    """A compact human-readable report for the terminal."""
    race_d = report.get("race")
    prov = report["provenance"]
    lines = [f"WIND TUNNEL   mode: {report['mode']}   race: {prov.upper() if prov == MOCK else prov}", f"Problem: {_clip(report['problem'], 200)}", ""]

    plain = report["plain_answer"]
    lines.append("WITHOUT (plain LLM: no search, no tests)")
    lines.append(f"  {_clip(plain['text'], 500)}")
    lines.append("WITH WIND TUNNEL")
    lines.append(f"  {report['verdict']['headline']}")

    cards = report["research"]["cards"]
    lines += ["", f"TOP METHODS FOUND ({len(cards)} cards, {sum(c['verified'] for c in cards)} with verified IDs)"]
    for c in cards[:5]:
        mark = "✓" if c["verified"] else "web" if "web" in c["flags"] else "?"
        lines.append(f"  {mark} [{c['family']}] {_clip(c['method_name'], 80)} ({c['citation'].get('year') or 'n.d.'})  {c['citation'].get('url')}")
    if not cards:
        lines.append("  (no sources found)")

    lines += ["", "CANDIDATES"]
    for c in report["candidates"]:
        lines.append(f"  {c['id']:<26} {c['role']:<11} by {c['generated_by']:<17} {_clip(c['description'], 70)}")

    if race_d:
        lines += ["", f"LEADERBOARD ({race_d['role']}, {race_d['task']['n_examples']} examples, provenance={race_d['provenance']})"]
        rows = sorted(race_d["designs"], key=lambda r: (r["eliminated_round"] is not None, -(r["accuracy"] or 0.0)))
        for r in rows:
            mark = "*" if r["id"] == race_d["recommended_id"] else "~" if r["id"] in race_d["tied_ids"] else " "
            lo, hi = r["accuracy_ci"]
            status = "alive" if r["eliminated_round"] is None else f"out r{r['eliminated_round']}: {_clip(r['elimination_reason'], 50)}"
            if r.get("unmeasured"):
                status += f" ({len(r['unmeasured'])} unmeasured)"
            lines.append(
                f"  {mark} {r['id']:<30} {_num(r['accuracy'])} ({_num(lo)}-{_num(hi)})  ${r['cost_per_1k_usd']:.3f}/1k  "
                f"p50 {_num(r['p50_latency_s'])}s  n={r['n']}  {status}"
            )
        lines.append(f"  * recommended   ~ tied with the best (within {race_d['config']['tie_margin']:.2f} accuracy)")
        earlier = (report.get("breeding") or {}).get("races", [])[:-1]
        final_ids = {r["id"] for r in race_d["designs"]}
        dropped = {}
        for past in earlier:
            for r in past["designs"]:
                if r["id"] not in final_ids and r["eliminated_round"] is not None:
                    dropped[r["id"]] = (past["role"], r)
        for design_id, (role, r) in dropped.items():
            lines.append(f"    {design_id:<30} {_num(r['accuracy'])}  n={r['n']}  out in {role}: {_clip(r['elimination_reason'], 60)}")
    breeding = report.get("breeding")
    if breeding and breeding.get("bred"):
        lines += ["", "BRED DESIGNS"]
        for b in breeding["bred"]:
            m = b["mechanism"]
            lines.append(f"  {b['id']:<30} from {b['parent_id']} via {m['name']} ({m['field']}) for {b['failure_mode']}")

    verdict = report["verdict"]
    lines += ["", "VERDICT", f"  {verdict['headline']}"]
    adv = report["advisor"]
    if adv["available"]:
        lines.append(f"  LLM advisor ({adv['model']}) would have picked: {adv['design_id'] or 'no valid design'}")
        measured = verdict.get("advisor_pick", {}).get("measured")
        if measured and verdict.get("recommended") and measured["id"] != verdict["recommended"]["id"]:
            rec = verdict["recommended"]
            kind = "simulated" if prov == MOCK else "measured"
            if measured.get("comparable"):
                lines.append(
                    f"    {kind} in the same race: accuracy {_num(measured['accuracy'])} (n={measured['n']}) vs "
                    f"{_num(rec['accuracy'])} (n={rec['n']}), ${measured['cost_per_1k_usd']:.3f} vs ${rec['cost_per_1k_usd']:.3f} per 1k"
                )
            else:
                lines.append(
                    f"    {kind} in {measured['measured_in']}: accuracy {_num(measured['accuracy'])} on n={measured['n']}; "
                    f"not comparable with the recommendation's {_num(rec['accuracy'])} (n={rec['n']}, {verdict['decided_in']})"
                )
    elif adv.get("typical_default_id"):
        lines.append(f"  (LLM advisor not run offline; the typical LLM default here is {adv['typical_default_id']})")

    lines += ["", "LIMITATIONS"] + [f"  - {item}" for item in report["limitations"]]
    lines += ["", "MODEL RECEIPT"]
    for e in report["model_receipt"]:
        sim = f"  (simulated ${e['simulated_cost_usd']:.4f})" if e.get("simulated_cost_usd") is not None else ""
        lines.append(f"  {e['stage']:<18} {e['status']:<8} {e['produced_by']:<30} ${e['cost_usd']:.4f}  {e['seconds']:.2f}s{sim}")
    lines.append(f"  total real spend: ${report['total_cost_usd']:.4f}")
    return "\n".join(lines)
