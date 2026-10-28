# Wind Tunnel

**Don't ask an LLM which AI architecture to build. Measure it.**

Off-the-shelf frontier LLMs predict which approach will work no better than chance
([Wen et al., NeurIPS 2025](https://arxiv.org/abs/2506.00794)), and they all suggest
the same few designs. Wind Tunnel takes your task and real labelled examples, races
genuinely different system designs on NVIDIA Nemotron models served by
[Nebius Token Factory](https://tokenfactory.nebius.com), eliminates clear losers early,
and recommends the **cheapest design that is statistically tied with the best**.
Every result goes into a shared **Outcome Atlas**.

> Status: foundation. The race, cost accounting, confidence intervals, the
> LLM-advisor baseline and the Atlas log work end to end. Candidate generation,
> failure diagnosis and breeding, the UI and deployment are next (see Roadmap).

## Quickstart

```bash
uv sync
uv run pytest
uv run windtunnel race tasks/ticket_triage --mock     # offline, simulated models
```

Real run on Token Factory:

```bash
export NEBIUS_API_KEY=...
uv run windtunnel models --check                       # verify model IDs are live
uv run windtunnel race tasks/ticket_triage
```

Mock runs simulate answers. They test the plumbing, are labelled `provenance: "mock"`
in every output, and must never be quoted as results.

## How a race works

1. **Designs.** Each design is an architecture family plus a Nemotron tier plus
   parameters. Built-in families: `single_call`, `self_check`, `self_consistency`,
   `plan_execute` and `orchestrator_workers`. Tiers are Nano, Lightning, Super and Ultra.
   Pass your own list with `--designs designs.json`.
2. **Advisor baseline.** Nemotron Ultra is asked which design it would build. That
   pick is shown next to the measured result.
3. **Rounds.** Examples are shuffled and revealed in rounds (10, 25, 50, 100, then all).
   After each round, any design whose paired accuracy difference against the leader
   has a 95% bootstrap interval entirely below zero is eliminated. Designs that break
   the task's cost or latency limits are eliminated too.
4. **Verdict.** Among the survivors, the designs not clearly worse than the best are
   "tied". The cheapest tied design is recommended.
5. **Outputs.** The run is saved as `runs/<task>-<mode>-<provenance>-<time>.json`, and
   one Atlas record per design is appended to `atlas/outcomes.jsonl`.

`--mode grid` runs every design on every example. It exists so we can measure how
much racing saves.

## Task format

`tasks/<name>/task.json`:

```json
{
  "name": "ticket_triage",
  "description": "Route a customer-support ticket to exactly one queue: ...",
  "output_instructions": "Answer with the queue name only.",
  "scorer": "exact",
  "labels": ["billing", "bug", "account_access", "shipping", "feature_request"],
  "constraints": {"max_cost_per_1k": 5.0, "max_p50_latency_s": 15.0}
}
```

`tasks/<name>/examples.jsonl` has one `{"id", "input", "expected"}` object per line.
The scorers are `exact`, `contains` and `numeric`.

The bundled `ticket_triage` task is a 40-example hand-written development fixture.
It is fine for checking the pipeline but too small for credible numbers.

## Layout

```
windtunnel/
  models.py    Nemotron catalog and prices
  llm.py       Token Factory client (with backoff) and offline mock
  designs.py   design families and the pipeline runner
  racer.py     rounds, elimination, cost-aware verdict
  advisor.py   "what would the LLM pick" baseline
  atlas.py     Outcome Atlas records
  report.py    leaderboard and JSON export
  cli.py       windtunnel race / models
tasks/         task fixtures
tests/         offline tests (no API key needed)
```

## Roadmap

- **Diverge:** Ultra generates task-specific candidates in isolated calls, with a
  diversity check and a wildcard slot.
- **Diagnose and breed:** label each loser's failures, look up mechanisms by failure
  mode (with Tavily-cited sources), and add the adapted design to the race.
- **Atlas prior:** predict the leaderboard from past outcomes and skip families that
  are known losers.
- **Scale:** Batch API sweeps, Sandboxes for running generated code, Serverless
  Jobs, and one-click deploy of the winner to a Serverless Endpoint.
- **Evidence:** real benchmark tasks; pick regret, race-vs-grid savings, the breeding
  ablation, and the Atlas learning curve.
- **UI:** a live race view, the verdict screen, and an Atlas browser.

## License

MIT
