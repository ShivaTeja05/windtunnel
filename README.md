# Wind Tunnel

**Don't ask an LLM which AI architecture to build. Measure it.**

Off-the-shelf frontier LLMs predict which approach will work no better than chance
([Wen et al., NeurIPS 2025](https://arxiv.org/abs/2506.00794)), and they all suggest
the same few designs. Wind Tunnel takes your task and real labelled examples, races
genuinely different system designs on NVIDIA Nemotron models served by
[Nebius Token Factory](https://tokenfactory.nebius.com), eliminates clear losers early,
and recommends the **cheapest design that is statistically tied with the best**.
Every result goes into a shared **Outcome Atlas**.

> Status: the whole pipeline runs end to end: research (papers and the web),
> candidate generation by Nemotron Ultra, the race, failure diagnosis and breeding,
> the "what would an LLM pick" baseline, and a report with limitations and a model
> receipt. It runs from the CLI, a web app and an MCP server. See
> [ARCHITECTURE.md](ARCHITECTURE.md) for the layers and what is still next.

## Quickstart

```bash
uv sync
uv run pytest                                          # offline tests, no API key needed

# The whole pipeline: research -> designs -> race -> breed -> verdict
uv run windtunnel run "Route support tickets to one of five queues, under $5 per 1000" --sample ticket_triage
uv run windtunnel run "..." --task-dir my_task/       # your own task.json + examples.jsonl
uv run windtunnel run "..."                            # no examples: research and untested designs only

# Single stages
uv run windtunnel research "Extract totals from scanned invoices"
uv run windtunnel race tasks/ticket_triage --mock      # race the built-in designs
uv run windtunnel budget                               # real spend against your cap
```

`windtunnel run` prints a compact report (plain LLM answer vs Wind Tunnel's verdict,
top methods found, candidates, leaderboard, bred designs, limitations, model receipt)
and writes the full JSON report to `runs/`. Raced designs are appended to the Outcome
Atlas (`atlas/outcomes.jsonl`) with their provenance. Add `--no-breed` to skip
diagnose-and-breed.

### Offline vs Nemotron mode

| | Offline (`--offline`, or no `NEBIUS_API_KEY`) | Nemotron (`NEBIUS_API_KEY` set) |
|---|---|---|
| Plain-LLM answer, advisor pick | skipped, and the report says so | Nemotron Ultra |
| Search plan, method cards | keyword heuristics over a real paper search | Ultra plans, Super extracts |
| Candidate designs | templates mapped from the research | Ultra, one isolated call per slot |
| Race | **mock**: simulated answers, labelled `provenance: "mock"` | **measured** on Nano / Super / Ultra |

Every stage in the model receipt names who produced it: a model key,
`offline-heuristic`, or `mock`. Mock numbers test the plumbing and must never be
quoted as results.

Real runs on Token Factory:

```bash
export NEBIUS_API_KEY=...                              # or put it in .env
uv run windtunnel models --check                       # verify model IDs are live
uv run windtunnel run "..." --sample ticket_triage
```

**Spending cap.** Every real call goes through `BudgetGuard`. Each call reserves its
worst-case cost first and is refused if that would cross the cap. The running total is
kept in `.windtunnel/spend.json`, so the cap holds across runs. The default cap is $10;
set `WINDTUNNEL_BUDGET_USD` in `.env` to change it, and check with `windtunnel budget`.

### Web app and MCP

```bash
uv run uvicorn windtunnel.api:app                      # web app at http://127.0.0.1:8000
uv run windtunnel-mcp                                  # MCP server for Claude Code / Cursor
```

See [docs/MCP.md](docs/MCP.md) for connecting a coding agent.

## How a race works

1. **Designs.** Each design is an architecture family plus a Nemotron tier plus
   parameters. Built-in families: `single_call`, `self_check`, `self_consistency`,
   `plan_execute`, `orchestrator_workers` and `cascade` (a cheap model answers twice
   and escalates to a stronger one only when the answers disagree). Any design can
   carry short task-specific `instructions`. Tiers are Nano, Lightning, Super and Ultra.
   `windtunnel run` generates the designs from the research; `windtunnel race` uses a
   built-in spread or your own list (`--designs designs.json`).
2. **Advisor baseline.** Nemotron Ultra is asked which design it would build. That
   pick is shown next to the measured result (not in mock runs, where there is no model to ask).
3. **Rounds.** Examples are shuffled and revealed in rounds (10, 25, 50, 100, then all).
   After each round, a design is eliminated when an exact sign test on the examples where
   it and the leader disagree shows it is worse, with the 5% error rate split
   (Bonferroni) over every comparison at every check. Designs that break the task's cost
   or latency limits are eliminated too. Budget refusals and API outages are not scored
   as wrong answers: they are left out, and a budget refusal stops the race with no
   recommendation.
4. **Verdict.** Among the survivors, a design is "tied" with the best only when the 95%
   interval of its paired accuracy gap stays within a 0.05 margin (non-inferiority). The
   cheapest tied design is recommended; designs neither shown worse nor shown close are
   reported as inconclusive. Accuracy intervals are Wilson intervals.
5. **Diagnose and breed** (`windtunnel run`). The weaker designs' failures are sorted
   into a small taxonomy (format violations, label confusions, errors, over budget, ...).
   Mechanisms from other fields that address those failures (triple modular redundancy,
   poka-yoke, triage, ...) are applied as mutations, and the bred designs race the
   survivors. With 67+ examples a 30% held-out split is kept for the final race;
   otherwise the report warns that bred designs' scores are optimistic.
6. **Outputs.** The race is saved as `runs/<task>-<mode>-<provenance>-<time>.json`, and
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
  engine.py      run_wind_tunnel: every stage in order, model receipt, limitations
  understand.py  L1 problem -> research plan
  scout.py       L2 arXiv, OpenAlex, Semantic Scholar, Tavily search
  extract.py     L3 sources -> method cards (citations from the index, not the model)
  research.py    L1-L3 together
  generate.py    L4 candidate designs (Ultra, isolated calls) + offline templates
  designs.py     design families and the pipeline runner
  graph.py       design graphs and Mermaid diagrams
  racer.py       rounds, elimination, cost-aware verdict
  breed.py       failure diagnosis, mechanism library, breeding loop
  baseline.py    the plain-LLM answer ("without")
  advisor.py     "what would the LLM pick" baseline
  budget.py      spending cap
  atlas.py       Outcome Atlas records
  report.py      leaderboard and JSON export
  cli.py         windtunnel run / research / race / models / budget
  api.py, jobs.py, mcp_server.py   web app and MCP server
tasks/           task fixtures
tests/           offline tests (no API key needed)
```

## Roadmap

- **Atlas prior:** predict the leaderboard from past outcomes and skip families that
  are known losers.
- **Scale:** Batch API sweeps, Sandboxes for running generated code, Serverless
  Jobs, and one-click deploy of the winner to a Serverless Endpoint.
- **Evidence:** real benchmark tasks; pick regret, race-vs-grid savings, the breeding
  ablation, and the Atlas learning curve.
- **UI:** a live race view, the verdict screen, and an Atlas browser.

## License

MIT
