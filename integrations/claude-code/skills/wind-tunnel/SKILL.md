---
name: wind-tunnel
description: Before choosing an AI architecture, agent pattern, prompting strategy or model size for a feature (classification, extraction, routing, RAG, agents, "should this be multi-agent?", "which model should I use?"), call the Wind Tunnel MCP tools to get cited methods and a measured answer instead of guessing.
---

# Wind Tunnel: measure the AI architecture before building it

Use this whenever you are about to pick how the AI part of a feature works: one model call or
several, self-check or voting, planner/executor, multi-agent, which Nemotron tier. Opinions
(including yours) are a poor predictor of which design wins, so ask Wind Tunnel first.

1. **Research.** Call `find_methods` with a plain-English problem: inputs, desired output, and
   any cost or latency limits. Summarise the method families and the 3-5 most relevant sources.
   Say which citations are `verified`.
2. **Optional:** `design_candidates` shows the runnable candidates with Mermaid diagrams, and
   `compare_with_plain_llm` shows the "without vs with Wind Tunnel" view.
3. **Examples.** A race needs 30-100 labelled `{"input", "expected"}` pairs (at least 5).
   Look for fixtures, logs or test data in the repo first, then ask the user. Never invent the
   expected answers yourself. Pass `labels` when the answer set is fixed.
   Tell the user the examples are sent to the Wind Tunnel server and, on real runs, to Nebius
   Token Factory.
4. **Race.** Call `start_race`. It returns a `job_id` at once. It runs the whole pipeline:
   research, candidates, the race, then diagnose-and-breed (`breed: true` by default; pass
   `breed: false` for a single race).
5. **Report.** Poll `get_report` until `status` is `done`: every 30-60 s for real races (they
   take minutes), or after a few seconds for mock races. `progress.stages` shows which stage is
   running.
6. **Read `result.limitations` before acting on the verdict.** Every time, before you summarise,
   recommend or write code. They are computed from this run and say when not to trust it: too
   few examples, inconclusive or tied designs, early eliminations, errors, a spending cap that
   stopped the race, and bred designs whose scores are optimistic. Pass the relevant ones on to
   the user with the verdict.
7. **Be honest about mode.** Check `result.labels`. If it contains `MOCK` or `SIMULATED`, the
   race answers were simulated: say so plainly, and do not quote the numbers or choose a design
   from them (`do_not_quote` is true). If it contains `OFFLINE`, the method cards and candidates
   came from heuristics, not Nemotron. If `plain_llm` or `advisor` has `available: false`, say
   it was not produced; never write a stand-in answer yourself.
8. **Breeding.** If `result.breeding.bred` is not empty, tell the user which failure each bred
   design targets (`mechanisms_applied`: mechanism, field, parent). If `selection_bias` starts
   with "present", their scores are optimistic until confirmed on held-out examples.
9. **Build.** Only for a `MEASURED` result with a `recommended` design: implement it by
   following its Mermaid diagram and `tiers` (same steps, same model tiers, Nemotron via Nebius
   Token Factory; a cascade needs both its cheap and its escalation tier). Add the race examples
   as a regression test. Mention what the LLM advisor would have picked and the limitations.
