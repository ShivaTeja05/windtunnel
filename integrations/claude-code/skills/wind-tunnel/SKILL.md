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
4. **Race.** Call `start_race`. It returns a `job_id` at once.
5. **Report.** Poll `get_report` until `status` is `done`: every 30-60 s for real races (they
   take minutes), or after a few seconds for mock races.
6. **Be honest about mode.** If `mode` / `provenance` is `mock`, or research `mode` is
   `offline`, say so plainly. Mock numbers are simulated and must not drive the decision or be
   quoted as results. If `do_not_quote` is true, don't quote the numbers.
7. **Build.** For a measured result, implement the `recommended` design: follow its Mermaid
   diagram (same steps, same model tiers, Nemotron via Nebius Token Factory). Add the race
   examples as a regression test. Mention what the LLM advisor would have picked, and the
   limitations.
