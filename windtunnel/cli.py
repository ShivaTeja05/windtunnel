"""Command line: `windtunnel run`, `race`, `research`, `models` and `budget`."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

from .advisor import llm_pick
from .atlas import append_outcomes
from .budget import BudgetGuard
from .designs import default_designs, load_designs
from .engine import format_report, run_wind_tunnel
from .llm import MockClient, TokenFactoryClient
from .models import CATALOG
from .racer import RaceConfig, race
from .report import leaderboard, to_dict
from .research import format_research, research
from .scout import PROVIDERS
from .task import TaskSpec, load_task

TASKS_DIR = Path(__file__).resolve().parent.parent / "tasks"


def load_dotenv(path: str | Path = ".env") -> None:
    """Read KEY=value lines into the environment without overriding variables already set."""
    path = Path(path)
    if not path.is_file():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        value = value.strip().strip('"').strip("'")
        if value:
            os.environ.setdefault(key.strip(), value)


def _cmd_race(args: argparse.Namespace) -> int:
    task = load_task(args.task_dir)
    if args.limit:
        task.examples = task.examples[: args.limit]
    designs = load_designs(args.designs) if args.designs else default_designs()
    client = MockClient.for_task(task, seed=args.seed) if args.mock else BudgetGuard(TokenFactoryClient(thinking=args.thinking))
    provenance = "mock" if args.mock else "measured"
    if args.mock:
        print("MOCK RUN: answers are simulated. These numbers test the plumbing and are not results.\n", file=sys.stderr)

    # The mock cannot judge designs (it always picks the first), so a mock run has no advisor, as in the engine.
    advisor = None if args.no_advisor or args.mock else llm_pick(task, designs, client, model=args.advisor_model)
    config = RaceConfig(mode=args.mode, max_workers=args.max_workers, seed=args.seed)
    result = race(task, designs, client, config)
    print(leaderboard(result, advisor, provenance))

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out = Path(args.out) / f"{task.name}-{args.mode}-{provenance}-{stamp}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(to_dict(result, advisor, provenance), indent=2))
    n = append_outcomes(result, args.atlas, provenance)
    print(f"\nWrote {out} and appended {n} outcomes to {args.atlas}")
    if isinstance(client, BudgetGuard):
        print(f"Spend so far: ${client.spent_usd:.4f} of your ${client.limit_usd:.2f} cap")
    return 0


def _cmd_research(args: argparse.Namespace) -> int:
    offline = args.offline or not os.environ.get("NEBIUS_API_KEY")
    if offline:
        print("OFFLINE: real paper search, but plan and method cards come from heuristics, not Nemotron.\n", file=sys.stderr)
    client = None if offline else BudgetGuard(TokenFactoryClient())
    providers = {name: PROVIDERS[name] for name in args.providers.split(",")}
    result = research(args.problem, client, providers=providers, max_sources=args.max_sources)
    print(format_research(result))
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out = Path(args.out) / f"research-{result.mode}-{stamp}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result.to_dict(), indent=2))
    print(f"\nWrote {out}")
    return 0


def _sample_dir(name: str) -> Path:
    """A bundled sample task: ./tasks/<name> first, then the repo's tasks folder."""
    for base in (Path("tasks"), TASKS_DIR):
        if (base / name / "task.json").is_file():
            return base / name
    raise SystemExit(f"unknown sample {name!r}; look in {TASKS_DIR}")


def _cmd_run(args: argparse.Namespace) -> int:
    task: TaskSpec | None = None
    if args.sample or args.task_dir:
        task = load_task(_sample_dir(args.sample) if args.sample else args.task_dir)
        if args.limit:
            task.examples = task.examples[: args.limit]
    offline = args.offline or not os.environ.get("NEBIUS_API_KEY")
    if offline:
        print(
            "OFFLINE: real paper search, heuristic plan/cards/designs, and a MOCK race whose numbers are not results.\n",
            file=sys.stderr,
        )
    client = None if offline else BudgetGuard(TokenFactoryClient())
    providers = {name: PROVIDERS[name] for name in args.providers.split(",")}

    def progress(stage: str, status: str) -> None:
        if status != "running":
            print(f"  [{status:>7}] {stage}", file=sys.stderr)

    report = run_wind_tunnel(
        args.problem,
        task,
        client,
        providers=providers,
        breed=not args.no_breed,
        config=RaceConfig(max_workers=args.max_workers, seed=args.seed),
        progress=progress,
        max_sources=args.max_sources,
        generations=args.generations,
        atlas_path=args.atlas or None,
    )
    print(format_report(report))
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out = Path(args.out) / f"run-{task.name if task else 'untested'}-{report['mode']}-{report['provenance']}-{stamp}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    atlas = report["atlas"]
    print(f"\nWrote {out}" + (f" and appended {atlas['records']} {atlas['provenance']} outcomes to {atlas['path']}" if atlas else ""))
    if isinstance(client, BudgetGuard):
        print(f"Spend so far: ${client.spent_usd:.4f} of your ${client.limit_usd:.2f} cap")
    return 0


def _cmd_budget(args: argparse.Namespace) -> int:
    guard = BudgetGuard(MockClient({}, []))
    print(f"Spent ${guard.spent_usd:.4f} of ${guard.limit_usd:.2f} cap (${guard.remaining_usd:.4f} left).")
    print("Change the cap with WINDTUNNEL_BUDGET_USD in .env.")
    return 0


def _cmd_models(args: argparse.Namespace) -> int:
    for m in CATALOG.values():
        print(f"{m.key:<10} {m.id:<45} ${m.usd_per_m_input:.2f} in / ${m.usd_per_m_output:.2f} out per 1M tokens")
    if args.check:
        live = set(TokenFactoryClient().list_model_ids())
        missing = [m.id for m in CATALOG.values() if m.id not in live]
        print("\nAll catalog IDs are live on Token Factory." if not missing else f"\nNot found on Token Factory: {missing}")
        return 1 if missing else 0
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="windtunnel", description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    w = sub.add_parser("run", help="the whole pipeline: research, design, race, breed, verdict")
    w.add_argument("problem", help="plain-English description of what you are building")
    source = w.add_mutually_exclusive_group()
    source.add_argument("--sample", help="race on a bundled sample task, e.g. ticket_triage")
    source.add_argument("--task-dir", help="race on your own task.json + examples.jsonl")
    w.add_argument("--offline", action="store_true", help="no Nemotron: heuristics and a mock race (labelled mock)")
    w.add_argument("--no-breed", action="store_true", help="skip diagnosing losers and breeding new designs")
    w.add_argument("--generations", type=int, default=1, help="breeding generations")
    w.add_argument("--providers", default="arxiv,openalex,tavily", help="comma list of search providers")
    w.add_argument("--max-sources", type=int, default=15)
    w.add_argument("--limit", type=int, help="only use the first N examples")
    w.add_argument("--max-workers", type=int, default=8)
    w.add_argument("--seed", type=int, default=0)
    w.add_argument("--out", default="runs")
    w.add_argument("--atlas", default="atlas/outcomes.jsonl", help="Outcome Atlas to append to ('' to skip)")
    w.set_defaults(func=_cmd_run)

    r = sub.add_parser("race", help="race designs on a task and print the leaderboard")
    r.add_argument("task_dir", help="directory with task.json and examples.jsonl")
    r.add_argument("--designs", help="JSON list of designs (default: built-in spread)")
    r.add_argument("--mode", choices=["race", "grid"], default="race")
    r.add_argument("--mock", action="store_true", help="offline simulated models, no API key needed")
    r.add_argument("--limit", type=int, help="only use the first N examples")
    r.add_argument("--max-workers", type=int, default=8)
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--thinking", action="store_true", help="enable Nemotron reasoning traces")
    r.add_argument("--no-advisor", action="store_true", help="skip asking the LLM which design it would pick")
    r.add_argument("--advisor-model", default="ultra", choices=sorted(CATALOG))
    r.add_argument("--out", default="runs")
    r.add_argument("--atlas", default="atlas/outcomes.jsonl")
    r.set_defaults(func=_cmd_race)

    m = sub.add_parser("models", help="list the Nemotron models and prices")
    m.add_argument("--check", action="store_true", help="verify the IDs against the live Token Factory catalog")
    m.set_defaults(func=_cmd_models)

    s = sub.add_parser("research", help="find methods for a problem in papers and on the web")
    s.add_argument("problem", help="plain-English description of what you are building")
    s.add_argument("--offline", action="store_true", help="skip Nemotron; heuristics for plan and cards")
    s.add_argument(
        "--providers",
        default="arxiv,openalex,tavily",
        help="comma list from: arxiv, openalex, semantic_scholar (often rate limited without a key), tavily",
    )
    s.add_argument("--max-sources", type=int, default=15)
    s.add_argument("--out", default="runs")
    s.set_defaults(func=_cmd_research)

    b = sub.add_parser("budget", help="show real API spend against the cap")
    b.set_defaults(func=_cmd_budget)

    args = parser.parse_args(argv)
    load_dotenv()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
