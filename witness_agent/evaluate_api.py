"""The evaluation of a frontier model through OpenRouter: levels beaten per (game, seed) episode.

    [WITNESS_GAME_CATALOG=MODULE] OPENROUTER_API_KEY=... \\
    python -m witness_agent.evaluate_api --model MODEL --tasks PATH --out DIR [--seeds N [N ...]]
           [--games ID [ID ...]] [--effort max] [--max-tokens 128000] [--providers SLUG [SLUG ...]]
           [--concurrency 8] [--resume] [--dry-run]

Every episode is played with observation version 2 by `api_policy.OpenRouterPolicy` behind a
`journal.JournalPolicy`. Before any episode, `api_policy.plan_request` checks the model and its zero-data-retention
providers against OpenRouter's public endpoint lists (no model is called) and fixes the request; `--dry-run` prints
it and stops. At most `--concurrency` requests are in flight at a time.

DIR receives what `witness_agent.evaluate` writes (results.json, results.txt, episodes/, traces/) and
`journal/<game_id>_seed<seed>.jsonl`, every answer appended as it arrives. `--resume` continues a run in DIR: an
episode whose record shows it completed is kept as it is; every other one (aborted, or never ended) is played again
from the start, its journal answering the calls it already paid for. Resume a run with the command that started it
plus --resume: nothing checks that the model, the options and the task rows are the same, and results.json records the
settings of the resuming command. results.json records the request as sent (no messages, no key) and the totals of the
answers: calls, truncated ones, retried ones, tokens, cost, and the models and providers that served them. Exit
status: 0 when every episode completed, 1 when some episode aborted, 2 for invalid inputs and 3 when the model and
provider check fails or OPENROUTER_API_KEY is not set; with 2 and 3 no episode has started.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import threading
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Final

from .api_policy import DEFAULT_EFFORT, DEFAULT_MAX_TOKENS, KEY_ENV, OpenRouterPolicy, RequestPlan, plan_request
from .catalog import catalog_name, make_game
from .episode import OBSERVATION_VERSION, Task, load_tasks
from .evaluate import (EpisodeOutcome, PolicyFactory, format_report, play_episodes, select_tasks, summarize,
                       write_json)
from .http_policy import PreflightError
from .journal import JournalPolicy
from .loop import MAX_STEPS_PER_LEVEL
from .trace import write_utf8

DEFAULT_SEEDS: Final = (0, 1, 2, 3, 4)
DEFAULT_CONCURRENCY: Final = 8
TRUNCATION_GATE: Final = 0.05            # the report warns above this share of calls cut off at the output limit
logger = logging.getLogger(__name__)


def journal_path(out_dir: Path, game_id: str, seed: int) -> Path:
    return out_dir / "journal" / f"{game_id}_seed{seed}.jsonl"


def run(tasks: Sequence[Task], seeds: Sequence[int], policy_for: PolicyFactory, out_dir: Path, *,
        resume: bool) -> tuple[EpisodeOutcome, ...]:
    """Play every (task, seed) episode not already completed in `out_dir` (`resume`) at once
    (`evaluate.play_episodes`), and return the records of all of them in task order, then seed order. Without
    `resume`, `out_dir/episodes` must not exist."""
    episodes_dir = out_dir / "episodes"
    episodes_dir.mkdir(parents=True, exist_ok=resume)
    pairs = [(task, seed) for task in tasks for seed in seeds]
    outcomes: dict[int, EpisodeOutcome] = {}
    for i, (task, seed) in enumerate(pairs):
        record = episodes_dir / f"{task.game_id}_seed{seed}.json"
        if resume and record.exists():
            kept = EpisodeOutcome(**json.loads(record.read_text(encoding="utf-8")))
            if kept.error is None:
                outcomes[i] = kept
    todo = [i for i in range(len(pairs)) if i not in outcomes]
    games = [make_game(pairs[i][0].game_id) for i in todo]
    policies = [policy_for(pairs[i][0].game_id, pairs[i][1]) for i in todo]
    logger.info("%d episodes to play, %d kept from an earlier run", len(todo), len(outcomes))
    outcomes.update(zip(todo, play_episodes([pairs[i] for i in todo], games, policies, out_dir, write_traces=True)))
    return tuple(outcomes[i] for i in range(len(pairs)))


def run_completion_totals(out_dir: Path, tasks: Sequence[Task], seeds: Sequence[int]) -> dict[str, object]:
    """The totals of every episode's trace summary (`trace.completion_totals`), added up over the run; a token
    count or the cost stays None when no answer reported it."""
    counts = ("calls", "truncated", "retried", "extra_requests")
    reported = ("prompt_tokens", "completion_tokens", "reasoning_tokens", "cached_tokens", "cost")
    keys = counts + reported
    totals: dict[str, object] = {**{k: 0 for k in counts}, **{k: None for k in reported}}
    models: set[str] = set()
    providers: set[str] = set()
    for task in tasks:
        for seed in seeds:
            path = out_dir / "traces" / task.game_id / f"seed{seed}" / "summary.json"
            part = json.loads(path.read_text(encoding="utf-8")).get("completions") if path.exists() else None
            if not part:
                continue
            for k in keys:
                if part.get(k) is not None:
                    totals[k] = (totals[k] or 0) + part[k]
            models.update(part.get("served_models") or [])
            providers.update(part.get("served_providers") or [])
    calls = totals["calls"]
    totals["truncated_share"] = totals["truncated"] / calls if calls else None
    totals["served_models"] = sorted(models)
    totals["served_providers"] = sorted(providers)
    return totals


def settings(plan: RequestPlan, tasks: Sequence[Task], seeds: Sequence[int], concurrency: int) -> dict[str, object]:
    """The settings of the run for results.json: the request as sent and everything else that fixes the episodes."""
    return {"api": "openrouter", "model": plan.model, "request": plan.describe(),
            "max_steps_per_level": MAX_STEPS_PER_LEVEL, "observation_version": OBSERVATION_VERSION,
            "game_catalog": catalog_name(), "seeds": list(seeds), "tasks": [asdict(t) for t in tasks],
            "concurrency": concurrency}


def report_text(plan: RequestPlan, base: str, totals: dict[str, object]) -> str:
    """The text report: a heading naming the request, the levels-beaten table of `evaluate.format_report` (its own
    heading replaced), and the completion totals."""
    heading = (f"Levels beaten (model {plan.model} via OpenRouter; effort {plan.effort}; output limit "
               f"{plan.max_tokens} as {plan.limit_param}; providers {' '.join(plan.providers)}; zero data "
               "retention)")
    body = base.split("\n", 1)[1]
    share = totals["truncated_share"]
    lines = [f"answered calls {totals['calls']}; cut off at the output limit {totals['truncated']}"
             + (f" ({100 * share:.1f}%)" if share is not None else "")
             + f"; answered after a retry {totals['retried']}; cost "
             + (f"${totals['cost']:.2f}" if totals["cost"] is not None else "not reported"),
             f"served models: {' '.join(totals['served_models']) or '-'}; "
             f"served providers: {' '.join(totals['served_providers']) or '-'}"]
    if share is not None and share > TRUNCATION_GATE:
        lines.append(f"WARNING: {100 * share:.1f}% of the reflection calls were cut off at the output limit (more "
                     f"than {100 * TRUNCATION_GATE:.0f}%); raise --max-tokens or lower --effort before the run is "
                     "scored")
    return heading + "\n" + body + "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        tasks, seeds = _prepare(args)
    except (ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    logging.basicConfig(level=logging.INFO, format="%(threadName)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    try:
        plan = plan_request(args.model, effort=args.effort, max_tokens=args.max_tokens, providers=args.providers)
    except PreflightError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 3
    print(json.dumps({"request": plan.describe(), "episodes": len(tasks) * len(seeds)}, indent=2))
    if args.dry_run:
        return 0
    gate = threading.BoundedSemaphore(args.concurrency)
    try:
        OpenRouterPolicy(plan)                # the key check, before any episode starts
    except PreflightError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 3
    episodes = run(tasks, seeds,
                   lambda game_id, seed: JournalPolicy(OpenRouterPolicy(plan, gate=gate),
                                                       journal_path(args.out, game_id, seed)),
                   args.out, resume=args.resume)
    report = summarize(plan.model, tasks, seeds, episodes)
    totals = run_completion_totals(args.out, tasks, seeds)
    data = report.to_json()
    data["settings"] = settings(plan, tasks, seeds, args.concurrency)
    data["completions"] = totals
    text = report_text(plan, format_report(report), totals)
    write_json(args.out / "results.json", data)
    write_utf8(args.out / "results.txt", text)
    print(text, end="")
    return 1 if report.aborted else 0


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m witness_agent.evaluate_api",
        description="Play every (game, seed) episode against a frontier model through OpenRouter (observation "
                    f"version 2, zero data retention) and report the levels beaten. The key is read from {KEY_ENV}.",
        epilog="Exit status: 0 when every episode completed, 1 when some episode aborted, 2 for invalid input, "
               f"3 when the model and provider check failed or {KEY_ENV} is not set.")
    parser.add_argument("--model", required=True, help="OpenRouter model id, e.g. anthropic/claude-opus-5.5")
    parser.add_argument("--tasks", required=True, type=Path, metavar="PATH",
                        help="task rows (JSON Lines; a game outside witness_env needs WITNESS_GAME_CATALOG)")
    parser.add_argument("--out", required=True, type=Path, metavar="DIR",
                        help="output directory (a new one unless --resume)")
    parser.add_argument("--games", nargs="+", metavar="ID", help="play only these game ids of the task rows")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS), metavar="N",
                        help=f"episode seeds (default: {' '.join(map(str, DEFAULT_SEEDS))})")
    parser.add_argument("--effort", default=DEFAULT_EFFORT, metavar="LEVEL",
                        help="reasoning effort sent to OpenRouter (default: %(default)s)")
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS, metavar="N",
                        help="output limit, reasoning included (default: %(default)s)")
    parser.add_argument("--providers", nargs="+", metavar="SLUG",
                        help="provider slugs to admit (default: every one with a zero-data-retention endpoint)")
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY, metavar="N",
                        help="at most this many requests in flight at once (default: %(default)s)")
    parser.add_argument("--resume", action="store_true", help="continue the run in --out")
    parser.add_argument("--dry-run", action="store_true",
                        help="check the model and its providers, print the request and the episode count, and exit "
                             f"without calling the model (needs no {KEY_ENV})")
    return parser.parse_args(argv)


def _prepare(args: argparse.Namespace) -> tuple[tuple[Task, ...], tuple[int, ...]]:
    """The selected tasks (file order) and seeds (given order); ValueError for a bad selection, a repeated seed, a
    non-positive concurrency or output limit, or an output directory that holds a run without --resume."""
    tasks, seeds = select_tasks(load_tasks(args.tasks), args.games, args.seeds, args.tasks)
    if args.concurrency < 1 or args.max_tokens < 1:
        raise ValueError("--concurrency and --max-tokens must be at least 1")
    if not args.dry_run:
        args.out.mkdir(parents=True, exist_ok=True)
        if (args.out / "episodes").exists() and not args.resume:
            raise ValueError(f"{args.out} already holds a run; pass --resume to continue it, or use a new directory")
    return tasks, seeds


if __name__ == "__main__":
    raise SystemExit(main())
