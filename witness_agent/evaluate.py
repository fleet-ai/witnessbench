"""The validation evaluator: levels beaten per (validation game, seed) episode against a policy served over HTTP.

    python -m witness_agent.evaluate --base-url URL --out DIR [--model NAME] [--tasks PATH]
                                     [--games ID [ID ...]] [--seeds N [N ...]] [--no-traces]

The task rows (`data/val.jsonl` of the repository unless `--tasks` is given) fix each game's `max_levels` and
reflection-call cap. Their games are the games of witness_env (the public and the training-only games) and, when the
environment variable WITNESS_GAME_CATALOG names a module, that module's games (`witness_agent.catalog`). The episodes
play the harness's prompt contract, observation version 2 (`run_episode`); the report records it and the catalog
module.

The measure of an episode is the number of levels it beat, at most the row's `max_levels`: levels are completed
strictly in order, so these are the levels solved before the episode ended. An aborted episode is not a
measurement: it is reported with its error and left out of every mean and total. The report gives the levels beaten
per game and seed, their mean and sample standard deviation per game, the per-seed totals with their mean and
sample standard deviation, and the maximum total.

All episodes run at once, one thread each. DIR receives `results.json` and `results.txt` (the report),
`episodes/<game_id>_seed<seed>.json` (one record per episode, with its observation version, written when it ends)
and `traces/<game_id>/seed<seed>/` (every prompt, answer and action; skipped with `--no-traces`).
"""
from __future__ import annotations

import argparse
import json
import logging
import statistics
import sys
import threading
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final

from witness_env import WitnessGame

from .catalog import catalog_name, make_game
from .episode import (OBSERVATION_VERSION, EpisodeAborted, EpisodeLimits, EpisodeResult, Policy, Task, load_tasks,
                      repeated)
from .http_policy import (CHAT_TEMPLATE_KWARGS, MAX_TOKENS, TEMPERATURE, HttpPolicy, PreflightError,
                          probe_thinking_off)
from .loop import MAX_STEPS_PER_LEVEL, run_episode
from .trace import write_trace, write_utf8

# Exists only in a source checkout of the repository.
DEFAULT_TASKS: Final = Path(__file__).resolve().parent.parent / "data" / "val.jsonl"
DEFAULT_SEEDS: Final = (0, 1, 2)
OUTPUT_NAMES: Final = ("results.json", "results.txt", "episodes", "traces")

PolicyFactory = Callable[[str, int], Policy]          # (game_id, seed) -> the policy of that episode
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EpisodeOutcome:
    """The record of one (game, seed) episode."""

    game_id: str
    seed: int
    max_levels: int                    # from the task row
    max_reflection_calls: int          # from the task row
    levels_beaten: int | None          # min(levels_completed, max_levels); None iff the episode aborted
    levels_completed: int              # also for an aborted episode (what it completed before it aborted)
    reflection_calls: int              # answered reflection calls
    steps: int                         # actions taken
    stop_reason: str                   # why the episode stopped ("aborted" for an aborted episode)
    error: str | None                  # "<ExceptionType>: <message>" of the abort cause, else None
    observation_version: int = OBSERVATION_VERSION   # the observation version the episode was played with

    @classmethod
    def from_result(cls, task: Task, result: EpisodeResult, error: str | None) -> EpisodeOutcome:
        """The record of `result`, an episode of `task`; `error` is None for a completed episode and the abort
        cause for an aborted one (whose `result` holds what was recorded before it aborted)."""
        return cls(game_id=result.game_id, seed=result.seed, max_levels=task.max_levels,
                   max_reflection_calls=task.max_reflection_calls,
                   levels_beaten=None if error is not None else min(result.levels_completed, task.max_levels),
                   levels_completed=result.levels_completed, reflection_calls=len(result.calls),
                   steps=len(result.steps), stop_reason=result.stop_reason, error=error,
                   observation_version=result.observation_version)

    def to_json(self) -> dict[str, object]:
        """The fields as a JSON object, in declaration order."""
        return asdict(self)


@dataclass(frozen=True)
class GameSummary:
    """The levels beaten on one game, one entry per seed in report seed order (None = aborted)."""

    game_id: str
    max_levels: int                         # the most levels an episode can beat: min(row max_levels, game levels)
    levels_beaten: tuple[int | None, ...]
    mean: float | None                      # over the non-None entries; None if there are none
    std: float | None                       # sample std (ddof=1) over the non-None entries; None if fewer than 2


@dataclass(frozen=True)
class ValidationReport:
    """The levels-beaten report of one evaluation: every game of `tasks` played with every seed of `seeds`."""

    model: str
    seeds: tuple[int, ...]
    tasks: tuple[Task, ...]
    games: tuple[GameSummary, ...]          # task order
    seed_totals: tuple[int | None, ...]     # per seed: sum over games; None if any episode of that seed aborted
    max_total: int                          # sum of the games' max_levels
    mean_total: float | None                # over the non-None seed totals
    std_total: float | None                 # sample std (ddof=1) over the non-None seed totals; None if fewer than 2
    episodes: tuple[EpisodeOutcome, ...]    # task order, then seed order

    @property
    def aborted(self) -> tuple[EpisodeOutcome, ...]:
        """The aborted episodes, in episode order."""
        return tuple(e for e in self.episodes if e.error is not None)

    def to_json(self) -> dict[str, object]:
        """The report as a JSON object: the settings of the evaluation (model name, sampling constants, per-level
        step budget, observation version, the game catalog module named by WITNESS_GAME_CATALOG or null, seeds, task
        rows), the per-game summaries, the per-seed totals with their maximum, mean and sample standard deviation,
        and every episode record."""
        return {
            "settings": {
                "model": self.model,
                "temperature": TEMPERATURE,
                "max_tokens": MAX_TOKENS,
                "chat_template_kwargs": dict(CHAT_TEMPLATE_KWARGS),
                "max_steps_per_level": MAX_STEPS_PER_LEVEL,
                "observation_version": OBSERVATION_VERSION,
                "game_catalog": catalog_name(),
                "seeds": list(self.seeds),
                "tasks": [asdict(t) for t in self.tasks],
            },
            "games": [{"game_id": g.game_id, "max_levels": g.max_levels, "levels_beaten": list(g.levels_beaten),
                       "mean": g.mean, "std": g.std} for g in self.games],
            "seed_totals": list(self.seed_totals),
            "max_total": self.max_total,
            "mean_total": self.mean_total,
            "std_total": self.std_total,
            "episodes": [e.to_json() for e in self.episodes],
        }


def run_validation(tasks: Sequence[Task], seeds: Sequence[int], policy_for: PolicyFactory, out_dir: Path,
                   *, write_traces: bool = True) -> tuple[EpisodeOutcome, ...]:
    """Play every (task, seed) episode at once (`play_episodes`) and return their records in task order, then seed
    order. Every game and every policy (`policy_for(game_id, seed)`) is made before any episode starts. `out_dir`
    must exist and `out_dir/episodes` must not exist yet."""
    pairs = [(task, seed) for task in tasks for seed in seeds]
    games = [make_game(task.game_id) for task, _ in pairs]
    policies = [policy_for(task.game_id, seed) for task, seed in pairs]
    (out_dir / "episodes").mkdir(exist_ok=False)
    return play_episodes(pairs, games, policies, out_dir, write_traces=write_traces)


def play_episodes(pairs: Sequence[tuple[Task, int]], games: Sequence[WitnessGame], policies: Sequence[Policy],
                  out_dir: Path, *, write_traces: bool) -> tuple[EpisodeOutcome, ...]:
    """Play the episode of every (task, seed) of `pairs`, with the game and the policy of the same index, at once,
    one thread each, and return their records in the order of `pairs`.

    Each episode writes its record to `out_dir/episodes/<game_id>_seed<seed>.json` when it ends and then, if
    `write_traces`, its trace to `out_dir/traces/<game_id>/seed<seed>/`. An episode that aborts is recorded with
    its error and logged with the full traceback; it does not stop the other episodes. A trace that cannot be
    written (an `OSError` or a `ValueError`) is logged with its traceback and changes nothing else. Any other
    exception, e.g. a record that cannot be written, is raised once every episode has ended (with several, the one
    of the first episode in that order). An exception that is not an `Exception` (e.g. KeyboardInterrupt raised by
    a custom policy) ends only its thread, where `threading.excepthook` reports it; the call then raises
    RuntimeError naming that episode.
    """
    outcomes: dict[int, EpisodeOutcome] = {}          # by episode index; each thread writes its own entry
    errors: dict[int, Exception] = {}

    def work(i: int) -> None:
        task, seed = pairs[i]
        try:
            outcomes[i] = _play(task, seed, games[i], policies[i], out_dir, write_traces=write_traces)
        except Exception as exc:
            errors[i] = exc

    threads = [threading.Thread(target=work, args=(i,), name=f"{task.game_id}-seed{seed}", daemon=True)
               for i, (task, seed) in enumerate(pairs)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    for i, (task, seed) in enumerate(pairs):
        if i in errors:
            raise errors[i]
        if i not in outcomes:
            raise RuntimeError(f"the episode thread of {task.game_id} seed {seed} ended without a record; its "
                               "exception was reported by threading.excepthook")
    return tuple(outcomes[i] for i in range(len(pairs)))


def _play(task: Task, seed: int, game: WitnessGame, policy: Policy, out_dir: Path, *,
          write_traces: bool) -> EpisodeOutcome:
    """Play one episode, write its record and then its trace (logging a trace that cannot be written), log its end,
    and return its record."""
    limits = EpisodeLimits(max_levels=task.max_levels, max_reflection_calls=task.max_reflection_calls)
    aborted: EpisodeAborted | None = None
    error: str | None = None
    try:
        result = run_episode(game, seed=seed, limits=limits, policy=policy)
    except EpisodeAborted as exc:
        aborted, result = exc, exc.partial
        error = f"{type(exc.__cause__).__name__}: {exc.__cause__}"
    outcome = EpisodeOutcome.from_result(task, result, error)
    write_json(out_dir / "episodes" / f"{task.game_id}_seed{seed}.json", outcome.to_json())
    if write_traces:
        trace_dir = out_dir / "traces" / task.game_id / f"seed{seed}"
        try:
            write_trace(result, trace_dir)
        except (OSError, ValueError):
            logger.exception("%s seed %d: the trace could not be written to %s", task.game_id, seed, trace_dir)
    if aborted is None:
        logger.info("%s seed %d: %d/%d levels beaten (%d reflection calls, %d steps, stop %s)", task.game_id,
                    seed, outcome.levels_beaten, _level_maximum(task, game), outcome.reflection_calls, outcome.steps,
                    outcome.stop_reason)
    else:
        logger.error("%s seed %d: aborted after %d reflection calls: %s", task.game_id, seed,
                     outcome.reflection_calls, error, exc_info=aborted)
    return outcome


def summarize(model: str, tasks: Sequence[Task], seeds: Sequence[int],
              episodes: Sequence[EpisodeOutcome]) -> ValidationReport:
    """The report over `episodes`, which must hold exactly one episode per (task game id, seed) pair.

    A game's maximum is its row's `max_levels`, or its number of levels if that is smaller (an episode ends when
    the game is won); the maximum total is their sum. Means are `statistics.fmean` and standard deviations
    `statistics.stdev` (sample, ddof=1) over the non-aborted values. Raises ValueError if a game id or a seed
    repeats, or if an episode of a pair is missing, repeated, or not asked for.
    """
    pairs = [(t.game_id, s) for t in tasks for s in seeds]
    wanted = set(pairs)
    if len(wanted) != len(pairs):
        raise ValueError("the tasks repeat a game id or the seeds repeat a seed")
    by_pair: dict[tuple[str, int], EpisodeOutcome] = {}
    for e in episodes:
        if (e.game_id, e.seed) in by_pair:
            raise ValueError(f"more than one episode of {e.game_id} seed {e.seed}")
        by_pair[(e.game_id, e.seed)] = e
    missing = [f"{g} seed {s}" for g, s in pairs if (g, s) not in by_pair]
    unexpected = [f"{g} seed {s}" for g, s in by_pair if (g, s) not in wanted]
    if missing or unexpected:
        raise ValueError(f"episodes do not match the tasks and seeds: missing {missing}, unexpected {unexpected}")
    games = []
    for task in tasks:
        levels = tuple(by_pair[(task.game_id, s)].levels_beaten for s in seeds)
        maximum = _level_maximum(task, make_game(task.game_id))
        games.append(GameSummary(task.game_id, maximum, levels, *_mean_std(levels)))
    seed_totals: list[int | None] = []
    for i in range(len(seeds)):
        column = [g.levels_beaten[i] for g in games]
        present = [v for v in column if v is not None]
        seed_totals.append(sum(present) if len(present) == len(column) else None)
    mean_total, std_total = _mean_std(seed_totals)
    return ValidationReport(model=model, seeds=tuple(seeds), tasks=tuple(tasks), games=tuple(games),
                            seed_totals=tuple(seed_totals), max_total=sum(g.max_levels for g in games),
                            mean_total=mean_total, std_total=std_total, episodes=tuple(by_pair[p] for p in pairs))


def format_report(report: ValidationReport) -> str:
    """The report as a text table: a heading with the sampling settings and a line naming the observation version;
    one row per game with its maximum, the levels beaten per seed ("abort" for an aborted episode), their mean and
    standard deviation ("-" when undefined); a total row with the maximum total and the per-seed totals; then one
    line per aborted episode with the first line of its error. A seed column is 8 characters wide, or wider when its
    label needs more room to keep a space before it."""
    labels = [f"seed {s}" for s in report.seeds]
    widths = [max(8, len(label) + 1) for label in labels]

    def cells(values: Sequence[int | None]) -> str:
        return "".join(_cell(v, w) for v, w in zip(values, widths, strict=True))

    lines = [f"Levels beaten (model {report.model}; temperature {TEMPERATURE}; max_tokens {MAX_TOKENS}; "
             f"thinking off)",
             f"observation version: {OBSERVATION_VERSION}",
             f"{'game':<6}{'max':>5}" + "".join(f"{label:>{w}}" for label, w in zip(labels, widths, strict=True))
             + f"{'mean':>8}{'std':>8}"]
    for g in report.games:
        lines.append(f"{g.game_id:<6}{g.max_levels:>5}" + cells(g.levels_beaten) + _number(g.mean) + _number(g.std))
    lines.append(f"{'total':<6}{report.max_total:>5}" + cells(report.seed_totals)
                 + _number(report.mean_total) + _number(report.std_total))
    for e in report.episodes:
        if e.error is not None:
            first_line = e.error.partition("\n")[0]
            lines.append(f"aborted: {e.game_id} seed {e.seed}: {first_line}")
    return "\n".join(lines) + "\n"


def select_tasks(tasks: Sequence[Task], games: Sequence[str] | None, seeds: Sequence[int],
                 path: Path) -> tuple[tuple[Task, ...], tuple[int, ...]]:
    """The tasks of `games` in task-file order (every task when `games` is None) and `seeds` in the given order.
    Raises ValueError for a game id the tasks file at `path` does not offer or that is given twice (the message
    lists the ids it offers) and for a repeated seed."""
    tasks = tuple(tasks)
    if games is not None:
        offered = [t.game_id for t in tasks]
        unknown = [g for g in games if g not in offered]
        twice = repeated(games)
        if unknown or twice:
            problem = f"not in {path}: {' '.join(unknown)}" if unknown else f"given twice: {' '.join(twice)}"
            raise ValueError(f"--games {problem}; the tasks file offers {' '.join(offered)}")
        tasks = tuple(t for t in tasks if t.game_id in games)
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"--seeds repeats a seed: {' '.join(map(str, seeds))}")
    return tasks, tuple(seeds)


def write_json(path: Path, obj: object) -> None:
    """Write `obj` as indented JSON (`write_utf8`)."""
    write_utf8(path, json.dumps(obj, indent=2, ensure_ascii=False) + "\n")


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command line: check the inputs, check that the server answers without thinking, play every
    selected (game, seed) episode, write `results.json` and `results.txt` into the output directory and print
    the report.

    Returns 0 when every episode completed, 1 when some episode aborted (its error is in the report), 2 for
    invalid inputs (arguments, tasks file, output directory) and 3 when the server check fails; with 2 and 3 no
    episode has started. Any other exception propagates.
    """
    args = _parse_args(argv)
    try:
        tasks, seeds = _prepare(args)
    except (ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    logging.basicConfig(level=logging.INFO, format="%(threadName)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    try:
        tokens = probe_thinking_off(args.base_url, args.model)
    except PreflightError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 3
    logger.info("server check passed: thinking off (%d completion tokens)", tokens)
    episodes = run_validation(tasks, seeds, lambda game_id, seed: HttpPolicy(args.base_url, args.model), args.out,
                              write_traces=not args.no_traces)
    report = summarize(args.model, tasks, seeds, episodes)
    text = format_report(report)
    write_json(args.out / "results.json", report.to_json())
    write_utf8(args.out / "results.txt", text)
    print(text, end="")
    return 1 if report.aborted else 0


def _level_maximum(task: Task, game: WitnessGame) -> int:
    """The most levels an episode of `task` can beat: the row's `max_levels`, or the number of levels of `game` if
    that is smaller (an episode ends when its game is won)."""
    return min(task.max_levels, game.win_score)


def _mean_std(values: Sequence[int | None]) -> tuple[float | None, float | None]:
    """Mean and sample standard deviation of the non-None values (None when there are no / fewer than 2)."""
    present = [v for v in values if v is not None]
    return (statistics.fmean(present) if present else None,
            statistics.stdev(present) if len(present) >= 2 else None)


def _cell(value: int | None, width: int) -> str:
    return f"{value:>{width}}" if value is not None else f"{'abort':>{width}}"


def _number(value: float | None) -> str:
    return f"{value:>8.2f}" if value is not None else f"{'-':>8}"


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    """The command line (argparse exits with status 2 on a usage error)."""
    parser = argparse.ArgumentParser(
        prog="python -m witness_agent.evaluate",
        description="Play every (validation game, seed) episode against an OpenAI-compatible server and report "
                    "the levels beaten.",
        epilog="Exit status: 0 when every episode completed, 1 when some episode aborted, 2 for invalid input, "
               "3 when the server check failed.")
    parser.add_argument("--base-url", required=True, metavar="URL",
                        help="OpenAI-compatible base URL, e.g. http://127.0.0.1:30000/v1")
    parser.add_argument("--out", required=True, type=Path, metavar="DIR",
                        help="output directory (created if missing; must not hold outputs of an earlier run)")
    parser.add_argument("--model", default="witness-policy", metavar="NAME",
                        help="served model name (default: %(default)s)")
    parser.add_argument("--tasks", type=Path, metavar="PATH",
                        help="task rows (default: data/val.jsonl of the repository, present in a source checkout)")
    parser.add_argument("--games", nargs="+", metavar="ID", help="play only these game ids of the task rows")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS), metavar="N",
                        help=f"episode seeds (default: {' '.join(map(str, DEFAULT_SEEDS))})")
    parser.add_argument("--no-traces", action="store_true", help="do not write the episode traces")
    return parser.parse_args(argv)


def _prepare(args: argparse.Namespace) -> tuple[tuple[Task, ...], tuple[int, ...]]:
    """Validate the parsed command line, create `args.out`, and return the selected tasks (file order) and seeds
    (given order).

    Raises ValueError for a missing default task file, a malformed tasks file, a `--games` id the tasks file
    does not offer or gives twice (the message lists the ids it offers), a repeated seed, and an output directory
    that already holds one of `OUTPUT_NAMES`; OSError for an unreadable tasks file or an output path that cannot be a
    directory.
    """
    path = args.tasks
    if path is None:
        path = DEFAULT_TASKS
        if not path.is_file():
            raise ValueError(f"the default task file {DEFAULT_TASKS} does not exist (it is part of a source "
                             "checkout); pass --tasks PATH")
    tasks, seeds = select_tasks(load_tasks(path), args.games, args.seeds, path)
    args.out.mkdir(parents=True, exist_ok=True)
    existing = [name for name in OUTPUT_NAMES if (args.out / name).exists()]
    if existing:
        raise ValueError(f"{args.out} already holds evaluator outputs ({' '.join(existing)}); "
                         "use a new output directory")
    return tasks, seeds


if __name__ == "__main__":
    raise SystemExit(main())
