"""Episode definitions of the harness: task rows, limits, the policy interface and the records of one episode.

A `Task` is one row of a task file (`data/*.jsonl`). An episode plays one game with one policy under
`EpisodeLimits` and returns an `EpisodeResult` holding every reflection call (`CallRecord`), every action taken
(`StepRecord`) and every RESET the engine executed as a level reset (`ResetRecord`).
"""
from __future__ import annotations

import json
from collections import Counter
from collections.abc import Collection, Iterable, Mapping
from dataclasses import InitVar, dataclass, field
from pathlib import Path
from typing import Final, Literal, Protocol


RESET: Final = 0        # engine RESET; admitted in plans; never chosen by the random fallback
CONFIRM: Final = 5      # the CONFIRM action; the only action that can complete a level
# The prompt contract the harness plays, recorded with every episode: observation version 2 (an earlier version 1 is
# no longer supported).
OBSERVATION_VERSION: Final = 2

Actor = Literal["plan", "random"]
StopReason = Literal["max_levels", "game_won", "level_budget", "call_budget", "policy_stop"]

_ROW_KEYS: Final = ("prompt", "metadata")
_METADATA_KEYS: Final = ("game_id", "max_levels", "max_reflection_calls")


@dataclass(frozen=True)
class Completion:
    """One answer of a policy that reports how it was produced (the OpenRouter policy of `witness_agent.api_policy`).

    `text` is the answer text only, never reasoning text. `finish_reason` is the reason the provider reported:
    "length" means the reply hit the output limit (`truncated`). `usage` holds prompt_tokens, completion_tokens,
    reasoning_tokens, cached_tokens and cost (US dollars), each None when not reported. `attempts` counts the requests
    made for this answer (1 = first try) and `seconds` the wall clock they took, pauses included.
    """

    text: str
    finish_reason: str | None
    usage: Mapping[str, int | float | None]
    served_model: str | None
    served_provider: str | None
    attempts: int
    seconds: float

    @property
    def truncated(self) -> bool:
        return self.finish_reason == "length"

    def meta(self) -> dict[str, object]:
        """Everything but the text, as recorded with the call (`CallRecord.meta`)."""
        return {"finish_reason": self.finish_reason, "truncated": self.truncated, "usage": dict(self.usage),
                "served_model": self.served_model, "served_provider": self.served_provider,
                "attempts": self.attempts, "seconds": self.seconds}


class Policy(Protocol):
    """One stateless, single-turn completion: (system prompt, user message) -> text.

    Returns the completion verbatim ("" is a valid, planless completion), a `Completion` (its text, with how it was
    produced), or None to decline: the episode then ends at once with stop reason "policy_stop" and no further
    engine action. A transport retries its own failures and raises when it gives up; the exception aborts the
    episode (EpisodeAborted).
    """

    def __call__(self, system: str, user: str) -> str | Completion | None: ...


@dataclass(frozen=True)
class Task:
    """One task row: which game to play and the episode limits of that row.

    Construction raises ValueError unless `game_id` is one of `witness_agent.catalog.game_ids()` (the public games,
    plus the games of the catalog module named by WITNESS_GAME_CATALOG; the message of an unknown id says when the
    variable is unset) and both limits are integers >= 1 (booleans are rejected). A harness that plays its games in
    another process passes the ids of the games that process serves as `known_game_ids` instead; the row is then
    checked against them without importing any game code.
    """

    game_id: str
    max_levels: int                 # the episode stops right after this many levels are completed
    max_reflection_calls: int       # reflection calls allowed in one episode of this row

    known_game_ids: InitVar[Collection[str] | None] = field(default=None, kw_only=True)

    def __post_init__(self, known_game_ids) -> None:
        # Remote hosts supply their admitted catalog; never import games to validate it.
        if known_game_ids is None:
            from .catalog import CATALOG_ENV, catalog_name, game_ids
            known_game_ids = game_ids()
            suffix = "" if catalog_name() is not None else f" ({CATALOG_ENV} is unset: public games only)"
        else:
            if (isinstance(known_game_ids, (str, bytes)) or not isinstance(known_game_ids, Collection)
                    or any(not isinstance(g, str) or not g for g in known_game_ids)):
                raise ValueError("known_game_ids must be a collection of game ids")
            suffix = " (explicit game catalog)"
        if self.game_id not in known_game_ids:
            raise ValueError(f"game_id must be a known game id, got {self.game_id!r}{suffix}")
        _require_count("max_levels", self.max_levels)
        _require_count("max_reflection_calls", self.max_reflection_calls)

    @classmethod
    def from_row(cls, row: Mapping, *, known_game_ids: Collection[str] | None = None) -> Task:
        """The task of one parsed line of a task file.

        The row is an object with exactly the keys "prompt" and "metadata"; "metadata" is read by
        `from_metadata`, and "prompt" must equal its game id (it is the trainer's input key and is not read
        otherwise). Raises ValueError quoting the row if any of this fails.
        """
        try:
            _require_keys(row, _ROW_KEYS, "")
            task = cls.from_metadata(row["metadata"], known_game_ids=known_game_ids)
            if row["prompt"] != task.game_id:
                raise ValueError(f"prompt {row['prompt']!r} differs from game_id {task.game_id!r}")
        except ValueError as exc:
            raise ValueError(f"malformed task row {row!r}: {exc}") from None
        return task

    @classmethod
    def from_metadata(cls, md: Mapping, *, known_game_ids: Collection[str] | None = None) -> Task:
        """The task of a row's "metadata" object, which has exactly the keys "game_id", "max_levels" and
        "max_reflection_calls". Raises ValueError listing missing and unexpected keys, or naming a bad value."""
        _require_keys(md, _METADATA_KEYS, "metadata: ")
        return cls(**md, known_game_ids=known_game_ids)


def load_tasks(path: Path, *, known_game_ids: Collection[str] | None = None) -> tuple[Task, ...]:
    """The task rows of the JSON Lines file `path`, in file order; blank lines are skipped (as miles' reader skips
    them).

    Raises ValueError naming the file and the 1-based line of a row that is not JSON or not a valid task row,
    and for a file without rows or with a game id on more than one row; OSError if the file cannot be read.
    """
    tasks = []
    for number, line in enumerate(path.read_text(encoding="utf-8").split("\n"), start=1):
        if not line.strip():
            continue
        try:
            tasks.append(Task.from_row(json.loads(line), known_game_ids=known_game_ids))
        except ValueError as err:
            raise ValueError(f"{path}:{number}: {err}") from None
    if not tasks:
        raise ValueError(f"{path}: no task rows")
    twice = repeated(t.game_id for t in tasks)
    if twice:
        raise ValueError(f"{path}: game ids on more than one row: {' '.join(twice)}")
    return tuple(tasks)


def repeated(items: Iterable[str]) -> list[str]:
    """The items that occur more than once, in order of first appearance."""
    return [item for item, count in Counter(items).items() if count > 1]


@dataclass(frozen=True)
class EpisodeLimits:
    """The limits of one episode. Construction raises ValueError unless `max_levels` is an integer >= 1 and
    `max_reflection_calls` is None or an integer >= 1 (booleans are rejected).

    A reply cut off at the output limit (a `Completion` with finish reason "length") takes no action, so a policy
    that can return such replies needs `max_reflection_calls`: without a cap, an episode whose replies are all cut
    off never ends."""

    max_levels: int                           # the episode stops right after this many levels are completed
    max_reflection_calls: int | None = None   # evaluation cap: after this many answered reflection calls, the last
                                              # call's plan runs to its end; None = no cap

    def __post_init__(self) -> None:
        _require_count("max_levels", self.max_levels)
        if self.max_reflection_calls is not None:
            _require_count("max_reflection_calls", self.max_reflection_calls)


@dataclass(frozen=True)
class CallRecord:
    """One answered reflection call."""

    index: int                                # 0-based call number in the episode
    level: int                                # 0-based level index = levels completed when the call was made
    level_step: int                           # actions already taken on this level (no-op RESETs included)
    system: str
    user: str
    response: str                             # verbatim policy text
    plan: tuple[int, ...] | None              # parsed plan after the 16-action cap; None if planless or cut off
                                              # at the output limit (meta["truncated"]: nothing of it was applied)
    meta: Mapping[str, object] | None = None  # `Completion.meta()` when the policy returned a Completion


@dataclass(frozen=True)
class StepRecord:
    """One action taken by the agent."""

    level: int
    step: int                                 # 0-based index of this action within its level
    action: int
    actor: Actor
    noop_reset: bool                          # a RESET that was not sent to the engine


@dataclass(frozen=True)
class ResetRecord:
    """A RESET the agent chose that the engine executed as a level reset (no-op RESETs are not recorded)."""

    level: int
    step: int                                 # level step count after the RESET (= its StepRecord.step + 1)


@dataclass(frozen=True)
class EpisodeResult:
    """Everything one episode recorded, in order, why it stopped ("aborted" only inside EpisodeAborted) and the
    observation version it was played with."""

    game_id: str
    seed: int
    max_levels: int
    calls: tuple[CallRecord, ...]
    steps: tuple[StepRecord, ...]
    resets: tuple[ResetRecord, ...]
    levels_completed: int
    stop_reason: StopReason | Literal["aborted"]
    observation_version: int = OBSERVATION_VERSION


class EpisodeAborted(Exception):
    """An episode raised: `partial` holds everything recorded so far; the cause is chained (__cause__)."""

    def __init__(self, partial: EpisodeResult) -> None:
        super().__init__(f"episode aborted: game {partial.game_id} seed {partial.seed} "
                         f"after {len(partial.calls)} reflection calls")
        self.partial = partial


def _require_count(name: str, value: object) -> None:
    """Raise ValueError naming `name` and `value` unless `value` is an int (not a bool) >= 1."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be an integer >= 1, got {value!r}")


def _require_keys(obj: object, keys: tuple[str, ...], prefix: str) -> None:
    """Raise ValueError, with a message starting with `prefix`, unless `obj` is a mapping whose key set is exactly
    `keys`; the message lists the missing and the unexpected keys."""
    if not isinstance(obj, Mapping):
        raise ValueError(f"{prefix}expected an object with the keys {list(keys)}, got {type(obj).__name__}")
    missing = [k for k in keys if k not in obj]
    unexpected = sorted((k for k in obj if k not in keys), key=repr)
    problems = []
    if missing:
        problems.append(f"missing keys {missing}")
    if unexpected:
        problems.append(f"unexpected keys {unexpected}")
    if problems:
        raise ValueError(prefix + "; ".join(problems))
