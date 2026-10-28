"""Loaders for the golden files under `tests/golden/`, plus small helpers shared by the test modules.

Every golden file is JSON (`<stem>.json`, or `<stem>.json.gz` when the text is longer than 16 KiB). Each
loader checks the exact key set and the JSON type of every object it reads and raises `ValueError` naming
the file on a missing key, an unknown key or a value of the wrong type. `GoldenStore` re-reads the file on
every call and keeps no state.

Episode names under `agent/episodes/` (`<seed>` is the episode seed):
- `rl_<game>_s<seed>_full`: the recorded model responses of one training rollout, replayed with the training
  wiring; `..._n<k>` keeps only its first k responses. `rl9b_` marks responses of the 9B model, `rl_` those of
  the 27B model.
- `ev_<game>_s<seed>_full`: the responses of `rl_<game>_s<seed>_full` replayed with the evaluation wiring. The tw09
  pair was recorded when tw09 had 5 levels and is replayed with `max_levels` 5, so its prompts are the ones the model
  answered.
- `play_<game>`: scripted responses that play the stored solution of the game's first level in chunks, then a
  CONFIRM that is rejected, then one answer without a plan.
- `scripted_<behavior>`: hand-written responses for the harness behavior in the name (call caps, plan tokens,
  memory operations, ...).
Memory segments under `agent/memory_segments/` are named `m_<game>_s<seed>[_9b]_l<level>`: the events of one
level (counted from 0) of a recorded training rollout of that game and seed.

Helpers:
- `ScriptedPolicy` answers an episode's recorded responses in order and checks every prompt byte for byte.
- `first_diff` describes the first difference between two strings.
- `fallback_moves` gives the first moves of the harness's random fallback for a game and seed.
- `sha16` is the short digest of a frame used by the frame fixtures; `LEVEL_COUNTS`, `MARKER_KEYS` and
  `REQUIRED_MARKER_KEYS` give each game's level count and level-file marker keys.
- `count_paths` enumerates the end-reaching paths of a level through a game's own move and win hooks.
- `encode_truth` turns a board snapshot into the plain-JSON form stored in `env/truth/<game>.json`.

This module uses the standard library only and imports nothing from the packages under test.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import random
import re
import zlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final, Literal

GOLDEN_DIR: Final[Path] = Path(__file__).with_name("golden")

TRAINING_GAMES: Final = ("tw01", "tw02", "tw03", "tw05", "tw06", "tw07", "tw09", "tw10",
                         "ft01", "ft02", "ft03", "ft04", "ft05", "ft06", "ft10", "ft11", "ft12", "ft13",
                         "ft14", "ft15", "ft16", "ft20")
VALIDATION_GAMES: Final = ("tw08", "tw11", "tw12", "tw13", "fv01", "fv02", "fv03", "fv04")
GAME_IDS: Final = TRAINING_GAMES + VALIDATION_GAMES
# The revised training games (`witness_env.TRAIN_R1_IDS`, docs/training.md), with entries in env/solutions_r1.json.
TRAIN_R1_GAMES: Final = ("tw02r1", "tw03r1", "tw10r1", "ft03r1", "ft06r1")
# The training-only games of training set r2 (`witness_env.TRAIN_R2_IDS`), with entries in env/solutions_r2.json.
TRAIN_R2_GAMES: Final = ("rg01", "rg02", "rg03", "rg04", "rg06")
# Games with entries in env/marker_layers.json and env/rule_cases.json (one entry per listed id).
MARKER_CASE_GAMES: Final = ("ft01", "ft02", "ft03", "ft04", "ft05", "ft06", "ft10", "ft11", "ft12", "ft13",
                            "ft14", "ft15", "ft16", "ft20", "fv01", "fv02", "fv03", "fv04")

# Levels in each game's level file.
LEVEL_COUNTS: Final = MappingProxyType({g: 5 if g in VALIDATION_GAMES else 10 for g in GAME_IDS})
# Marker keys each game's level configs may use, and those every level config must contain.
MARKER_KEYS: Final = MappingProxyType({g: frozenset(keys) for g, keys in (
    ("tw01", {"dots"}), ("tw02", {"squares"}), ("tw03", {"tetris"}), ("tw05", {"stars"}), ("tw06", {"triangles"}),
    ("tw07", {"erasers", "squares", "stars", "triangles"}), ("tw08", {"squares", "stars"}), ("tw09", {"dots"}),
    ("tw10", {"squares", "filters"}), ("tw11", {"squares", "stars", "triangles", "tetris"}),
    ("tw12", {"dots", "squares", "stars", "triangles", "tetris"}),
    ("tw13", {"erasers", "squares", "stars", "triangles", "tetris", "dots"}),
    ("ft01", {"censers", "squares"}), ("ft02", {"hearths", "squares"}), ("ft03", {"granaries", "squares"}),
    ("ft04", {"beacon", "squares"}), ("ft05", {"coffers", "squares"}), ("ft06", {"buoys"}),
    ("ft10", {"knots", "beads"}), ("ft11", {"vanes", "rift_vanes", "posts"}), ("ft12", {"gargoyles", "squares"}),
    ("ft13", {"abaci", "counters"}), ("ft14", {"braziers", "squares"}), ("ft15", {"cairns"}),
    ("ft16", {"flints", "straws"}), ("ft20", {"banners"}),
    ("fv01", {"wickets_rank1", "wickets_rank2", "wickets_rank3"}), ("fv02", {"weirs"}), ("fv03", {"estates"}),
    ("fv04", {"wattles"}))})
REQUIRED_MARKER_KEYS: Final = MappingProxyType({
    **MARKER_KEYS, "tw07": frozenset({"erasers"}), "tw11": frozenset(), "tw12": frozenset({"dots"}),
    "tw13": frozenset({"erasers"}), "fv01": frozenset({"wickets_rank1", "wickets_rank2"})})

_HEX16 = re.compile(r"[0-9a-f]{16}")
_MOVES = ((0, -1), (0, 1), (-1, 0), (1, 0))

Step = tuple[int, int, int, str, bool]            # (level, step, action, actor, noop_reset)


# ---------------------------------------------------------------------------------------------------- records
@dataclass(frozen=True)
class ExpectedCall:
    """One answered reflection call of an episode: the prompt pair and where the call happened."""
    system: str
    user: str
    level: int
    level_step: int
    plan: tuple[int, ...] | None


@dataclass(frozen=True)
class ExpectedReward:
    """Reward of one answered call in RL training: total and its three parts."""
    total: float
    step_penalty: float
    level_gain: int
    completion_bonus: float


@dataclass(frozen=True)
class Episode:
    """One complete or partial episode: its inputs (game, seed, limits, responses) and expected outputs.

    `wiring` is "rl" (training: the call budget declines call `min(len(responses), max_reflection_calls) + 1`)
    or "eval" (validation: the harness's own call cap). The RL-only fields are None for "eval".
    """
    name: str
    wiring: Literal["rl", "eval"]
    game_id: str
    seed: int
    max_levels: int
    max_reflection_calls: int
    responses: tuple[str, ...]
    finish_reasons: tuple[str, ...] | None
    calls: tuple[ExpectedCall, ...]
    final_prompt: tuple[str, str] | None
    steps: tuple[Step, ...]
    resets: tuple[tuple[int, int], ...]           # (level, step) of every executed RESET
    levels_completed: int
    stop_reason: str
    rewards: tuple[ExpectedReward, ...] | None
    episode_reward: float | None
    kept: tuple[bool, ...] | None
    representative: int | None


@dataclass(frozen=True)
class PerceptionCall:
    """The board block, the action-semantics text and the observations text of one reflection call."""
    level: int
    level_step: int
    board_state: str
    action_semantics: str
    observations: str


@dataclass(frozen=True)
class PerceptionEpisode:
    """One scripted episode that enters every level of a game, with the prompt texts of every call."""
    game_id: str
    seed: int
    max_levels: int
    max_reflection_calls: int
    responses: tuple[str, ...]
    calls: tuple[PerceptionCall, ...]
    steps: tuple[Step, ...]
    levels_completed: int
    stop_reason: str


@dataclass(frozen=True)
class EnvStep:
    """The engine's answer to one action: frame digests and the state after the action."""
    action: int
    frames: tuple[str, ...]                       # first 16 hex digits of the sha1 of each returned frame
    levels_completed: int
    state: str                                    # GameState name
    level: int                                    # level index after the action
    path: tuple[tuple[int, int], ...]
    start: tuple[int, int]


@dataclass(frozen=True)
class EnvTrace:
    """A scripted action sequence on a fresh game. `solution_free` traces never complete a level."""
    game_id: str
    solution_free: bool
    steps: tuple[EnvStep, ...]


@dataclass(frozen=True)
class MarkerLayerCase:
    """Digest of the marker layer drawn for a small board configuration that is not a shipped level."""
    config: dict
    sha16: str


@dataclass(frozen=True)
class RuleCase:
    """Whether CONFIRM accepts a path on a small board configuration that is not a shipped level."""
    config: dict
    path: tuple[tuple[int, int], ...]
    accepted: bool


@dataclass(frozen=True)
class AdvantageCase:
    """A batch of episodes `(game_id, reward, abort type or None)` and the expected advantage of each."""
    name: str
    n_samples_per_prompt: int
    episodes: tuple[tuple[str, float, str | None], ...]
    expected: tuple[float, ...]


@dataclass(frozen=True)
class RngCase:
    """The first 20 random-fallback draws of the per-episode random source of `(game_id, seed)`."""
    game_id: str
    seed: int
    draws: tuple[int, ...]


# -------------------------------------------------------------------------------------------- schema checking
def _is_kind(value: Any, kind: str) -> bool:
    if kind == "int":
        return type(value) is int
    if kind == "float":
        return type(value) in (int, float)
    if kind == "str":
        return type(value) is str
    if kind == "bool":
        return type(value) is bool
    if kind == "array":
        return type(value) is list
    if kind == "object":
        return type(value) is dict
    raise ValueError(f"unknown JSON kind {kind!r}")


def _type_ok(value: Any, types: str) -> bool:
    return any(value is None if k == "null" else _is_kind(value, k) for k in types.split("|"))


class _Checker:
    """Raises ValueError("<file>: <where>: <problem>") for every schema problem of one file."""

    def __init__(self, file: str) -> None:
        self.file = file

    def fail(self, where: str, problem: str) -> ValueError:
        return ValueError(f"{self.file}: {where}: {problem}")

    def value(self, value: Any, types: str, where: str) -> Any:
        if not _type_ok(value, types):
            raise self.fail(where, f"expected {types}, got {type(value).__name__}")
        return value

    def obj(self, value: Any, keys: Mapping[str, str], where: str) -> dict:
        self.value(value, "object", where)
        for key in keys:
            if key not in value:
                raise self.fail(where, f"missing key {key!r}")
        for key in value:
            if key not in keys:
                raise self.fail(where, f"unknown key {key!r}")
        for key, types in keys.items():
            self.value(value[key], types, f"{where}.{key}")
        return value

    def keyset(self, value: Any, keys: Sequence[str], where: str) -> dict:
        """An object with exactly the keys `keys`, values of any JSON type."""
        return self.obj(value, {k: "int|float|str|bool|array|object|null" for k in keys}, where)

    def tuple_(self, value: Any, kinds: Sequence[str], where: str) -> list:
        self.value(value, "array", where)
        if len(value) != len(kinds):
            raise self.fail(where, f"expected {len(kinds)} elements, got {len(value)}")
        for i, (item, types) in enumerate(zip(value, kinds)):
            self.value(item, types, f"{where}[{i}]")
        return value

    def array(self, value: Any, types: str, where: str) -> list:
        self.value(value, "array", where)
        for i, item in enumerate(value):
            self.value(item, types, f"{where}[{i}]")
        return value

    def node(self, value: Any, where: str) -> tuple[int, int]:
        c, r = self.tuple_(value, ("int", "int"), where)
        return (c, r)

    def nodes(self, value: Any, where: str) -> tuple[tuple[int, int], ...]:
        self.value(value, "array", where)
        return tuple(self.node(n, f"{where}[{i}]") for i, n in enumerate(value))

    def step(self, value: Any, where: str) -> Step:
        level, step, action, actor, noop = self.tuple_(value, ("int", "int", "int", "str", "bool"), where)
        return (level, step, action, actor, noop)

    def steps(self, value: Any, where: str) -> tuple[Step, ...]:
        self.value(value, "array", where)
        return tuple(self.step(s, f"{where}[{i}]") for i, s in enumerate(value))

    def ids(self, value: Any, allowed: Sequence[str], where: str, *, exact: bool) -> dict:
        self.value(value, "object", where)
        unknown = [k for k in value if k not in allowed]
        if unknown:
            raise self.fail(where, f"unknown key {unknown[0]!r}")
        if exact:
            missing = [k for k in allowed if k not in value]
            if missing:
                raise self.fail(where, f"missing key {missing[0]!r}")
        return value


_SEMANTICS_MOVE = MappingProxyType({"action": "int", "status": "str", "direction": "str|null", "times": "int",
                                    "noop_rate": "float|null", "inverse": "int|null"})

_UNIT_SCHEMAS = MappingProxyType({
    "plan_parse": (("response", "recorded_plan"), ("plan", "notices", "cap_note")),
    "kas_lines": (("semantics",), ("text",)),
    "rules_list": (("rules",), ("text", "dissonance")),
    "notices_block": (("notices",), ("block",)),
    "user_message_grid": (("board", "kas", "showcase", "window", "rules", "notices", "attempts"), ("user",)),
    "user_message_example": (("board", "kas", "showcase", "window", "rules", "notices", "attempts"), ("user",)),
    "perception_transition_line": (("action", "pre", "post", "frame_changed", "grid", "random_fallback"),
                                   ("line",)),
    "perception_showcase": (("grid", "transitions", "snapshots"), ("text",)),
    "perception_window": (("entries",), ("text",)),
    "perception_actions_block": (("semantics",), ("actions_block",)),
    "perception_action_tracker": (("ops",), ("semantics", "actions_block")),
})
_MEMORY_KINDS = MappingProxyType({
    "support": (("rule", "n"), ("confidences", "rule_after")),
    "apply": (("rules", "deletes", "adds", "table"), ("notices", "rules_after")),
    "verify": (("rule", "table"), ("result", "rule_after")),
    "outcomes": (("table",), ("outcome_counts", "transitions")),
    "carry_over": (("rules",), ("notice", "rules_after")),
    "trials": (("events",), ("sections",)),
})
_MEMORY_EVENTS = MappingProxyType({
    "step": MappingProxyType({"t": "str", "remaining": "int"}),
    "reflection": MappingProxyType({"t": "str", "deletes": "array", "adds": "array", "section": "str",
                                    "notices": "array", "rules_after": "array", "plan": "array|null",
                                    "cap_note": "str", "start_key": "str|null"}),
    "transition": MappingProxyType({"t": "str", "before": "str", "action": "int", "after": "str"}),
    "rejected_confirm": MappingProxyType({"t": "str", "remaining": "int"}),
    "level_solved": MappingProxyType({"t": "str", "notice": "str|null", "rules_after": "array"}),
})
_TRUTH_KEYS = ("cols", "rows", "path", "start", "starts", "end", "dots", "squares", "stars", "triangles",
               "polyominoes", "erasers", "framed_cells", "breakpoints", "node_markers", "cell_markers")


# -------------------------------------------------------------------------------------------------- the store
class GoldenStore:
    """Read-only access to one golden directory; every method re-reads its file (no caching, no state)."""

    _unit_schemas: Mapping[str, tuple[tuple[str, ...], tuple[str, ...]]] = _UNIT_SCHEMAS

    def __init__(self, root: Path = GOLDEN_DIR) -> None:
        self.root = Path(root)

    # -- file access
    def _read(self, stem: str) -> tuple[_Checker, Any]:
        gz, plain = self.root / f"{stem}.json.gz", self.root / f"{stem}.json"
        if gz.exists() and plain.exists():
            raise ValueError(f"{stem}: both {gz.name} and {plain.name} exist")
        if gz.exists():
            path, text = gz, gzip.decompress(gz.read_bytes()).decode("utf-8")
        elif plain.exists():
            path, text = plain, plain.read_text(encoding="utf-8")
        else:
            raise FileNotFoundError(f"no golden file {stem}.json(.gz) under {self.root}")
        name = path.relative_to(self.root).as_posix()
        try:
            data = json.loads(text)
        except ValueError as exc:
            raise ValueError(f"{name}: not valid JSON: {exc}") from None
        return _Checker(name), data

    def _stems(self, sub: str) -> list[str]:
        folder = self.root / sub
        if not folder.is_dir():
            return []
        stems: list[str] = []
        for p in folder.iterdir():
            if p.name.endswith(".json.gz"):
                stems.append(p.name[:-len(".json.gz")])
            elif p.name.endswith(".json"):
                stems.append(p.name[:-len(".json")])
        dup = sorted({s for s in stems if stems.count(s) > 1})
        if dup:
            raise ValueError(f"{sub}/{dup[0]}: both .json and .json.gz exist")
        return sorted(stems)

    # -- environment
    def grid(self) -> dict:
        """Board geometry, drawing calls and region splits for every board size of the levels."""
        chk, data = self._read("env/grid")
        chk.obj(data, {"sizes": "array", "draws": "array", "regions": "array"}, "top")
        for i, s in enumerate(data["sizes"]):
            chk.obj(s, {"cols": "int", "rows": "int", "cell_size": "int", "offset_x": "int", "offset_y": "int",
                        "nodes": "array", "cells": "array", "grid_sha1": "str"}, f"sizes[{i}]")
        for i, p in enumerate(data["draws"]):
            chk.obj(p, {"cols": "int", "rows": "int", "call": "str", "args": "object", "pixels": "array"},
                    f"draws[{i}]")
        for i, r in enumerate(data["regions"]):
            chk.obj(r, {"cols": "int", "rows": "int", "kind": "str", "seed": "str|null", "path": "array",
                        "edges": "array", "regions": "array", "edge_counts": "array"}, f"regions[{i}]")
        return data

    def env_trace(self, game_id: str) -> EnvTrace:
        """Scripted action sequence on a fresh game, with the engine's answer to every action."""
        chk, data = self._read(f"env/traces/{game_id}")
        chk.obj(data, {"game_id": "str", "solution_free": "bool", "steps": "array"}, "top")
        if data["game_id"] != game_id:
            raise chk.fail("top.game_id", f"{data['game_id']!r} != {game_id!r}")
        steps = []
        for i, s in enumerate(data["steps"]):
            where = f"steps[{i}]"
            action, frames, lc, state, level, path, start = chk.tuple_(
                s, ("int", "array", "int", "str", "int", "array", "array"), where)
            chk.array(frames, "str", f"{where}[1]")
            steps.append(EnvStep(action, tuple(frames), lc, state, level, chk.nodes(path, f"{where}[5]"),
                                 chk.node(start, f"{where}[6]")))
        return EnvTrace(game_id, data["solution_free"], tuple(steps))

    def entry_frames(self) -> dict[str, list[str]]:
        """Digest of the entry frame of every level of every game."""
        chk, data = self._read("env/entry_frames")
        chk.ids(data, GAME_IDS, "top", exact=True)
        for g, digests in data.items():
            chk.array(digests, "str", g)
            bad = [d for d in digests if not _HEX16.fullmatch(d)]
            if bad:
                raise chk.fail(g, f"not a 16-hex digest: {bad[0]!r}")
        return data

    def marker_layers(self) -> dict[str, list[MarkerLayerCase]]:
        """Per game: marker-layer digests of small board configurations that are not shipped levels."""
        chk, data = self._read("env/marker_layers")
        chk.ids(data, MARKER_CASE_GAMES, "top", exact=True)
        out = {}
        for g, cases in data.items():
            chk.value(cases, "array", g)
            out[g] = [MarkerLayerCase(**chk.obj(c, {"config": "object", "sha16": "str"}, f"{g}[{i}]"))
                      for i, c in enumerate(cases)]
        return out

    def path_counts(self) -> dict[str, list[dict]]:
        """Per level: the number of end-reaching paths and how many of them CONFIRM accepts."""
        chk, data = self._read("env/path_counts")
        chk.ids(data, TRAINING_GAMES, "top", exact=False)
        for g, rows in data.items():
            chk.value(rows, "array", g)
            for i, row in enumerate(rows):
                chk.obj(row, {"level": "int", "paths": "int|null", "accepted": "int|null"}, f"{g}[{i}]")
        return data

    def rule_cases(self) -> dict[str, list[RuleCase]]:
        """Per game: whether CONFIRM accepts a path on small board configurations that are not shipped levels."""
        chk, data = self._read("env/rule_cases")
        chk.ids(data, MARKER_CASE_GAMES, "top", exact=True)
        out = {}
        for g, cases in data.items():
            chk.value(cases, "array", g)
            out[g] = []
            for i, c in enumerate(cases):
                chk.obj(c, {"config": "object", "path": "array", "accepted": "bool"}, f"{g}[{i}]")
                out[g].append(RuleCase(c["config"], chk.nodes(c["path"], f"{g}[{i}].path"), c["accepted"]))
        return out

    def truth_levels(self, game_id: str) -> list[dict]:
        """Board snapshot right after entering each level, in the form `encode_truth` returns."""
        chk, data = self._read(f"env/truth/{game_id}")
        chk.obj(data, {"game_id": "str", "levels": "array"}, "top")
        if data["game_id"] != game_id:
            raise chk.fail("top.game_id", f"{data['game_id']!r} != {game_id!r}")
        keys = {k: "int" if k in ("cols", "rows") else "array" for k in _TRUTH_KEYS}
        for i, level in enumerate(data["levels"]):
            chk.obj(level, keys, f"levels[{i}]")
        return data["levels"]

    def solutions(self) -> dict[str, list[list[int]]]:
        """Stored action sequence that solves each level of each training game."""
        chk, data = self._read("env/solutions")
        chk.ids(data, TRAINING_GAMES, "top", exact=True)
        for g, sols in data.items():
            chk.value(sols, "array", g)
            for i, sol in enumerate(sols):
                chk.array(sol, "int", f"{g}[{i}]")
        return data

    def solutions_r1(self) -> dict[str, list[list[int]]]:
        """Stored action sequence that solves each level of each revised training game."""
        chk, data = self._read("env/solutions_r1")
        chk.ids(data, TRAIN_R1_GAMES, "top", exact=True)
        for g, sols in data.items():
            chk.value(sols, "array", g)
            for i, sol in enumerate(sols):
                chk.array(sol, "int", f"{g}[{i}]")
        return data

    def solutions_r2(self) -> dict[str, list[list[int]]]:
        """Stored shortest action sequence that solves each level of each r2 training game."""
        chk, data = self._read("env/solutions_r2")
        chk.ids(data, TRAIN_R2_GAMES, "top", exact=True)
        for g, sols in data.items():
            chk.value(sols, "array", g)
            for i, sol in enumerate(sols):
                chk.array(sol, "int", f"{g}[{i}]")
        return data

    # -- harness
    def perception_games(self) -> list[str]:
        """Sorted game ids that have a perception episode under `agent/perception/`."""
        return self._stems("agent/perception")

    def perception_episode(self, game_id: str) -> PerceptionEpisode:
        """Scripted episode of one game with the board block, action-semantics text and observations text per call."""
        chk, data = self._read(f"agent/perception/{game_id}")
        chk.obj(data, {"game_id": "str", "stop_reason": "str", "seed": "int", "max_levels": "int",
                       "max_reflection_calls": "int", "levels_completed": "int", "responses": "array",
                       "calls": "array", "steps": "array"}, "top")
        if data["game_id"] != game_id:
            raise chk.fail("top.game_id", f"{data['game_id']!r} != {game_id!r}")
        chk.array(data["responses"], "str", "responses")
        calls = tuple(PerceptionCall(**chk.obj(c, {"level": "int", "level_step": "int", "board_state": "str",
                                                   "action_semantics": "str", "observations": "str"},
                                               f"calls[{i}]"))
                      for i, c in enumerate(data["calls"]))
        return PerceptionEpisode(data["game_id"], data["seed"], data["max_levels"], data["max_reflection_calls"],
                                 tuple(data["responses"]), calls, chk.steps(data["steps"], "steps"),
                                 data["levels_completed"], data["stop_reason"])

    def episode_names(self) -> list[str]:
        """Sorted names of the episodes under `agent/episodes/`."""
        return self._stems("agent/episodes")

    def episode(self, name: str) -> Episode:
        """One episode: responses in call order and every expected prompt, step, reset and reward."""
        chk, data = self._read(f"agent/episodes/{name}")
        chk.obj(data, {"name": "str", "wiring": "str", "game_id": "str", "seed": "int", "max_levels": "int",
                       "max_reflection_calls": "int", "responses": "array", "finish_reasons": "array|null",
                       "expected": "object"}, "top")
        if data["name"] != name:
            raise chk.fail("top.name", f"{data['name']!r} != {name!r}")
        wiring = data["wiring"]
        if wiring not in ("rl", "eval"):
            raise chk.fail("top.wiring", f"{wiring!r} is neither 'rl' nor 'eval'")
        responses = tuple(chk.array(data["responses"], "str", "responses"))
        exp = chk.obj(data["expected"], {"calls": "array", "steps": "array", "resets": "array",
                                         "final_prompt": "object|null", "levels_completed": "int",
                                         "stop_reason": "str", "rewards": "array|null",
                                         "episode_reward": "float|null", "kept": "array|null",
                                         "representative": "int|null"}, "expected")
        calls = []
        for i, c in enumerate(exp["calls"]):
            chk.obj(c, {"system": "str", "user": "str", "level": "int", "level_step": "int", "plan": "array|null"},
                    f"expected.calls[{i}]")
            plan = None if c["plan"] is None else tuple(chk.array(c["plan"], "int", f"expected.calls[{i}].plan"))
            calls.append(ExpectedCall(c["system"], c["user"], c["level"], c["level_step"], plan))
        final = exp["final_prompt"]
        if final is not None:
            chk.obj(final, {"system": "str", "user": "str"}, "expected.final_prompt")
            final = (final["system"], final["user"])
        resets = tuple(tuple(chk.tuple_(r, ("int", "int"), f"expected.resets[{i}]"))
                       for i, r in enumerate(exp["resets"]))
        rl_fields = (data["finish_reasons"], exp["rewards"], exp["episode_reward"], exp["kept"])
        finish = rewards = kept = None
        if wiring == "eval":
            if any(v is not None for v in rl_fields) or exp["representative"] is not None:
                raise chk.fail("expected", "an eval episode has RL fields")
        else:
            if any(v is None for v in rl_fields):
                raise chk.fail("expected", "an RL episode lacks finish_reasons, rewards, episode_reward or kept")
            finish = tuple(chk.array(data["finish_reasons"], "str", "finish_reasons"))
            if len(finish) != len(responses) or any(f not in ("stop", "length") for f in finish):
                raise chk.fail("finish_reasons", "one of 'stop' / 'length' per response is required")
            rewards = tuple(ExpectedReward(float(t), float(p), g, float(b)) for t, p, g, b in (
                chk.tuple_(r, ("float", "float", "int", "float"), f"expected.rewards[{i}]")
                for i, r in enumerate(exp["rewards"])))
            kept = tuple(chk.array(exp["kept"], "bool", "expected.kept"))
            if len(rewards) != len(calls) or len(kept) != len(calls):
                raise chk.fail("expected", "rewards and kept need one entry per answered call")
        return Episode(name=name, wiring=wiring, game_id=data["game_id"], seed=data["seed"],
                       max_levels=data["max_levels"], max_reflection_calls=data["max_reflection_calls"],
                       responses=responses, finish_reasons=finish, calls=tuple(calls), final_prompt=final,
                       steps=chk.steps(exp["steps"], "expected.steps"), resets=resets,
                       levels_completed=exp["levels_completed"], stop_reason=exp["stop_reason"], rewards=rewards,
                       episode_reward=None if exp["episode_reward"] is None else float(exp["episode_reward"]),
                       kept=kept, representative=exp["representative"])

    def unit_cases(self, topic: str) -> list[dict]:
        """Cases `{"id", "input", "expected"}` of one topic (memory cases also carry `"kind"`)."""
        chk, data = self._read(f"agent/units/{topic}")
        chk.obj(data, {"topic": "str", "cases": "array"}, "top")
        if data["topic"] != topic:
            raise chk.fail("top.topic", f"{data['topic']!r} != {topic!r}")
        memory = topic == "memory_cases"
        if not memory and topic not in self._unit_schemas:
            raise chk.fail("top.topic", f"unknown topic {topic!r}")
        for i, case in enumerate(data["cases"]):
            where = f"cases[{i}]"
            head = {"id": "str", "input": "object", "expected": "object"}
            if memory:
                head["kind"] = "str"
            chk.obj(case, head, where)
            if memory:
                if case["kind"] not in _MEMORY_KINDS:
                    raise chk.fail(f"{where}.kind", f"unknown kind {case['kind']!r}")
                inputs, expected = _MEMORY_KINDS[case["kind"]]
            else:
                inputs, expected = self._unit_schemas[topic]
            chk.keyset(case["input"], inputs, f"{where}.input")
            chk.keyset(case["expected"], expected, f"{where}.expected")
            for part in ("input", "expected"):
                if "semantics" in case[part]:
                    _check_semantics(chk, case[part]["semantics"], f"{where}.{part}.semantics")
        return data["cases"]

    def memory_segment_names(self) -> list[str]:
        """Sorted names of the memory event segments under `agent/memory_segments/`."""
        return self._stems("agent/memory_segments")

    def memory_segment(self, name: str) -> dict:
        """Memory events of one whole level: rules at entry, then every step, reflection, transition,
        rejected CONFIRM and the level change, in order."""
        chk, data = self._read(f"agent/memory_segments/{name}")
        chk.obj(data, {"name": "str", "game_id": "str", "seed": "int", "level": "int", "seed_rules": "array",
                       "pending_notices": "array", "events": "array"}, "top")
        if data["name"] != name:
            raise chk.fail("top.name", f"{data['name']!r} != {name!r}")
        for i, ev in enumerate(data["events"]):
            chk.value(ev, "object", f"events[{i}]")
            kind = ev.get("t")
            if kind not in _MEMORY_EVENTS:
                raise chk.fail(f"events[{i}].t", f"unknown event {kind!r}")
            chk.obj(ev, _MEMORY_EVENTS[kind], f"events[{i}]")
        return data

    # -- training
    def advantage_cases(self) -> list[AdvantageCase]:
        """Batches of episode rewards with the expected advantage of every episode."""
        chk, data = self._read("rl/advantages")
        chk.obj(data, {"cases": "array"}, "top")
        out = []
        for i, c in enumerate(data["cases"]):
            where = f"cases[{i}]"
            chk.obj(c, {"name": "str", "n": "int", "episodes": "array", "expected": "array"}, where)
            episodes = tuple((g, float(r), a) for g, r, a in (
                chk.tuple_(e, ("str", "float", "str|null"), f"{where}.episodes[{j}]")
                for j, e in enumerate(c["episodes"])))
            expected = tuple(float(x) for x in chk.array(c["expected"], "float", f"{where}.expected"))
            if len(expected) != len(episodes):
                raise chk.fail(where, "one expected advantage per episode is required")
            out.append(AdvantageCase(c["name"], c["n"], episodes, expected))
        return out

    def rng_cases(self) -> list[RngCase]:
        """First random-fallback draws of the per-episode random source for a few `(game_id, seed)` pairs."""
        chk, data = self._read("agent/rng")
        chk.obj(data, {"cases": "array"}, "top")
        out = []
        for i, c in enumerate(data["cases"]):
            chk.obj(c, {"game_id": "str", "seed": "int", "draws": "array"}, f"cases[{i}]")
            out.append(RngCase(c["game_id"], c["seed"], tuple(chk.array(c["draws"], "int", f"cases[{i}].draws"))))
        return out

    def rl_dryrun(self, model: Literal["27b", "9b"]) -> str:
        """Expected stdout of `scripts/train.py --model MODEL --run-name RUN --dry-run` with
        MODEL_DIR=/MODELS, CKPT_DIR=/CKPT and DATA_DIR=/DATA."""
        chk, data = self._read("rl/dryrun")
        chk.obj(data, {"27b": "str", "9b": "str"}, "top")
        if model not in data:
            raise ValueError(f"unknown model {model!r}; expected '27b' or '9b'")
        return data[model]


def _check_semantics(chk: _Checker, value: Any, where: str) -> None:
    chk.obj(value, {"moves": "array", "confirm_presses": "int"}, where)
    if len(value["moves"]) != 4:
        raise chk.fail(f"{where}.moves", f"expected 4 moves, got {len(value['moves'])}")
    for i, move in enumerate(value["moves"]):
        chk.obj(move, _SEMANTICS_MOVE, f"{where}.moves[{i}]")


STORE: Final[GoldenStore] = GoldenStore(GOLDEN_DIR)


# ---------------------------------------------------------------------------------------------- replay policy
class ScriptedPolicy:
    """Policy that answers an episode's responses in order and checks every prompt byte for byte.

    It answers the first `limit` calls, where `limit` is `min(len(responses), max_reflection_calls)` for an
    RL episode (the training call budget declines the next call) and `len(responses)` for an evaluation
    episode (the harness's own call cap stops first). The call after the last answer is compared with the
    episode's final prompt and declined. On the first prompt difference, or on a call the episode does not
    expect, the policy records the difference in `mismatch` and declines, so the episode stops cleanly and
    the test can report the message.
    """

    def __init__(self, episode: Episode) -> None:
        self.episode = episode
        self.calls_made = 0
        self.mismatch: str | None = None
        n = len(episode.responses)
        self._limit = n if episode.wiring == "eval" else min(n, episode.max_reflection_calls)

    def _compare(self, i: int, expected: tuple[str, str], actual: tuple[str, str]) -> bool:
        for field, want, got in (("system", expected[0], actual[0]), ("user", expected[1], actual[1])):
            if want != got:
                self.mismatch = f"call {i} {field}: " + first_diff(want, got)
                return False
        return True

    def __call__(self, system: str, user: str) -> str | None:
        i = self.calls_made
        self.calls_made += 1
        if self.mismatch is not None:
            return None
        ep = self.episode
        if i < self._limit:
            call = ep.calls[i] if i < len(ep.calls) else None
            if call is None:
                self.mismatch = f"unexpected call {i}"
                return None
            if not self._compare(i, (call.system, call.user), (system, user)):
                return None
            return ep.responses[i]
        if i == self._limit and ep.final_prompt is not None:
            self._compare(i, ep.final_prompt, (system, user))
            return None
        self.mismatch = f"unexpected call {i}"
        return None


def first_diff(expected: str, actual: str, *, context: int = 80) -> str:
    """"identical", or the index and 1-based line of the first differing character (the shorter length
    if one string is a prefix of the other) with `context` characters of both strings around it."""
    if expected == actual:
        return "identical"
    n = next((i for i, (x, y) in enumerate(zip(expected, actual)) if x != y), min(len(expected), len(actual)))
    line = expected[:n].count("\n") + 1
    lo, hi = max(0, n - context), n + context
    return (f"char {n} (line {line}): expected {expected[lo:hi]!r}\n"
            f"                    actual   {actual[lo:hi]!r}")


def fallback_moves(game_id: str, seed: int, n: int) -> list[int]:
    """The first `n` moves of the random fallback of an episode of `game_id` with `seed`: a `random.Random` seeded
    with (seed * 1000003) XOR crc32(game id) chooses among ACTION1-ACTION4."""
    rng = random.Random((seed * 1_000_003) ^ zlib.crc32(game_id.encode("utf-8")))
    return [rng.choice((1, 2, 3, 4)) for _ in range(n)]


# ------------------------------------------------------------------------------------------- env test helpers
def sha16(frame: Any) -> str:
    """The first 16 hex digits of the SHA-1 of a frame's cells, one byte per cell (the value modulo 256) in row
    order; `frame` is a sequence of rows of integers, e.g. a list of lists or a 2-D numpy array."""
    return hashlib.sha1(bytes(int(v) & 0xFF for row in frame for v in row)).hexdigest()[:16]


def count_paths(game: Any, board: Any, markers: Any) -> tuple[int, int]:
    """(paths, accepted) over every simple path from any listed start of `board` that has at least two
    nodes and ends at `board.end`.

    A path grows by `game._move_target(board, tip, dc, dr)` in the order up, down, left, right, skipping
    blocked moves and nodes already on the path. A path may pass through the end and continue; every prefix
    that ends at the end is counted, and accepted when `game._wins(board, markers, path)` is true.
    """
    paths = accepted = 0
    end = board.end
    for start in board.starts:
        path = [start]
        on_path = {start}
        next_move = [0]                           # per path node: index of the next move to try
        while next_move:
            k = next_move[-1]
            if k == len(_MOVES):
                next_move.pop()
                on_path.discard(path.pop())
                continue
            next_move[-1] = k + 1
            target = game._move_target(board, path[-1], *_MOVES[k])
            if target is None or target in on_path:
                continue
            path.append(target)
            on_path.add(target)
            next_move.append(0)
            if target == end:
                paths += 1
                if game._wins(board, markers, tuple(path)):
                    accepted += 1
    return paths, accepted


def encode_truth(truth: Any) -> dict:
    """The plain-JSON form of a board snapshot, as stored in `env/truth/<game>.json`.

    Nodes and cells are `[col, row]`. `path`, `starts`, `dots` and the nodes of each node marker keep their
    order; every other collection is sorted. Only the snapshot's public attributes are read.
    """
    def cells(mapping: Mapping) -> list:
        return sorted([c, r, v] for (c, r), v in mapping.items())

    return {
        "cols": truth.cols,
        "rows": truth.rows,
        "path": [[c, r] for c, r in truth.path],
        "start": list(truth.start),
        "starts": [[c, r] for c, r in truth.starts],
        "end": list(truth.end),
        "dots": [[c, r] for c, r in truth.dots],
        "squares": cells(truth.squares),
        "stars": cells(truth.stars),
        "triangles": cells(truth.triangles),
        "polyominoes": sorted([c, r, [list(o) for o in p.shape], p.rotatable]
                              for (c, r), p in truth.polyominoes.items()),
        "erasers": sorted([c, r] for c, r in truth.erasers),
        "framed_cells": sorted([c, r] for c, r in truth.framed_cells),
        "breakpoints": sorted([list(min(a, b)), list(max(a, b))] for a, b in truth.breakpoints),
        "node_markers": [{"name": m.name, "nodes": [[c, r] for c, r in m.nodes]} for m in truth.node_markers],
        "cell_markers": sorted(({"name": m.name, "glyph": m.glyph,
                                 "values": sorted([[c, r, v] for (c, r), v in m.values.items()],
                                                  key=lambda e: (e[0], e[1])),
                                 "values_are_colours": m.values_are_colours} for m in truth.cell_markers),
                               key=lambda d: d["name"]),
    }
