"""Shared pieces of the prompt tests (`test_agent_obs_v2.py` and `test_agent_obs_v2_golden.py`).

- Test-local games built on tw01 that use the optional snapshot fields of the game base: `RingGame` (a CONFIRM
  rejected at the end rings cells), `LampGame` (two lamps whose look changes once visited and that reveal a cell
  marker) and `StripGame` (a legend drawn outside the board).
- `single_level`: a copy of a game that plays one of its levels only.
- `FuzzPolicy`: a policy whose answers do not depend on the prompts (apart from the level number, read only to
  replay a stored solution): a directed prelude (`prelude`) that walks a level through the branches of the step
  texts, then random answers drawn from a source seeded by a text key.
- `EpisodeSpec`, `episode_specs`, `play_spec` and `digest`: the recorded episodes and the digests the recording
  keeps of each; `played` caches their runs for the session.
- Checks used by the prompt tests: `check_nothing_changed` (over an episode, its
  snapshots and frame keys, `frame_key`) and `check_keys_and_texts` (over the states of a random walk, `walk_level`).
- `Scripted`, `play_answers` and `lines`: a policy that gives scripted answers and then declines, the episode it
  plays with the run kept for inspection, and the lines of a prompt that start with given prefixes.

The board internals of a game (`_board`, `_markers`, `_move_target`, `_wins`) are read only on probe copies, by
`prelude` and the helpers it uses (`rejected_route`, `start_switching_move`, `free_move`, `reveal_nodes`, the first
and last also called by some tests); the episodes themselves drive the games through the harness.
"""
from __future__ import annotations

import hashlib
import json
import random
import re
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from functools import cache

import numpy as np
from arcengine import ActionInput, GameAction
from numpy.typing import ArrayLike

from golden import STORE
from witness_agent import EpisodeLimits, EpisodeResult, run_episode
from witness_agent.actions import ActionTracker
from witness_agent.board import board_block, budget_line
from witness_agent.loop import MAX_STEPS_PER_LEVEL, _EpisodeRun
from witness_agent.view import board_view, state_key
from witness_env import GAME_IDS, CellMarkers, LegendEntry, RevealNodes, WitnessGame, make_game
from witness_env.base import load_level_configs
from witness_env.games.tw01 import Tw01
from witness_env.grid import LIGHT_GRAY, WHITE

MOVES = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}          # action -> (dc, dr) of the engine
_LEVEL_RE = re.compile(r"^Level: (\d+)/", re.MULTILINE)
SOLUTIONS = STORE.solutions()


# ------------------------------------------------------------------------------------------- test-local games
class RingGame(Tw01):
    """tw01 whose CONFIRM rejected with the tip on the end node rings the cells whose top-left node the path
    covers (at most three: the smallest in (col, row) order)."""

    GAME_ID = "tw01"

    def _flagged_cells(self, board, markers, path, seen):
        return frozenset(sorted({(c, r) for c, r in path if c < board.cols and r < board.rows})[:3])


def lamp_nodes(cols: int, rows: int, taken) -> tuple[tuple[int, int], ...]:
    """The lamps of a `LampGame` level, in the order the snapshot lists them (not (row, col) order): the first two
    of the nodes (0, rows), (cols, 0), (0, 0), (cols, rows), (1, rows), (cols, 1) ((col, row)) that are not in
    `taken` (the starts, the end and the dots)."""
    candidates = ((0, rows), (cols, 0), (0, 0), (cols, rows), (1, rows), (cols, 1))
    return tuple(n for n in dict.fromkeys(candidates) if n not in taken)[:2]


def lamp_cell(node: tuple[int, int], cols: int, rows: int) -> tuple[int, int]:
    """The cell a lamp reveals: the cell whose corner the lamp node is, nearest the top-left."""
    return (min(node[0], cols - 1), min(node[1], rows - 1))


class LampGame(Tw01):
    """tw01 with the lamps of `lamp_nodes`: each drawn as one light-gray pixel until the visit record holds it and
    white afterwards. The snapshot lists them as the reveal nodes "lamps" (in `lamp_nodes` order) and, once some
    lamp is lit, the cells the lit lamps reveal (`lamp_cell`) as the cell marker "blue_dot" (glyph 'b'). A CONFIRM
    rejected at the end rings the revealed cells."""

    GAME_ID = "tw01"

    @staticmethod
    def _revealed(cols, rows, taken, seen):
        lamps = lamp_nodes(cols, rows, set(taken))
        return lamps, sorted({lamp_cell(n, cols, rows) for n in lamps if n in seen})

    def _draw_reveals(self, frame, board, markers, seen):
        lamps, _ = self._revealed(board.cols, board.rows, (*board.starts, board.end, *markers.dots), seen)
        for lamp in lamps:
            x, y = board.grid.node_xy(lamp)
            frame[y, x] = WHITE if lamp in seen else LIGHT_GRAY

    def _truth_reveals(self, truth, markers, seen):
        lamps, cells = self._revealed(truth.cols, truth.rows, (*truth.starts, truth.end, *truth.dots), seen)
        shown = (CellMarkers("blue_dot", "b", {c: None for c in cells}),) if cells else ()
        return replace(truth, reveal_nodes=(RevealNodes("lamps", {n: n in seen for n in lamps}),),
                       revealed_markers=shown)

    def _flagged_cells(self, board, markers, path, seen):
        return frozenset(self._revealed(board.cols, board.rows, (*board.starts, board.end, *markers.dots),
                                        seen)[1])


class StripGame(Tw01):
    """tw01 whose snapshot also holds the cell marker "blue_dot" (glyph 'b', a color value) on the cell (0, 0)
    and two legend entries for it, listed in the reverse of (marker, colour, icon) order."""

    GAME_ID = "tw01"

    @staticmethod
    def _truth_markers(truth, markers):
        truth = Tw01._truth_markers(truth, markers)
        return replace(truth, cell_markers=(CellMarkers("blue_dot", "b", {(0, 0): 9}, values_are_colours=True),),
                       legend=(LegendEntry("blue_dot", 9, "small_square"), LegendEntry("blue_dot", 6, "hollow_ring")))


MADE_GAMES: dict[str, type[WitnessGame]] = {"rings": RingGame, "lamps": LampGame, "strip": StripGame}


# ------------------------------------------------------------------------------------------------ single levels
@cache
def game_class(game_id: str) -> type[WitnessGame]:
    return type(make_game(game_id))


def level_count(cls: type[WitnessGame]) -> int:
    return len(load_level_configs(cls.GAME_ID, cls.LEVELS_PACKAGE))


def single_level(cls: type[WitnessGame], index: int) -> WitnessGame:
    """A new game of class `cls` that plays only its level `index` (0-based) of its level file."""
    config = load_level_configs(cls.GAME_ID, cls.LEVELS_PACKAGE)[index]
    one = type(f"{cls.__name__}Level{index + 1}", (cls,), {"_level_configs": lambda self: (config,)})
    return one()


# ------------------------------------------------------------------------------------------------- the policy
def plan_text(actions: Sequence[int]) -> str:
    return "<meta>x</meta><plan>" + ",".join(map(str, actions)) + "</plan>"


def chunks(actions: Sequence[int], size: int = 16) -> list[list[int]]:
    return [list(actions[i:i + size]) for i in range(0, len(actions), size)]


def random_answer(rng: random.Random) -> str:
    """15% an empty answer; otherwise a plan of 1-20 random moves, of which 5% hold a RESET, 10% end with a
    CONFIRM and 5% hold an ACTION7; 10% of the plans come with a rule deletion and a rule addition."""
    if rng.random() < 0.15:
        return ""
    moves = [rng.choice((1, 2, 3, 4)) for _ in range(rng.randint(1, 20))]
    u = rng.random()
    if u < 0.05:
        moves[rng.randrange(len(moves))] = 0
    elif u < 0.15:
        moves.append(5)
    elif u < 0.20:
        moves[rng.randrange(len(moves))] = 7
    text = plan_text(moves)
    if rng.random() < 0.10:
        text = f"<delete>{rng.randint(1, 3)}</delete><add>rule {rng.randint(0, 99)} holds</add>" + text
    return text


class FuzzPolicy:
    """Answers `prelude` in order, then random answers (`random_answer`) from a source seeded by `key`. With
    `solutions` (stored solutions by level), call number 6, 12, ... instead starts the stored solution of the level
    its prompt shows ("Level: K/M"), after a RESET, in plans of at most 16 actions on consecutive calls."""

    def __init__(self, key: str, prelude: Sequence[str] = (), solutions: Sequence[Sequence[int]] | None = None):
        self.rng = random.Random(key)
        self.queue: deque[str] = deque(prelude)
        self.solutions = solutions
        self.calls = 0
        self.prompts: list[tuple[str, str]] = []

    def __call__(self, system: str, user: str) -> str:
        self.calls += 1
        self.prompts.append((system, user))
        if self.queue:
            return self.queue.popleft()
        if self.solutions and self.calls % 6 == 0:
            k = int(_LEVEL_RE.search(user).group(1)) - 1
            if k < len(self.solutions):
                first, *rest = chunks([0, *self.solutions[k]])
                self.queue.extend(plan_text(c) for c in rest)
                return plan_text(first)
        return random_answer(self.rng)


def _step(game: WitnessGame, action: int):
    return game.perform_action(ActionInput(id=GameAction.from_id(action)))


def rejected_route(factory: Callable[[], WitnessGame], through: Sequence[tuple[int, int]] = (),
                   limit: int = 50_000) -> list[int] | None:
    """The moves of a shortest simple path from the first start of the game's first level to its end, through every
    node of `through` ((col, row)), that a CONFIRM rejects (checked by playing it, with the CONFIRM, on a fresh
    copy), or None if none is found among the first `limit` paths."""
    probe = factory()
    _step(probe, 0)
    board, markers = probe._board, probe._markers
    start = board.starts[0]
    through = frozenset(through)
    frontier: deque[tuple[tuple[tuple[int, int], ...], list[int]]] = deque([((start,), [])])
    explored = 0
    while frontier and explored < limit:
        path, acts = frontier.popleft()
        explored += 1
        for action, (dc, dr) in MOVES.items():
            target = probe._move_target(board, path[-1], dc, dr)
            if target is None or target in path:
                continue
            new = path + (target,)
            if target == board.end and through <= set(new) and not probe._wins(board, markers, new):
                check = factory()
                _step(check, 0)
                for a in acts + [action, 5]:
                    fd = _step(check, a)
                if fd.levels_completed == 0 and check.truth().path == (start,):
                    return acts + [action]
            frontier.append((new, acts + [action]))
    return None


def start_switching_move(factory: Callable[[], WitnessGame]) -> int | None:
    """On a first level with several starts: a move blocked at the first start that another start can make (the
    engine then switches the start), else None."""
    probe = factory()
    _step(probe, 0)
    board = probe._board
    if len(board.starts) < 2:
        return None
    for action, (dc, dr) in MOVES.items():
        if probe._move_target(board, board.starts[0], dc, dr) is None and any(
                probe._move_target(board, s, dc, dr) is not None for s in board.starts[1:]):
            return action
    return None


def free_move(factory: Callable[[], WitnessGame]) -> int:
    """A move the first start of the first level can make (ACTION1 if none can)."""
    probe = factory()
    _step(probe, 0)
    board = probe._board
    return next((a for a, (dc, dr) in MOVES.items()
                 if probe._move_target(board, board.starts[0], dc, dr) is not None), 1)


def reveal_nodes(factory: Callable[[], WitnessGame]) -> list[tuple[int, int]]:
    """The reveal nodes ((col, row)) of the game's first level, sorted."""
    probe = factory()
    _step(probe, 0)
    return sorted(n for r in probe.truth().reveal_nodes for n in r.lit)


def prelude(factory: Callable[[], WitnessGame], key: str) -> list[str]:
    """Answers that walk the first level of the game through the branches of the step texts, in order: a rejected
    route to the end (through every reveal node if there is such a route) with CONFIRM and then ACTION7 (in plans
    of at most 16 actions); a move, a RESET the engine executes and a RESET it does not get; a CONFIRM at the start,
    twice (two identical attempts); on a level with several starts a blocked move that switches the start and a
    CONFIRM that reverts it; a 20-action plan (cut by the plan cap); an empty answer (the random fallback); an answer
    with a rule deletion and addition."""
    rng = random.Random("prelude:" + key)
    answers = []
    lamps = reveal_nodes(factory)
    route = (lamps and rejected_route(factory, lamps)) or rejected_route(factory)
    if route is not None:
        answers += [plan_text(c) for c in chunks([*route, 5, 7])]
    answers += [plan_text([free_move(factory), 0, 0]), plan_text([5]), plan_text([5])]
    switch = start_switching_move(factory)
    if switch is not None:
        answers.append(plan_text([switch, 5]))
    answers += [plan_text([rng.choice((1, 2, 3, 4)) for _ in range(20)]), "",
                "<delete>1</delete><add>moving extends the trail</add>" + plan_text([rng.choice((1, 2, 3, 4))])]
    return answers


# ------------------------------------------------------------------------------------------ recorded episodes
@dataclass(frozen=True)
class EpisodeSpec:
    """One recorded episode: `kind` "multi" (the whole game, seed `seed`, at most `cap` calls; stored solutions
    replayed on the training games), "level" (level `level` of the game alone, prelude first) or "made" (a
    test-local game, prelude first)."""

    name: str
    kind: str
    game_id: str            # a public game id, or a key of MADE_GAMES for kind "made"
    level: int | None
    seed: int
    cap: int

    def factory(self) -> Callable[[], WitnessGame]:
        if self.kind == "multi":
            return lambda: make_game(self.game_id)
        if self.kind == "level":
            return lambda: single_level(game_class(self.game_id), self.level)
        return MADE_GAMES[self.game_id]

    def policy(self) -> FuzzPolicy:
        if self.kind == "multi":
            return FuzzPolicy(self.name, solutions=SOLUTIONS.get(self.game_id))
        return FuzzPolicy(self.name, prelude(self.factory(), self.name))


def episode_specs() -> list[EpisodeSpec]:
    specs = [EpisodeSpec(f"multi_{g}_s{s}", "multi", g, None, s, 40) for g in GAME_IDS for s in (0, 1, 2)]
    specs += [EpisodeSpec(f"level_{g}_l{i + 1}", "level", g, i, 0, 20)
              for g in GAME_IDS for i in range(level_count(game_class(g)))]
    specs += [EpisodeSpec(f"made_{k}_s{s}", "made", k, None, s, 20) for k in MADE_GAMES for s in (0, 1, 2)]
    return specs


def play(game: WitnessGame, policy: FuzzPolicy, *, seed: int, cap: int) -> tuple[EpisodeResult, list, list[str]]:
    """Play `game` (all its levels, at most `cap` calls); returns the result, the snapshot the harness read after its
    first RESET and after every executed step, and the frame key of the last frame of each of those engine answers
    (in the same order)."""
    snapshots, frames = [], []
    original_truth, original_action = game.truth, game.perform_action

    def truth():
        t = original_truth()
        snapshots.append(t)
        return t

    def perform_action(action_input):
        fd = original_action(action_input)
        frames.append(frame_key(fd.frame[-1]))
        return fd

    game.truth, game.perform_action = truth, perform_action
    result = run_episode(game, seed=seed, limits=EpisodeLimits(game.win_score, cap), policy=policy)
    return result, snapshots, frames


def play_spec(spec: EpisodeSpec) -> tuple[EpisodeResult, FuzzPolicy, list, list[str]]:
    """Play `spec`; returns the result, the policy (with every prompt it was asked), the snapshots and the frame keys
    (`play`)."""
    policy = spec.policy()
    result, snapshots, frames = play(spec.factory()(), policy, seed=spec.seed, cap=spec.cap)
    return result, policy, snapshots, frames


@cache
def specs_by_name() -> dict[str, EpisodeSpec]:
    return {spec.name: spec for spec in episode_specs()}


@cache
def played(name: str) -> tuple[EpisodeResult, FuzzPolicy, list, list[str]]:
    """`play_spec` of the episode `name` of `episode_specs()`, computed once per process."""
    return play_spec(specs_by_name()[name])


def frame_key(frame: ArrayLike) -> str:
    """12 lowercase hex characters identifying a frame's pixel values: the md5 of the values as little-endian int64 in
    C order, so a nested list and an array of any integer dtype with equal values get the same key. The hash only
    identifies states; it serves no security purpose."""
    return hashlib.md5(np.asarray(frame, dtype="<i8").tobytes(), usedforsecurity=False).hexdigest()[:12]


def sha(obj: object) -> str:
    return hashlib.sha256(json.dumps(obj, ensure_ascii=False).encode("utf-8")).hexdigest()[:16]


# What the recording counts in the user messages of an episode (the number of messages that hold the text).
PROMPT_COUNTERS = {
    "rejected_path": "submit REJECTED — path reset to start",
    "rejected_at_start": "submit REJECTED — nothing changed",
    "start_reverted": "(selected start reverted from",
    "start_switched": "start switched to",
    "reset_cleared": "RESET — path cleared",
    "reset_at_start": "RESET — path already at start",
    "blocked": "blocked — cursor stays at",
    "action7": " ACTION7: ",
    "plan_cap": "plan cap:",
    "random_fallback": "[random fallback",
    "attempts": "## Recent Attempts",
    "identical": "IDENTICAL",
    "level_complete": "COMPLETE after",
    "no_visible_change": "no visible change",
}


def digest(result: EpisodeResult, snapshots: list) -> dict:
    """What the recording keeps of one episode: a digest of every call (system, user, response, plan), of the
    steps, the stop reason, the levels completed, and the counters: `PROMPT_COUNTERS` over the user messages, and
    from the snapshots the CONFIRMs that ringed cells, the steps that lit a lamp, the RESETs that put a lit lamp
    out and the snapshots with legend entries."""
    users = [c.user for c in result.calls]
    counters = {k: sum(1 for u in users if text in u) for k, text in PROMPT_COUNTERS.items()}
    executed = [s for s in result.steps if not s.noop_reset]
    rings = lit = put_out = 0
    for step, before, after in zip(executed, snapshots, snapshots[1:]):
        rings += step.action == 5 and bool(after.flagged_cells)
        was = {n for r in before.reveal_nodes for n, on in r.lit.items() if on}
        now = {n for r in after.reveal_nodes for n, on in r.lit.items() if on}
        lit += bool(now - was)
        put_out += step.action == 0 and bool(was - now)
    counters.update(ringing_confirms=rings, lamps_lit=lit, lamps_put_out=put_out,
                    legend_snapshots=sum(1 for t in snapshots if t.legend))
    return {
        "calls": [sha([c.system, c.user, c.response, None if c.plan is None else list(c.plan)])
                  for c in result.calls],
        "steps": sha([[s.level, s.step, s.action, s.actor, s.noop_reset] for s in result.steps]),
        "stop": result.stop_reason,
        "levels": result.levels_completed,
        "counters": counters,
    }


def multi_start_games() -> set[str]:
    """The public games with a level of several starts (from the level files)."""
    out = set()
    for g in GAME_IDS:
        cls = game_class(g)
        for config in load_level_configs(cls.GAME_ID, cls.LEVELS_PACKAGE):
            if len(config.get("starts", ())) > 1:
                out.add(g)
    return out


# ------------------------------------------------------------------------------------------------ prompt checks
_STEP_LINE = re.compile(r"^  #(\d+) (?:RESET|ACTION\d): (.*)$")


def step_texts(result) -> dict[tuple[int, int], str]:
    """The observation line of every step some prompt shows, by (level, 1-based step); one text per step."""
    texts: dict[tuple[int, int], str] = {}
    for call in result.calls:
        for line in call.user.split("\n"):
            m = _STEP_LINE.match(line)
            if m and m.group(1) != "0":
                assert texts.setdefault((call.level, int(m.group(1))), m.group(2)) == m.group(2)
    return texts


def step_states(result) -> list[tuple[object, int, int]]:
    """(step, index before, index after) of every step that did not complete a level: indexes into the snapshots and
    frame keys `play` returns (a RESET not sent to the engine leaves the state as it was)."""
    out, i = [], 0
    for n, step in enumerate(result.steps):
        before = i
        if not step.noop_reset:
            i += 1
        nxt = result.steps[n + 1] if n + 1 < len(result.steps) else None
        completed = (nxt.level != step.level) if nxt is not None else result.levels_completed > step.level
        if not completed:
            out.append((step, before, i))
    return out


def check_nothing_changed(result, snapshots, frames, *, frame_exempt: bool = False) -> int:
    """The harness says "nothing changed" (or "no visible change") of a step, in its line and in the last change, only
    when its views before and after are equal, flags included, and so are its frame keys before and after (not
    checked with `frame_exempt`: tw09 redraws the start state one pixel off its entry frame); and the line of a RESET
    or CONFIRM between equal views says "nothing changed". Returns the number of texts checked."""
    texts = step_texts(result)
    last_changes = {(c.level, c.level_step): line[len("Last change: "):] for c in result.calls
                    for line in c.user.split("\n") if line.startswith("Last change: ")}
    checked = 0
    for step, a, b in step_states(result):
        pre, post = board_view(snapshots[a]), board_view(snapshots[b])
        for is_line, text in ((True, texts.get((step.level, step.step + 1))),
                              (False, last_changes.get((step.level, step.step + 1)))):
            if text is None:
                continue
            checked += 1
            unchanged = "nothing changed" in text or "no visible change" in text
            assert not unchanged or pre == post, (step, text)
            assert not unchanged or frame_exempt or frames[a] == frames[b], (step, text)
            if is_line and step.action in (0, 5) and pre == post:
                assert "nothing changed" in text, (step, text)
    return checked


@cache
def walk_level(cls, index: int, steps: int, episode: int = 0) -> list[tuple[str, str, str]]:
    """(frame key, state key, board text without the ring line) of every state met on random walk number
    `episode` of `steps` actions (moves, CONFIRM and RESET as the loop sends it) on level `index` of `cls` alone."""
    rng = random.Random(f"walk:{cls.__name__}:{index}:{episode}")
    game = single_level(cls, index)
    fd = game.perform_action(ActionInput(id=GameAction.RESET))
    semantics = ActionTracker().semantics()
    budget = budget_line(0, MAX_STEPS_PER_LEVEL, 0, None)
    seen = []

    def note():
        view = board_view(game.truth())
        text = board_block(view, level=0, level_total=1, semantics=semantics, movement="", budget=budget)
        seen.append((frame_key(fd.frame[-1]), state_key(view), re.sub(r"\n\nCells ringed in red by [^\n]*", "", text)))

    note()
    since = 0
    for _ in range(steps):
        action = rng.choice((1, 2, 3, 4, 1, 2, 3, 4, 5, 0))
        if action == 0 and since == 0:
            continue                                     # the loop does not send such a RESET
        fd = game.perform_action(ActionInput(id=GameAction.from_id(action)))
        since = 0 if action == 0 else since + 1
        if fd.levels_completed:
            break
        note()
    return seen


def check_keys_and_texts(seen: list[tuple[str, str, str]]) -> None:
    """Equal state keys give equal board texts (apart from the ring line) and different keys different texts."""
    texts: dict[str, str] = {}
    keys: dict[str, str] = {}
    for _, key, text in seen:
        assert texts.setdefault(key, text) == text, key
        assert keys.setdefault(text, key) == key, text


def partition_conflicts(seen: list[tuple[str, str, str]]) -> set[str]:
    """How frame keys and state keys of the states of a walk disagree: "one frame, two states" and/or "one state, two
    frames"."""
    kinds, frames, states = set(), {}, {}
    for frame, state, _ in seen:
        if frames.setdefault(frame, state) != state:
            kinds.add("one frame, two states")
        if states.setdefault(state, frame) != frame:
            kinds.add("one state, two frames")
    return kinds


# ------------------------------------------------------------------------------------------- scripted answers
class Scripted:
    """Answers `answers` in order, then declines; `users` keeps every user message it was asked."""

    def __init__(self, answers: Sequence[str]) -> None:
        self.answers, self.users = list(answers), []

    def __call__(self, system: str, user: str) -> str | None:
        self.users.append(user)
        return self.answers[len(self.users) - 1] if len(self.users) <= len(self.answers) else None


def play_answers(game: WitnessGame, answers: Sequence[str]) -> tuple[list[str], _EpisodeRun]:
    """Play `answers` on `game` (all its levels); the user messages (the declined one last) and the finished run."""
    policy = Scripted(answers)
    run = _EpisodeRun(game, 0, EpisodeLimits(game.win_score), policy)
    run.play()
    return policy.users, run


def lines(user: str, *prefixes: str) -> list[str]:
    """The lines of `user` that start with one of `prefixes`, in order."""
    return [line for line in user.split("\n") if line.startswith(prefixes)]
