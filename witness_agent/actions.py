"""What the harness knows about the actions on the current level: the direction and effect of each move action
(ACTION1-ACTION4) and the number of CONFIRM presses.

`ActionTracker` learns this from the executed steps: per move action how often it ran and left the state
unchanged (states are told apart by the view's state key), which other move undoes it, and the direction its cursor
movements vote for. Directions are assigned so that no two actions share one; the assignment of a finished level
carries over as a fallback for the actions that show no direction on the next level.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Final, Literal

from .episode import CONFIRM
from .view import BoardView, Pos, is_wrap

Direction = Literal["up", "down", "left", "right"]
MoveStatus = Literal["untested", "carried", "direction_carried", "observed"]

_MOVES: Final = (1, 2, 3, 4)
_OPPOSITE: Final[Mapping[Direction, Direction]] = MappingProxyType(
    {"up": "down", "down": "up", "left": "right", "right": "left"})


@dataclass(frozen=True)
class MoveSemantics:
    """What is known about one move action.

    `status` is "untested" (not executed on this level and no direction known), "carried" (not executed on this
    level; the direction comes from earlier levels), "direction_carried" (executed on this level without showing a
    direction; the direction from earlier levels stands) or "observed" (executed on this level otherwise; the
    direction may be unknown).
    """

    action: int                        # 1-4
    status: MoveStatus
    direction: Direction | None
    times: int                         # executions of this action on this level (0 unless observed/direction_carried)
    noop_rate: float | None            # fraction of those executions that left the state unchanged; None iff times == 0
    inverse: int | None                # the action that undoes this one, only when times > 0


@dataclass(frozen=True)
class ActionSemantics:
    """What is known about all actions on the current level."""

    moves: tuple[MoveSemantics, ...]   # exactly ACTION1..ACTION4, ascending
    confirm_presses: int               # CONFIRMs executed on this level; none completed it, so each was rejected


# The wording shared by the two texts of the action semantics: the Actions block of the board (`board.py`) and the
# "Known Action Semantics" of the system prompt (`prompts.py`).
def usage_phrase(move: MoveSemantics) -> str:
    """Where the knowledge of a tested or carried move comes from: an earlier level, or its uses on this level."""
    if move.status == "carried":
        return "learned on an earlier level — unverified here"
    if move.status == "direction_carried":
        return f"direction carried from an earlier level; used ×{move.times} here"
    return f"observed ×{move.times} this level"


def confirm_phrase(presses: int) -> str:
    """The CONFIRM presses of this level, every one of them rejected."""
    return f"pressed ×{presses} this level, {presses} rejected"


def move_vote(pre: BoardView, post: BoardView) -> tuple[Pos, Pos] | None:
    """The cursor movement a step shows, as (from, to), or None if it shows none.

    If the step switched the selected start, the movement runs from the new start to the new cursor (None when
    the path is only that start); otherwise it runs from the old cursor to the new one (None when unmoved).
    """
    if pre.start != post.start:
        if len(post.path) <= 1 or post.start == post.cursor:
            return None
        return (post.start, post.cursor)
    return (pre.cursor, post.cursor) if pre.cursor != post.cursor else None


def _step_direction(dr: int, dc: int) -> Direction:
    """The direction of a step along one axis (only the sign of the nonzero component counts)."""
    if dr:
        return "down" if dr > 0 else "up"
    return "right" if dc > 0 else "left"


@dataclass
class _LevelCounts:
    """The per-level statistics of an `ActionTracker`."""

    outcomes: defaultdict[tuple[str, int], set[str]] = field(default_factory=lambda: defaultdict(set))
    times: Counter[int] = field(default_factory=Counter)
    noops: Counter[int] = field(default_factory=Counter)
    votes: defaultdict[int, list[Direction]] = field(default_factory=lambda: defaultdict(list))
    inverse: dict[int, int] = field(default_factory=dict)
    inverse_evidence: defaultdict[tuple[int, int], set[tuple[str, str]]] = field(
        default_factory=lambda: defaultdict(set))
    presses: int = 0


class ActionTracker:
    """The action statistics of one episode; per-level counts restart at every `new_level`."""

    def __init__(self) -> None:
        self._prior: dict[int, Direction] = {}     # the directions carried over from finished levels
        self._level = _LevelCounts()

    def record(self, action: int, pre_key: str, post_key: str, move: tuple[Pos, Pos] | None,
               dims: tuple[int, int]) -> None:
        """Record one step that did not complete a level: its action, the state keys before and after it, its
        cursor movement (`move_vote`) and the board's (rows, cols).

        A CONFIRM counts one (rejected) press. A move action updates its statistics: a step whose keys are equal
        is a no-op; a step from key A to key B for which another move action once led from B back to A is
        evidence that the two are inverses (two distinct key pairs make them inverses, replacing earlier
        partners); a movement by one lattice edge votes for its direction, and a jump across the whole board
        along one axis votes for the opposite direction (a wrap). Every other action changes nothing.
        """
        if action == CONFIRM:
            self._level.presses += 1
            return
        if action not in _MOVES:
            return
        level = self._level
        level.outcomes[(pre_key, action)].add(post_key)
        level.times[action] += 1
        if pre_key == post_key:
            level.noops[action] += 1
        else:
            for other in _MOVES:
                if other != action and pre_key in level.outcomes.get((post_key, other), ()):
                    evidence = level.inverse_evidence[(min(action, other), max(action, other))]
                    evidence.add((min(pre_key, post_key), max(pre_key, post_key)))
                    if len(evidence) >= 2:
                        level.inverse[action] = other
                        level.inverse[other] = action
        if move is not None:
            dr, dc = move[1][0] - move[0][0], move[1][1] - move[0][1]
            if abs(dr) + abs(dc) == 1:
                level.votes[action].append(_step_direction(dr, dc))
            elif is_wrap(move[0], move[1], dims):
                level.votes[action].append(_step_direction(-dr, -dc))

    def new_level(self) -> None:
        """Enter a new level: the current direction assignment joins the carried directions (merged; carried
        directions of actions left unassigned stay), and every per-level count restarts."""
        self._prior.update(self._assign())
        self._level = _LevelCounts()

    def _assign(self) -> dict[int, Direction]:
        """Directions: actions with votes in order of clarity (share of the top direction, ties in action order),
        each taking its most voted free direction that agrees with its inverse's; an inverse without a direction
        takes the opposite one. Actions still without a direction take their carried direction if it is free."""
        tallies = {action: Counter(votes) for action, votes in sorted(self._level.votes.items())}

        def clarity(action: int) -> float:
            counts = tallies[action]
            return counts.most_common(1)[0][1] / sum(counts.values())

        taken: set[Direction] = set()
        result: dict[int, Direction] = {}
        for action in sorted(tallies, key=clarity, reverse=True):
            inverse = self._level.inverse.get(action)
            for direction, _ in tallies[action].most_common():
                if direction in taken or (inverse in result and result[inverse] != _OPPOSITE[direction]):
                    continue
                taken.add(direction)
                result[action] = direction
                if inverse is not None and inverse not in result and _OPPOSITE[direction] not in taken:
                    taken.add(_OPPOSITE[direction])
                    result[inverse] = _OPPOSITE[direction]
                break
        for action in _MOVES:
            carried = self._prior.get(action)
            if action not in result and carried is not None and carried not in taken:
                taken.add(carried)
                result[action] = carried
        return result

    def observed_directions(self) -> dict[int, Direction]:
        """The assigned direction of each move action whose cursor movements on this level voted (the rule check
        confirms a direction claim only against these)."""
        return {action: direction for action, direction in self._assign().items() if self._level.votes.get(action)}

    def semantics(self) -> ActionSemantics:
        """What is known now about ACTION1-ACTION4 and the CONFIRM presses of this level."""
        directions = self._assign()
        level = self._level
        moves = []
        for action in _MOVES:
            direction = directions.get(action)
            times = level.times[action]
            if times == 0:
                status: MoveStatus = "carried" if direction else "untested"
                moves.append(MoveSemantics(action=action, status=status, direction=direction, times=0,
                                           noop_rate=None, inverse=None))
                continue
            carried = direction is not None and not level.votes.get(action) and self._prior.get(action) == direction
            moves.append(MoveSemantics(action=action, status="direction_carried" if carried else "observed",
                                       direction=direction, times=times, noop_rate=level.noops[action] / times,
                                       inverse=level.inverse.get(action)))
        return ActionSemantics(tuple(moves), level.presses)
