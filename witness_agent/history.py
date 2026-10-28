"""The steps of the current level: the transition table read by the rule checks and the action showcase, and
the observation window of the most recent steps shown in the user message."""
from __future__ import annotations

from collections import deque
from typing import Final

from .episode import CONFIRM, RESET

Transition = tuple[str, int, str]          # (state key before, action, state key after)

OBSERVATION_WINDOW: Final = 8              # observation lines shown in one user message


class TransitionTable:
    """The executed steps of one level as (before, action, after) triples of state keys (`view.state_key`).

    For each distinct `before` key only the latest outcome of each action counts: the action moved if `after`
    differs from `before`, and stayed otherwise.
    """

    def __init__(self) -> None:
        self._transitions: list[Transition] = []
        self._latest: dict[str, dict[int, bool]] = {}     # before -> {action: moved}

    def record(self, before: str, action: int, after: str) -> None:
        """Add one executed step; it replaces any earlier outcome of `action` from `before`."""
        self._transitions.append((before, action, after))
        self._latest.setdefault(before, {})[action] = before != after

    def outcome_counts(self, action: int) -> tuple[int, int]:
        """(moved, stayed): the numbers of before-keys whose latest outcome of `action` moved or stayed; keys
        from which `action` was never taken count in neither."""
        moved = sum(1 for outcomes in self._latest.values() if outcomes.get(action) is True)
        stayed = sum(1 for outcomes in self._latest.values() if outcomes.get(action) is False)
        return moved, stayed

    def transitions(self) -> tuple[Transition, ...]:
        """Every recorded step, in recording order."""
        return tuple(self._transitions)


class ObservationWindow:
    """The observation lines of the last `OBSERVATION_WINDOW` steps of one level (a new window per level)."""

    def __init__(self) -> None:
        self._entries: deque[tuple[int, int, str, bool]] = deque(maxlen=OBSERVATION_WINDOW)

    def add(self, step: int, action: int, line: str) -> None:
        """Add the line of the step numbered `step` (1-based within the level)."""
        self._entries.append((step, action, line, False))

    def add_breadcrumb(self, line: str) -> None:
        """Add the level-complete line; it is shown as step 0 with action CONFIRM, tagged " [level_complete]"."""
        self._entries.append((0, CONFIRM, line, True))

    def text(self) -> str:
        """The window as shown in the user message."""
        if not self._entries:
            return "No informative observations yet."
        return "Recent actions:\n" + "\n".join(
            f"  #{step} {'RESET' if action == RESET else f'ACTION{action}'}: {line}"
            + (" [level_complete]" if level_complete else "")
            for step, action, line, level_complete in self._entries)
