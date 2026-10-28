"""What the model is told about the effects of its actions: the observation line of every step, the level-complete
line, the action showcase (one example movement per action) and the "Last change" text of the board block.

All texts are built from the (row, col) views before and after a step and the flag of whether the view's state key
changed.

The text of a step says how the path changed and adds (after " | ") what else became visible: the new look of reveal
nodes ("lamps (2,0) now white_block", fields by name, nodes by position), the revealed cell markers that appeared or
disappeared ("blue_dot now shown at (0,2), (1,0)", "blue_dot no longer shown at (0,2), (1,0)", markers by name,
cells by position), then the rings of a rejected CONFIRM leaving the frame ("red rings cleared"). A rejected
CONFIRM's line names the submitted path (when it had more than its start node) and the cells it ringed. "nothing
changed" is said only when nothing visible changed.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Final

from .episode import CONFIRM, RESET
from .view import NODE_LOOKS, BoardView, MarkerView, Pos, is_wrap, pos_text

_RANDOM_FALLBACK_PREFIX: Final = "[random fallback — your last reflection produced no plan] "
_WRAP_TAG: Final = " [wrap: crossed from one edge to the opposite edge]"
_SHOWCASE_HEADER: Final = ("Action effects (one observed movement per action; CONFIRM and RESET are described in the "
                           "system prompt):")
# Step texts returned by more than one branch below ({at}: the cursor after the step).
_RESET_UNCHANGED: Final = "RESET — path already at start {at}; nothing changed"
_RESET_CLEARED: Final = "RESET — path cleared, cursor at start {at}"
_SUBMIT_UNCHANGED: Final = "submit REJECTED — nothing changed (path is only the start node {at})"
_SUBMIT_REJECTED: Final = "submit REJECTED — "
_RESET_AT_START: Final = "RESET — path already at start {at}"
_AT_START_ONLY: Final = "path is only the start node {at}"
_RINGS_CLEARED: Final = "red rings cleared"
_NO_VISIBLE_CHANGE: Final = "no visible change"


def _wrap_tag(a: Pos, b: Pos, dims: tuple[int, int]) -> str:
    return _WRAP_TAG if is_wrap(a, b, dims) else ""


def _cell_list(cells: Iterable[Pos]) -> str:
    return ", ".join(pos_text(c) for c in sorted(cells))


def submit_details(pre: BoardView, post: BoardView) -> list[str]:
    """What a rejected CONFIRM from `pre` to `post` submitted and drew: the submitted path (when it had
    more than one node) and the cells it ringed (when any), in this order."""
    details = []
    if len(pre.path) > 1:
        details.append(f"submitted path ({len(pre.path)} nodes): " + " → ".join(pos_text(p) for p in pre.path))
    if post.flagged:
        details.append("cells ringed in red: " + _cell_list(post.flagged))
    return details


def look_changes(pre: BoardView, post: BoardView) -> list[str]:
    """The reveal nodes whose look changed from `pre` to `post`, one text per field (by name) and new look
    (lit first), nodes by position."""
    before = {r.name: r.lit for r in pre.reveal_nodes}
    texts = []
    for field in post.reveal_nodes:
        old = before.get(field.name, {})
        for lit in (True, False):
            nodes = [p for p, on in field.lit.items() if on == lit and p in old and old[p] != lit]
            if nodes:
                texts.append(f"{field.name} {_cell_list(nodes)} now {NODE_LOOKS[lit]}")
    return texts


def revealed_changes(pre: BoardView, post: BoardView) -> list[str]:
    """The revealed cell markers that appeared or disappeared from `pre` to `post`, one text per marker
    (by name) and direction (shown first), cells by position, each with its value as Cell content shows it."""
    before = {m.name: m for m in pre.revealed}
    after = {m.name: m for m in post.revealed}
    texts = []
    for name in sorted(before.keys() | after.keys()):
        old, new = before.get(name), after.get(name)
        old_values = {} if old is None else old.values
        new_values = {} if new is None else new.values
        shown = [p for p in new_values if p not in old_values or old_values[p] != new_values[p]]
        gone = [p for p in old_values if p not in new_values]
        if shown:
            texts.append(f"{name} now shown at {_valued_cells(new, shown)}")
        if gone:
            texts.append(f"{name} no longer shown at {_valued_cells(old, gone)}")
    return texts


def _valued_cells(marker: MarkerView, cells: Iterable[Pos]) -> str:
    """The cells (by position) of `marker`, each followed by its value as Cell content shows it, if it has one."""
    word = "color" if marker.values_are_colours else "value"
    return ", ".join(pos_text(p) + ("" if marker.values[p] is None else f" {word}={marker.values[p]}")
                     for p in sorted(cells))


def _side_changes(pre: BoardView, post: BoardView) -> list[str]:
    """What else became visible in a step: the look changes of reveal nodes, the revealed cell markers that appeared
    or disappeared, then "red rings cleared" when `pre` shows the rings of a rejected CONFIRM (every step clears
    them)."""
    return look_changes(pre, post) + revealed_changes(pre, post) + ([_RINGS_CLEARED] if pre.flagged else [])


def transition_line(action: int, pre: BoardView, post: BoardView, *, state_changed: bool,
                    random_fallback: bool) -> str:
    """The observation line of one step from state `pre` to state `post` (the same view for a RESET that was
    not sent to the engine). `state_changed` says whether the view's state key changed. `random_fallback` marks the
    step taken at random after an answer without a plan."""
    line = _changed_line(action, pre, post) if state_changed else _unchanged_line(action, post)
    line = _with_side_changes(line, action, pre, post)
    return _RANDOM_FALLBACK_PREFIX + line if random_fallback else line


def _with_side_changes(line: str, action: int, pre: BoardView, post: BoardView) -> str:
    """The text of a step whose path text is `line`: a rejected CONFIRM names what it submitted and ringed, and the
    side changes follow after " | "; "nothing changed" and "no visible change" are dropped when there is a side
    change."""
    at = pos_text(post.cursor)
    extras = _side_changes(pre, post)
    if action == CONFIRM:
        details = submit_details(pre, post)
        if details or extras:
            rest = (_AT_START_ONLY.format(at=at) if line == _SUBMIT_UNCHANGED.format(at=at)
                    else line.removeprefix(_SUBMIT_REJECTED))
            line = _SUBMIT_REJECTED + "; ".join([*details, rest])
    elif action == RESET and line == _RESET_UNCHANGED.format(at=at) and extras:
        line = _RESET_AT_START.format(at=at)
    elif line == _NO_VISIBLE_CHANGE and extras:
        return " | ".join(extras)
    return " | ".join([line, *extras])


def _unchanged_line(action: int, post: BoardView) -> str:
    at = pos_text(post.cursor)
    if action == RESET:
        return _RESET_UNCHANGED.format(at=at)
    if action == CONFIRM:
        return _SUBMIT_UNCHANGED.format(at=at)
    return f"blocked — cursor stays at {at}"


def _changed_line(action: int, pre: BoardView, post: BoardView) -> str:
    at = pos_text(post.cursor)
    old, new = pre.path, post.path
    switched = pre.start != post.start or old[0] != new[0]
    old_start = pre.start if pre.start != post.start else old[0]
    if action == CONFIRM:
        if len(old) <= 1 and old[-1] == new[-1] and not switched:
            return _SUBMIT_UNCHANGED.format(at=at)
        return f"submit REJECTED — path reset to start {at}" + (
            f" (selected start reverted from {pos_text(old_start)})" if switched else "")
    if action == RESET:
        if len(old) <= 1 and not switched:
            return _RESET_UNCHANGED.format(at=at)
        return _RESET_CLEARED.format(at=at)
    parts = []
    if switched:
        parts.append(f"start switched to {pos_text(new[0])} (was {pos_text(old_start)})")
        if len(new) > 1:
            parts += [f"@{pos_text(new[0])}→{at}", f"trail +{len(new) - 1}"]
    else:
        if old[-1] != new[-1]:
            parts.append(f"@{pos_text(old[-1])}→{at}" + _wrap_tag(old[-1], new[-1], post.dims))
        grown = len(new) - len(old)
        if grown > 0:
            parts.append(f"trail +{grown}")
        elif grown < 0:
            parts.append(f"path shortened to {len(new)} (backtrack)")
    return " | ".join(parts) if parts else _NO_VISIBLE_CHANGE


def breadcrumb(level: int, completed_in: int, level_total: int) -> str:
    """The level-complete line: `level` is the number of the level just completed (1-based) and `completed_in`
    the number of actions taken on it."""
    return (f"LEVEL {level} COMPLETE after {completed_in} actions — now on level {level + 1}/{level_total}: "
            "fresh board")


def showcase(transitions: Sequence[tuple[str, int, str]], snapshots: Mapping[str, BoardView],
             dims: tuple[int, int]) -> str:
    """One example movement per move action, from the current level's steps `(before key, action, after key)` in
    recording order, or "" if there is none.

    A step qualifies if its state key changed, it is neither RESET nor CONFIRM, and the paths of the views stored for
    its keys show one node appended from the same start, one lattice edge (or a whole-axis wrap on a board of
    `dims` = (rows, cols)) away. The latest qualifying step of each action is shown, without wrap tag. Every key
    must be in `snapshots` (KeyError otherwise).
    """
    best: dict[int, str] = {}
    for before, action, after in transitions:
        if before == after or action in (RESET, CONFIRM):
            continue
        old, new = snapshots[before].path, snapshots[after].path
        if not (old[0] == new[0] and len(new) == len(old) + 1 and old[-1] != new[-1]):
            continue
        a, b = old[-1], new[-1]
        if abs(b[0] - a[0]) + abs(b[1] - a[1]) == 1 or is_wrap(a, b, dims):
            best[action] = f"@{pos_text(a)}→{pos_text(b)}"
    if not best:
        return ""
    return _SHOWCASE_HEADER + "\n" + "\n".join(f"  ACTION{a}: {best[a]}" for a in sorted(best))


class MovementTracker:
    """The "Last change" text of the board block: how the most recent executed step changed the board, with the
    side changes of the step after " | " (and then never "nothing changed").

    `reset` at every level entry (the text becomes ""), `update` after every executed engine step, `clear` after
    a RESET that was not sent to the engine (the text becomes "", the compared state is kept).
    """

    def __init__(self) -> None:
        self._baseline: BoardView | None = None
        self._text = ""

    @property
    def text(self) -> str:
        """The current text ("" when there is nothing to report)."""
        return self._text

    def reset(self, view: BoardView) -> None:
        """Enter a level whose first state is `view`."""
        self._baseline, self._text = view, ""

    def clear(self) -> None:
        """Forget the text, keeping the compared state."""
        self._text = ""

    def update(self, view: BoardView, action: int) -> None:
        """Describe the step by `action` from the state compared last to `view`; `view` becomes the compared state.
        A RESET that took a one-node path at another start back to the selected start says "path cleared"."""
        prev = self._baseline
        assert prev is not None, "reset() enters every level before its first step"
        self._text = _movement_text(action, prev, view)
        extras = _side_changes(prev, view)
        if extras:
            at = pos_text(view.cursor)
            if self._text == _RESET_UNCHANGED.format(at=at):
                self._text = _RESET_AT_START.format(at=at)
            self._text = " | ".join([self._text, *extras])
        self._baseline = view


def _movement_text(action: int, prev: BoardView, cur: BoardView) -> str:
    at = pos_text(cur.cursor)
    start_changed = prev.start != cur.start
    if action == CONFIRM and len(cur.path) <= 1:
        if len(prev.path) > 1 or cur.cursor != prev.cursor or start_changed:
            return f"CONFIRM rejected — path reset to start {at}"
        return f"CONFIRM rejected — cursor at start {at}"
    if action == RESET:
        if len(prev.path) > 1 or (prev.start, prev.cursor) != (cur.start, cur.cursor):
            return _RESET_CLEARED.format(at=at)
        return _RESET_UNCHANGED.format(at=at)
    if start_changed:
        return f"start switched to {pos_text(cur.start)}; cursor at {at}"
    if cur.cursor != prev.cursor:
        return f"agent moved from {pos_text(prev.cursor)} to {at}" + _wrap_tag(prev.cursor, cur.cursor, cur.dims)
    return f"agent stayed at {at}"
