"""The board block of the user message ("## Current Board State"): header, legend, ASCII board, coordinate lists,
the Actions block and the last change.

The ASCII board has one row per node row: node glyphs separated by one character that shows a broken edge. When
the level has cell content or broken edges, a cell row follows every node row but the last. Coordinates are (row,
col) throughout.

- Every list that stands for a set is sorted (`view.sorted_view`), and so are the texts built from them.
- A cell holding two or more kinds of object shows '&'. The legend lists the node glyphs on the board and every kind
  of cell object in the current view, also when '&' covers its glyph.
- Values that are colors are grouped under "Color groups", other values under "Value groups".
- A cell marker may not use a glyph the board reserves (`RESERVED_GLYPHS`) or the glyph of another cell marker,
  revealed ones included (ValueError).
- The cells ringed by the last action, a rejected CONFIRM, are listed after the Actions block; no glyph marks them on
  the board.
- Reveal nodes show the glyph 'Z' and are listed with their look (`view.NODE_LOOKS`) under "Node marks", never under
  "Mandatory dots". The cell markers a reveal node has revealed are shown as cell markers.
- The entries of a legend the game draws outside the board make one sorted line.
"""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import replace
from typing import Final

from .actions import ActionSemantics, confirm_phrase, usage_phrase
from .view import NODE_LOOKS, BoardView, Pos, pos_text, sorted_view

MIXED_GLYPH: Final = "&"
NODE_MARK_GLYPH: Final = "Z"           # a reveal node, whatever its look
# The glyphs the board gives a meaning of its own: node glyphs, '&', the broken-edge
# separators, the glyphs of the built-in cell objects and the triangle digits.
RESERVED_GLYPHS: Final = frozenset("@+SEOZ&|-RPsQ0123456789")
_MIXED: Final = "two_or_more_objects_in_one_cell (each listed under Cell content)"
_RING_LINE: Final = "\nCells ringed in red by the last CONFIRM (rejected; path reset): "
_LEGEND_STRIP: Final = "\nLegend strip (drawn outside the board): "


def _cells(cells: Iterable[Pos]) -> str:
    return ", ".join(pos_text(c) for c in sorted(cells))


def _has_cell_content(view: BoardView) -> bool:
    return bool(view.triangles or view.erasers or view.pieces or view.stars or view.squares or view.framed
                or view.markers)


def _node_glyph(view: BoardView, p: Pos) -> str:
    if p == view.cursor:
        return "@"
    if p in view.path[:-1]:
        return "+"
    if p == view.end:
        return "E"
    if p in view.starts:
        return "S"
    if p in view.dots:
        return "O"
    if any(p in r.lit for r in view.reveal_nodes):
        return NODE_MARK_GLYPH
    return " "


def _cell_kinds(view: BoardView, p: Pos) -> list[str]:
    """The glyph of every kind of object the cell `p` holds, in the fixed order."""
    kinds = []
    if p in view.triangles:
        kinds.append(str(view.triangles[p]))
    kinds += [glyph for glyph, cells in (("R", view.erasers), ("P", view.pieces), ("s", view.stars),
                                         ("Q", view.squares)) if p in cells]
    return kinds + [m.glyph for m in view.markers if p in m.values]


def _cell_glyph(view: BoardView, p: Pos) -> str:
    kinds = _cell_kinds(view, p)
    return MIXED_GLYPH if len(kinds) >= 2 else kinds[0] if kinds else " "


def _board_rows(view: BoardView) -> list[str]:
    """The column header, then the node rows and (if any) the cell rows; rows keep their trailing spaces. Each column
    number stands over its node column."""
    rows = ["   " + "".join(f"{c:<2}" for c in range(view.cols + 1))]
    cell_rows = _has_cell_content(view) or bool(view.broken_h or view.broken_v)
    for r in range(view.rows + 1):
        row = f"{r:>2} "
        for c in range(view.cols + 1):
            row += _node_glyph(view, (r, c))
            if c < view.cols:
                row += "|" if (r, c) in view.broken_h else " "
        rows.append(row)
        if r < view.rows and cell_rows:
            row = "   "
            for c in range(view.cols + 1):
                row += "-" if (r, c) in view.broken_v else " "
                if c < view.cols:
                    row += _cell_glyph(view, (r, c))
            rows.append(row)
    return rows


def _legend(view: BoardView, board_rows: list[str]) -> str:
    """The node glyphs and separators on the board (row labels and the column header excluded), and one entry per
    kind of cell object in the view (the triangle digits share one entry), in a fixed order; '&' when it is on the
    board."""
    present = {ch for row in board_rows[1:] for ch in row[3:]} - {" "}
    entries: list[str] = []
    listed: set[str] = set()

    def add(glyph: str, meaning: str, shown: bool) -> None:
        if shown and glyph not in listed:
            entries.append(f"{glyph}={meaning}")
            listed.add(glyph)

    for glyph, meaning in (("@", "agent"), ("+", "trail"), ("S", "start"), ("E", "end"), ("O", "mandatory_dot")):
        add(glyph, meaning, glyph in present)
    add(NODE_MARK_GLYPH, "/".join(r.name for r in view.reveal_nodes), NODE_MARK_GLYPH in present)
    digits = sorted({str(v) for v in view.triangles.values()})
    if digits:
        entries.append("/".join(digits) + "=triangle_count")
        listed.update(digits)
    add("R", "eraser", bool(view.erasers))
    add("P", "polyomino", bool(view.pieces))
    add("s", "star", bool(view.stars))
    add("Q", "colored_square", bool(view.squares))
    for marker in view.markers:
        add(marker.glyph, marker.name, bool(marker.values))
    add(MIXED_GLYPH, _MIXED, MIXED_GLYPH in present)
    add("|", "broken_H_edge", "|" in present)
    add("-", "broken_V_edge", "-" in present)
    return "  ".join(entries)


def _dot_items(view: BoardView, nodes: Iterable[Pos]) -> str:
    visited = set(view.path)
    return ", ".join(f"{pos_text(n)} [{'visited' if n in visited else 'not visited'}]" for n in nodes)


def _cell_content_lines(view: BoardView) -> list[str]:
    lines = [f"  {pos_text(p)}: triangle count={view.triangles[p]}" for p in sorted(view.triangles)]
    lines += [f"  {pos_text(p)}: eraser" for p in sorted(view.erasers)]
    for p in sorted(view.pieces):
        piece = view.pieces[p]
        shape = ",".join(pos_text(o) for o in piece.offsets)
        lines.append(f"  {pos_text(p)}: polyomino {len(piece.offsets)} cells, shape [{shape}] (row,col offsets)"
                     + (" [rotations allowed]" if piece.rotatable else ""))
    lines += [f"  {pos_text(p)}: star color={view.stars[p]}" for p in sorted(view.stars)]
    lines += [f"  {pos_text(p)}: square color={view.squares[p]}"
              + (" [inside a maroon frame]" if p in view.framed else "") for p in sorted(view.squares)]
    for marker in view.markers:
        word = "color" if marker.values_are_colours else "value"
        for p in sorted(marker.values):
            value = marker.values[p]
            lines.append(f"  {pos_text(p)}: {marker.name}" + ("" if value is None else f" {word}={value}"))
    return lines


def _group_lines(view: BoardView) -> list[str]:
    """"Color groups" (squares, stars and the markers whose values are colors) and "Value groups" (the
    other markers with values), each heading only when it has a line."""
    colour = [f"  Squares of color {k}: " + _cells(p for p, v in view.squares.items() if v == k)
              for k in sorted(set(view.squares.values()))]
    colour += [f"  Stars of color {k}:   " + _cells(p for p, v in view.stars.items() if v == k)
               for k in sorted(set(view.stars.values()))]
    value: list[str] = []
    for marker in view.markers:
        for k in sorted({v for v in marker.values.values() if v is not None}):
            cells = _cells(p for p, v in marker.values.items() if v == k)
            if marker.values_are_colours:
                colour.append(f"  {marker.name.capitalize()} of color {k}: {cells}")
            else:
                value.append(f"  {marker.name.capitalize()} with value {k}: {cells}")
    lines = ["\nColor groups (cells sharing the same color):", *colour] if colour else []
    return lines + (["\nValue groups (cells sharing the same value):", *value] if value else [])


def _node_mark_lines(view: BoardView) -> list[str]:
    """One line per reveal-node field (by name) listing its nodes (by position) with their looks."""
    return [f"Node marks ({r.name}): " + ", ".join(f"{pos_text(p)} {NODE_LOOKS[r.lit[p]]}" for p in sorted(r.lit))
            for r in view.reveal_nodes if r.lit]


def _rendered_view(view: BoardView) -> BoardView:
    """The view the board renders: `sorted_view` with the revealed markers among the cell markers (by name); raises
    ValueError for a cell marker glyph in `RESERVED_GLYPHS` and for a glyph of two cell markers."""
    view = sorted_view(view)
    view = replace(view, markers=tuple(sorted(view.markers + view.revealed, key=lambda m: m.name)), revealed=())
    glyphs = [m.glyph for m in view.markers]
    clash = sorted(set(glyphs) & RESERVED_GLYPHS)
    if clash:
        raise ValueError(f"cell marker glyphs reserved by the board: {' '.join(clash)}")
    shared = sorted({g for g in glyphs if glyphs.count(g) > 1})
    if shared:
        raise ValueError(f"cell marker glyphs of more than one marker: {' '.join(shared)}")
    return view


def _action_lines(semantics: ActionSemantics) -> list[str]:
    lines = []
    for m in semantics.moves:
        if m.status == "untested":
            lines.append(f"  ACTION{m.action}: untested")
            continue
        line = f"  ACTION{m.action}" + (f" ({m.direction.upper()})" if m.direction else "")
        details = [f"noop={format(m.noop_rate, '.0%')}"] if m.noop_rate is not None else []
        if m.inverse is not None:
            details.append(f"inverse=ACTION{m.inverse}")
        if details:
            line += ": " + ", ".join(details)
        lines.append(f"{line} [{usage_phrase(m)}]")
    n = semantics.confirm_presses
    lines.append(f"  ACTION5 (CONFIRM/SUBMIT): {confirm_phrase(n)}" if n
                 else "  ACTION5 (CONFIRM/SUBMIT): not pressed yet this level")
    return lines


def budget_line(level_actions: int, level_cap: int, calls: int, call_cap: int | None) -> str:
    """The budget line: the actions taken on this level and the reflection calls answered so far, each with its
    limit."""
    cap = f" of {call_cap}" if call_cap is not None else " (no limit)"
    return (f"Budget: actions on this level {level_actions} of {level_cap} | reflection calls used {calls}{cap}")


def board_block(view: BoardView, *, level: int, level_total: int, semantics: ActionSemantics,
                movement: str, budget: str) -> str:
    """The board block, without a trailing newline.

    `level` is the 0-based index of the current level, `level_total` the number of levels the episode can play,
    `semantics` the current action semantics (the Actions block), `movement` the last change ("" for none) and
    `budget` the budget line (`budget_line`), printed after the level line.
    """
    view = _rendered_view(view)
    board_rows = _board_rows(view)
    row, col = view.cursor
    parts = [f"Grid: {view.rows + 1}×{view.cols + 1} dot intersections (cursor moves on dots, not in cells) | "
             f"Agent at row={row}, col={col}",
             f"Level: {level + 1}/{level_total}",
             budget,
             "Coordinates: (row, col), 0-indexed, (0,0) at top-left.",
             "Legend: " + _legend(view, board_rows),
             "",
             "Board:",
             *board_rows]
    if len(view.starts) > 1:
        parts.append("\nStarts: " + ", ".join(pos_text(s) + (" [selected]" if s == view.start else "")
                                             for s in view.starts))
    else:
        parts.append(f"\nStart: {pos_text(view.start)}")
    parts.append(f"End:   {pos_text(view.end)}")
    if len(view.dot_fields) >= 2:
        parts += [f"Mandatory dots ({name}): " + _dot_items(view, nodes) for name, nodes in view.dot_fields]
    elif view.dots:
        parts.append("Mandatory dots: " + _dot_items(view, view.dots))
    parts += _node_mark_lines(view)
    cell_content = _cell_content_lines(view)
    if cell_content:
        parts += ["\nCell content:", *cell_content]
    parts += _group_lines(view)
    if view.legend:
        parts.append(_LEGEND_STRIP + "; ".join(f"{e.marker} color={e.colour} next to {e.icon}" for e in view.legend))
    if view.broken_h or view.broken_v:
        parts.append("\nBroken edges (path cannot cross):")
        parts += [f"  H: between (r={r}, c={c}) and (r={r}, c={c + 1})" for r, c in sorted(view.broken_h)]
        parts += [f"  V: between (r={r}, c={c}) and (r={r + 1}, c={c})" for r, c in sorted(view.broken_v)]
    if len(view.path) == 1:
        parts.append(f"\nPath so far: at start {pos_text(view.path[0])}")
    else:
        parts.append(f"\nPath so far ({len(view.path)} nodes): " + " → ".join(pos_text(p) for p in view.path)
                     + " [current]")
    parts += ["", "Actions:", *_action_lines(semantics)]
    if view.flagged:
        parts.append(_RING_LINE + _cells(view.flagged))
    if movement:
        parts.append(f"\nLast change: {movement}")
    return "\n".join(parts)
