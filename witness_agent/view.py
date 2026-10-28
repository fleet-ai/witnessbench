"""The board as the harness sees it: a game's `BoardTruth` in (row, col) coordinates, the text of a position,
the wrap test and the state key.

The harness describes the board to the model from this view only. A game state is identified, for no-op
detection, the transition table and the action statistics, by its state key (`state_key`).
"""
from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass, fields, is_dataclass, replace
from types import MappingProxyType
from typing import Final, TypeVar

from witness_game.truth import BoardTruth, LegendEntry

Pos = tuple[int, int]                     # (row, col); the only coordinate order in witness_agent
_V = TypeVar("_V")

# The look a reveal node is reported with before and after the path has reached it (`RevealView.lit`). A game that
# draws reveal nodes otherwise would need its snapshot to carry the look.
NODE_LOOKS: Final = MappingProxyType({False: "light_gray_plus", True: "white_block"})


@dataclass(frozen=True)
class PieceView:
    """A polyomino marker: its cells as distinct, sorted (row, col) offsets shifted so that the smallest row and
    the smallest column are 0."""

    offsets: tuple[Pos, ...]
    rotatable: bool


@dataclass(frozen=True)
class MarkerView:
    """One named set of marked cells declared by the game. `values[cell]` is None for a plain marker, else an int
    shown as a value (or as a color when `values_are_colours`); `glyph` is its one-character board glyph."""

    name: str
    glyph: str
    values: Mapping[Pos, int | None]
    values_are_colours: bool


@dataclass(frozen=True)
class RevealView:
    """A named set of nodes whose look changes once the path has reached them: `lit[node]` is True from the first
    visit of the node in the current level on (a failed CONFIRM keeps it, a level reset or change clears it)."""

    name: str
    lit: Mapping[Pos, bool]


@dataclass(frozen=True)
class BoardView:
    """Immutable (row, col) snapshot of one level's state; every mapping is read-only.

    Attributes:
        rows, cols: Board size in cells.
        path: Nodes of the drawn path, oldest first; never empty.
        start: The start the path begins at.
        starts: Every start of the level, in level-file order.
        end: The end node.
        dots: Mandatory dots: the level's dot list in order, then the nodes of every node marker not listed yet.
        dot_fields: (name, nodes) of every node marker, in declaration order.
        triangles, stars, squares: Triangle count, star color and square color of each marked cell.
        erasers: Cells holding an eraser.
        pieces: Polyomino of each marked cell.
        framed: Cells drawn inside a maroon frame.
        markers: The game's named cell markers, sorted by name.
        broken_h: Left node (r, c) of every broken edge (r, c)-(r, c+1).
        broken_v: Top node (r, c) of every broken edge (r, c)-(r+1, c).
        flagged: Cells ringed by the last action, a CONFIRM rejected with the tip on the end node.
        reveal_nodes: The game's reveal nodes, sorted by name.
        revealed: The cell markers the game shows once a reveal node is reached, sorted by name.
        legend: The entries of a legend the game draws outside the board, in the game's order.
    """

    rows: int
    cols: int
    path: tuple[Pos, ...]
    start: Pos
    starts: tuple[Pos, ...]
    end: Pos
    dots: tuple[Pos, ...]
    dot_fields: tuple[tuple[str, tuple[Pos, ...]], ...]
    triangles: Mapping[Pos, int]
    erasers: frozenset[Pos]
    pieces: Mapping[Pos, PieceView]
    stars: Mapping[Pos, int]
    squares: Mapping[Pos, int]
    framed: frozenset[Pos]
    markers: tuple[MarkerView, ...]
    broken_h: frozenset[Pos]
    broken_v: frozenset[Pos]
    flagged: frozenset[Pos] = frozenset()
    reveal_nodes: tuple[RevealView, ...] = ()
    revealed: tuple[MarkerView, ...] = ()
    legend: tuple[LegendEntry, ...] = ()

    @property
    def cursor(self) -> Pos:
        """The path tip."""
        return self.path[-1]

    @property
    def dims(self) -> tuple[int, int]:
        """(rows, cols)."""
        return (self.rows, self.cols)


def pos_text(p: Pos) -> str:
    """A position as the prompts show it: `(row,col)`."""
    return f"({p[0]},{p[1]})"


def is_wrap(a: Pos, b: Pos, dims: tuple[int, int]) -> bool:
    """Whether the step a -> b spans the whole board along one axis, on a board of `dims` = (rows, cols) cells (a
    board of one row or column has no wrap)."""
    rows, cols = dims
    dr, dc = b[0] - a[0], b[1] - a[1]
    return (dr == 0 and abs(dc) == cols > 1) or (dc == 0 and abs(dr) == rows > 1)


def _rc(node: tuple[int, int]) -> Pos:
    """An env `(col, row)` coordinate as `(row, col)`."""
    return (node[1], node[0])


def _swapped(mapping: Mapping[tuple[int, int], _V]) -> Mapping[Pos, _V]:
    return MappingProxyType({_rc(k): v for k, v in mapping.items()})


def _piece(shape: tuple[tuple[int, int], ...], rotatable: bool) -> PieceView:
    offsets = sorted({(y, x) for x, y in shape})
    r0 = min(r for r, _ in offsets)
    c0 = min(c for _, c in offsets)
    return PieceView(tuple((r - r0, c - c0) for r, c in offsets), rotatable)


def board_view(truth: BoardTruth) -> BoardView:
    """The (row, col) view of a board snapshot."""
    dots = [_rc(d) for d in truth.dots]
    for field in truth.node_markers:
        for node in field.nodes:
            if _rc(node) not in dots:
                dots.append(_rc(node))
    broken_h: set[Pos] = set()
    broken_v: set[Pos] = set()
    for a, b in truth.breakpoints:
        (r1, c1), (r2, c2) = _rc(a), _rc(b)
        assert abs(r1 - r2) + abs(c1 - c2) == 1, (a, b)      # the games accept unit edges only
        if r1 == r2:
            broken_h.add((r1, min(c1, c2)))
        else:
            broken_v.add((min(r1, r2), c1))
    markers = sorted((MarkerView(m.name, m.glyph, _swapped(m.values), m.values_are_colours)
                      for m in truth.cell_markers), key=lambda m: m.name)
    revealed = sorted((MarkerView(m.name, m.glyph, _swapped(m.values), m.values_are_colours)
                       for m in truth.revealed_markers), key=lambda m: m.name)
    reveal_nodes = sorted((RevealView(r.name, _swapped(r.lit)) for r in truth.reveal_nodes), key=lambda r: r.name)
    return BoardView(
        rows=truth.rows, cols=truth.cols, path=tuple(_rc(p) for p in truth.path), start=_rc(truth.start),
        starts=tuple(_rc(s) for s in truth.starts), end=_rc(truth.end), dots=tuple(dots),
        dot_fields=tuple((f.name, tuple(_rc(n) for n in f.nodes)) for f in truth.node_markers),
        triangles=_swapped(truth.triangles), erasers=frozenset(_rc(c) for c in truth.erasers),
        pieces=MappingProxyType({_rc(c): _piece(p.shape, p.rotatable) for c, p in truth.polyominoes.items()}),
        stars=_swapped(truth.stars), squares=_swapped(truth.squares),
        framed=frozenset(_rc(c) for c in truth.framed_cells), markers=tuple(markers),
        broken_h=frozenset(broken_h), broken_v=frozenset(broken_v),
        flagged=frozenset(_rc(c) for c in truth.flagged_cells), reveal_nodes=tuple(reveal_nodes),
        revealed=tuple(revealed), legend=tuple(truth.legend))


def sorted_view(view: BoardView) -> BoardView:
    """`view` with every list that stands for a set in a fixed order: the starts, the dots and the nodes of each
    dot field by (row, col), the dot fields, cell markers, revealed markers and reveal nodes by name, the legend
    entries by (`marker`, `colour`, `icon`). The path keeps its order."""
    return replace(view, starts=tuple(sorted(view.starts)), dots=tuple(sorted(view.dots)),
                   dot_fields=tuple(sorted((name, tuple(sorted(nodes))) for name, nodes in view.dot_fields)),
                   markers=tuple(sorted(view.markers, key=lambda m: m.name)),
                   revealed=tuple(sorted(view.revealed, key=lambda m: m.name)),
                   reveal_nodes=tuple(sorted(view.reveal_nodes, key=lambda r: r.name)),
                   legend=tuple(sorted(view.legend, key=lambda e: (e.marker, e.colour, e.icon))))


def _canonical(x: object) -> object:
    """A form of `x` whose repr does not depend on the order of mappings and sets."""
    if is_dataclass(x) and not isinstance(x, type):
        return (type(x).__name__, *((f.name, _canonical(getattr(x, f.name))) for f in fields(x)))
    if isinstance(x, Mapping):
        return ("map", *sorted((_canonical(k), _canonical(v)) for k, v in x.items()))
    if isinstance(x, (set, frozenset)):
        return ("set", *sorted(_canonical(e) for e in x))
    if isinstance(x, tuple):
        return ("tuple", *(_canonical(e) for e in x))
    return x


def state_key(view: BoardView) -> str:
    """The state key: 12 lowercase hex characters of the md5 of a canonical form of everything the view holds except
    the flagged cells (lists that stand for sets in `sorted_view` order, mappings and sets sorted), so the key does
    not depend on the order a snapshot lists anything in. It only identifies states and serves no security
    purpose.

    The flagged cells are left out: the next action clears them, so a blocked move right after a flagging CONFIRM
    keeps the key (a no-op), and the start state after a rejected CONFIRM has the key it had before the attempt.
    """
    canonical = _canonical(replace(sorted_view(view), flagged=frozenset()))
    return hashlib.md5(repr(canonical).encode("utf-8"), usedforsecurity=False).hexdigest()[:12]
