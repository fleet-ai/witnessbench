"""Coordinate types and the immutable board snapshot a game reports through `WitnessGame.truth()`.

Nodes are lattice points `(col, row)` with `0 <= col <= cols` and `0 <= row <= rows`; cells are the unit squares
`(col, row)` with `0 <= col < cols` and `0 <= row < rows`, named by their top-left node. An edge joins two nodes
and is stored normalized, the smaller node first in tuple order.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TypeAlias

Node: TypeAlias = tuple[int, int]
Cell: TypeAlias = tuple[int, int]
Edge: TypeAlias = tuple[Node, Node]


@dataclass(frozen=True)
class Polyomino:
    """A polyomino marker: `shape` lists (x, y) offsets exactly as in the level config; `rotatable` says whether
    rotated placements are allowed."""
    shape: tuple[tuple[int, int], ...]
    rotatable: bool


@dataclass(frozen=True)
class NodeMarkers:
    """A named list of marked nodes declared by a game, in level-file order."""
    name: str
    nodes: tuple[Node, ...]


@dataclass(frozen=True)
class CellMarkers:
    """A named set of marked cells declared by a game. `values[cell]` is None for a plain marker, else an
    int shown as a value (or as a color when `values_are_colours`). `glyph` is the one-character board
    glyph."""
    name: str
    glyph: str
    values: Mapping[Cell, int | None]
    values_are_colours: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "values", MappingProxyType(dict(self.values)))


@dataclass(frozen=True)
class RevealNodes:
    """A named set of nodes declared by a game whose look changes once the path has reached them. `lit[node]` is
    True from the first visit of the node in the current level on; a failed CONFIRM keeps it lit, a level reset or
    a level change makes it unlit again. These nodes are never mandatory."""
    name: str
    lit: Mapping[Node, bool]

    def __post_init__(self) -> None:
        object.__setattr__(self, "lit", MappingProxyType(dict(self.lit)))


@dataclass(frozen=True)
class LegendEntry:
    """One entry of a legend a game draws outside the board: a swatch of the cell marker named `marker` in the
    palette color `colour`, next to an icon whose look `icon` describes."""
    marker: str
    colour: int
    icon: str


def _empty_map() -> Mapping:
    return MappingProxyType({})


@dataclass(frozen=True)
class BoardTruth:
    """Immutable snapshot of one level's state: board size, path, starts, end, broken edges and every
    marker. Coordinates are (col, row); mappings are read-only.

    Attributes:
        cols, rows: Board size in cells.
        path: Nodes of the drawn path, oldest first; the last node is the cursor.
        start: The start the path begins at.
        starts: Every start of the level, in level-file order.
        end: The end node.
        dots: Mandatory dots, in level-file order.
        squares: Painted palette color of each square.
        stars: Palette color of each star.
        triangles: Number of triangles in each cell (1 to 3).
        polyominoes: Polyomino marker of each cell.
        erasers: Cells holding an eraser.
        breakpoints: Broken lattice edges; the path cannot cross them.
        framed_cells: Cells drawn inside a maroon frame.
        node_markers: Named node lists declared by the game, in declaration order.
        cell_markers: Named cell sets declared by the game, each with its board glyph.
        flagged_cells: Cells flagged by the last action, a CONFIRM rejected with the path tip on the end node;
            empty after any other action and in every game that flags nothing.
        reveal_nodes: Named node sets declared by the game whose look changes once visited (`RevealNodes`).
        revealed_markers: Named cell sets the game shows only once the path has reached a reveal node.
        legend: Entries of a legend the game draws outside the board, in drawing order (`LegendEntry`).

    The last four fields are empty unless a game uses them. They are left out of the repr, so a snapshot's repr shows
    only the fields every game has, but they still take part in `==`: two snapshots that differ only in these fields
    compare unequal. A snapshot cannot be hashed (TypeError), because its mapping fields are unhashable.
    """
    cols: int
    rows: int
    path: tuple[Node, ...]
    start: Node
    starts: tuple[Node, ...]
    end: Node
    dots: tuple[Node, ...] = ()
    squares: Mapping[Cell, int] = field(default_factory=_empty_map)
    stars: Mapping[Cell, int] = field(default_factory=_empty_map)
    triangles: Mapping[Cell, int] = field(default_factory=_empty_map)
    polyominoes: Mapping[Cell, Polyomino] = field(default_factory=_empty_map)
    erasers: frozenset[Cell] = frozenset()
    breakpoints: frozenset[Edge] = frozenset()
    framed_cells: frozenset[Cell] = frozenset()
    node_markers: tuple[NodeMarkers, ...] = ()
    cell_markers: tuple[CellMarkers, ...] = ()
    flagged_cells: frozenset[Cell] = field(default=frozenset(), repr=False)
    reveal_nodes: tuple[RevealNodes, ...] = field(default=(), repr=False)
    revealed_markers: tuple[CellMarkers, ...] = field(default=(), repr=False)
    legend: tuple[LegendEntry, ...] = field(default=(), repr=False)

    def __post_init__(self) -> None:
        for name in ("squares", "stars", "triangles", "polyominoes"):
            object.__setattr__(self, name, MappingProxyType(dict(getattr(self, name))))
