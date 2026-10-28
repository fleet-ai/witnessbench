"""Board geometry, pixel drawing and lattice topology shared by all games.

A frame is a 64x64 `numpy.int8` array of palette indices, indexed `frame[y, x]`. A board of `cols x rows`
cells is centered in the frame inside a 4-pixel margin: lattice lines are 1 pixel wide, every cell is a square
of `cell_size` white pixels, and node `(c, r)` sits at pixel `(offset_x + c*(cell_size + 1),
offset_y + r*(cell_size + 1))`. Glyphs are drawn relative to a node pixel or to a cell's center pixel, and
every write outside the frame is skipped.
"""
from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Sequence, Set
from dataclasses import dataclass, field
from typing import Final, TypeAlias

import numpy as np
from numpy.typing import NDArray

from .truth import Cell, Edge, Node

Frame: TypeAlias = NDArray[np.int8]

FRAME_SIZE: Final = 64
MARGIN: Final = 4
MAX_BOARD_DIM: Final = 9

# Palette indices (16-color ARC palette).
WHITE: Final = 0
LIGHT_GRAY: Final = 1
GRAY: Final = 2
DARK_GRAY: Final = 3
CHARCOAL: Final = 4
BLACK: Final = 5
MAGENTA: Final = 6
PINK: Final = 7
RED: Final = 8
BLUE: Final = 9
LIGHT_BLUE: Final = 10
YELLOW: Final = 11
ORANGE: Final = 12
MAROON: Final = 13
GREEN: Final = 14
PURPLE: Final = 15

# Color roles.
BACKGROUND: Final = DARK_GRAY    # panel background and broken-edge gaps
LINE: Final = BLACK              # lattice lines and node pixels
CELL: Final = WHITE              # cell interiors
PATH: Final = BLUE               # the drawn path
SOLVED_PATH: Final = GREEN       # the path in the frame of a winning CONFIRM
CURSOR: Final = YELLOW           # cursor dot at the path tip; the 1x1 cursor sprite
START: Final = GREEN
END: Final = RED
DOT: Final = YELLOW              # a mandatory dot not on the path
VISITED_DOT: Final = GREEN       # a mandatory dot on the path
TRIANGLE: Final = ORANGE
POLYOMINO: Final = PURPLE
ERASER: Final = CHARCOAL
FLAGGED: Final = RED             # the ring on a cell flagged by a rejected CONFIRM

# Facing index -> unit step (dc, dr), which is also the pixel step (dx, dy): 0 north, 1 east, 2 south, 3 west.
FACING: Final[tuple[tuple[int, int], ...]] = ((0, -1), (1, 0), (0, 1), (-1, 0))

# Glyph pixel offsets (dx, dy) from the anchor pixel.
_DOT_PIXELS: Final = ((-1, 0), (0, -1), (0, 0), (1, 0), (0, 1))
_SQUARE_PIXELS: Final = tuple((dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1))
_STAR_PIXELS: Final = ((-2, 0), (-1, -1), (-1, 0), (-1, 1), (0, -2), (0, -1), (0, 0), (0, 1), (0, 2),
                       (1, -1), (1, 0), (1, 1), (2, 0))
_ERASER_PIXELS: Final = ((0, 0), (0, 1), (0, 2), (-1, -1), (1, -1), (-2, -2), (2, -2))


def paint(frame: Frame, pixels: Iterable[tuple[int, int]], colour: int) -> None:
    """Set `frame[y, x] = colour` for every `(x, y)` in `pixels` that lies inside the frame, in order."""
    for x, y in pixels:
        if 0 <= x < FRAME_SIZE and 0 <= y < FRAME_SIZE:
            frame[y, x] = colour


def edge(a: Node, b: Node) -> Edge:
    """The normalized edge between nodes `a` and `b` (smaller node first)."""
    return (a, b) if a <= b else (b, a)


def edge_between(a: Cell, b: Cell) -> Edge:
    """The lattice edge separating the side-adjacent cells `a` and `b` (either order)."""
    assert abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1, (a, b)
    if a[1] == b[1]:
        x = max(a[0], b[0])
        return ((x, a[1]), (x, a[1] + 1))
    y = max(a[1], b[1])
    return ((a[0], y), (a[0] + 1, y))


def path_edges(path: Sequence[Node]) -> frozenset[Edge]:
    """The normalized edges between consecutive nodes of `path` (empty for a path of one node)."""
    return frozenset(edge(p, q) for p, q in zip(path, path[1:]))


def count_sides(cell: Cell, edges: Set[Edge]) -> int:
    """How many of the four sides of `cell` are in `edges`."""
    c, r = cell
    sides = (((c, r), (c + 1, r)), ((c, r + 1), (c + 1, r + 1)), ((c, r), (c, r + 1)), ((c + 1, r), (c + 1, r + 1)))
    return sum(side in edges for side in sides)


def split_regions(cols: int, rows: int, edges: Set[Edge]) -> tuple[frozenset[Cell], ...]:
    """The regions into which `edges` cut the `cols x rows` cells.

    Side-adjacent cells belong to one region unless the edge between them is in `edges`. Regions are ordered
    by their first cell in row-major order (row by row, left to right).
    """
    seen: set[Cell] = set()
    regions: list[frozenset[Cell]] = []
    for r0 in range(rows):
        for c0 in range(cols):
            if (c0, r0) in seen:
                continue
            seen.add((c0, r0))
            region = [(c0, r0)]
            queue = deque(region)
            while queue:
                c, r = queue.popleft()
                for cell in ((c - 1, r), (c + 1, r), (c, r - 1), (c, r + 1)):
                    if (0 <= cell[0] < cols and 0 <= cell[1] < rows and cell not in seen
                            and edge_between((c, r), cell) not in edges):
                        seen.add(cell)
                        region.append(cell)
                        queue.append(cell)
            regions.append(frozenset(region))
    return tuple(regions)


def region_of(cell: Cell, regions: Iterable[frozenset[Cell]]) -> frozenset[Cell]:
    """The first region of `regions` that contains `cell`, or an empty set if none does."""
    return next((region for region in regions if cell in region), frozenset())


def rotate_offsets(offsets: Sequence[tuple[int, int]], facing: int) -> tuple[tuple[int, int], ...]:
    """Rotate an east-facing table of (dx, dy) offsets to `facing` (an index into `FACING`), keeping the order."""
    if not 0 <= facing < len(FACING):
        raise IndexError(f"facing {facing} is outside 0..{len(FACING) - 1}")
    dc, dr = FACING[facing]
    return tuple((x * dc - y * dr, x * dr + y * dc) for x, y in offsets)


@dataclass(frozen=True)
class Grid:
    """Pixel geometry of a `cols x rows` board (1 to 9 cells per side) and the drawing functions on it."""
    cols: int
    rows: int
    cell_size: int = field(init=False)
    offset_x: int = field(init=False)
    offset_y: int = field(init=False)

    def __post_init__(self) -> None:
        if not all(type(n) is int and 1 <= n <= MAX_BOARD_DIM for n in (self.cols, self.rows)):
            raise ValueError(f"board {self.cols!r}x{self.rows!r} is outside 1..{MAX_BOARD_DIM}")
        avail = FRAME_SIZE - 2 * MARGIN
        size = (avail - (max(self.cols, self.rows) + 1)) // max(self.cols, self.rows)
        object.__setattr__(self, "cell_size", size)
        object.__setattr__(self, "offset_x", MARGIN + (avail - (self.cols + 1 + self.cols * size)) // 2)
        object.__setattr__(self, "offset_y", MARGIN + (avail - (self.rows + 1 + self.rows * size)) // 2)

    @property
    def half(self) -> int:
        """Half-width of a cell glyph: `(cell_size - 1) // 2`."""
        return (self.cell_size - 1) // 2

    def node_xy(self, node: Node) -> tuple[int, int]:
        """The (x, y) pixel of a lattice node."""
        return (self.offset_x + node[0] * (self.cell_size + 1), self.offset_y + node[1] * (self.cell_size + 1))

    def cell_center(self, cell: Cell) -> tuple[int, int]:
        """The (x, y) center pixel of a cell."""
        x, y = self.node_xy(cell)
        return (x + 1 + self.cell_size // 2, y + 1 + self.cell_size // 2)

    def new_frame(self) -> Frame:
        """A frame showing the empty board: background, lattice lines and nodes, white cells."""
        frame = np.full((FRAME_SIZE, FRAME_SIZE), BACKGROUND, dtype=np.int8)
        size = self.cell_size
        for r in range(self.rows + 1):
            for c in range(self.cols + 1):
                x, y = self.node_xy((c, r))
                frame[y, x] = LINE
                if c < self.cols:
                    frame[y, x + 1:x + size + 1] = LINE
                if r < self.rows:
                    frame[y + 1:y + size + 1, x] = LINE
        for r in range(self.rows):
            for c in range(self.cols):
                x, y = self.node_xy((c, r))
                frame[y + 1:y + size + 1, x + 1:x + size + 1] = CELL
        return frame

    def draw_dot(self, frame: Frame, node: Node, colour: int) -> None:
        """A 5-pixel plus centered on a node (start, end, cursor and mandatory-dot markers)."""
        x, y = self.node_xy(node)
        paint(frame, ((x + dx, y + dy) for dx, dy in _DOT_PIXELS), colour)

    def draw_edge(self, frame: Frame, a: Node, b: Node, colour: int) -> None:
        """The straight line between two lattice-adjacent nodes, both node pixels included."""
        assert abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1, (a, b)
        (xa, ya), (xb, yb) = self.node_xy(a), self.node_xy(b)
        if xa == xb:
            paint(frame, ((xa, y) for y in range(min(ya, yb), max(ya, yb) + 1)), colour)
        else:
            paint(frame, ((x, ya) for x in range(min(xa, xb), max(xa, xb) + 1)), colour)

    def draw_gap(self, frame: Frame, e: Edge) -> None:
        """A background-colored gap in the middle of a broken edge; the node pixels stay untouched."""
        (xa, ya), (xb, yb) = self.node_xy(e[0]), self.node_xy(e[1])
        k = max(1, (self.cell_size + 1) * 2 // 10)  # the gap starts k pixels from each node: a fifth of their spacing
        if xa == xb:
            paint(frame, ((xa, y) for y in range(min(ya, yb) + k, max(ya, yb) - k + 1)), BACKGROUND)
        else:
            paint(frame, ((x, ya) for x in range(min(xa, xb) + k, max(xa, xb) - k + 1)), BACKGROUND)

    def draw_square(self, frame: Frame, cell: Cell, colour: int) -> None:
        """A 3x3 square on the cell center."""
        cx, cy = self.cell_center(cell)
        paint(frame, ((cx + dx, cy + dy) for dx, dy in _SQUARE_PIXELS), colour)

    def draw_star(self, frame: Frame, cell: Cell, colour: int) -> None:
        """A 13-pixel diamond on the cell center."""
        cx, cy = self.cell_center(cell)
        paint(frame, ((cx + dx, cy + dy) for dx, dy in _STAR_PIXELS), colour)

    def draw_triangles(self, frame: Frame, cell: Cell, count: int) -> None:
        """`count` (1 to 3) small triangles side by side on the cell center row."""
        assert count in (1, 2, 3), count
        cx, cy = self.cell_center(cell)
        h = self.half
        if count == 1:
            xs: tuple[int, ...] = (0,)
        elif count == 2:
            s = max(1, min(2, h - 1))
            xs = (-s, s)
        else:
            s = max(1, min(3, h - 1))
            xs = (-s, 0, s)
        for ox in xs:
            hw = max(0, min(1, h - abs(ox)))
            paint(frame, [(cx + ox, cy - 1)] + [(cx + ox + dx, cy) for dx in range(-hw, hw + 1)], TRIANGLE)

    def draw_polyomino(self, frame: Frame, cell: Cell, shape: Sequence[tuple[int, int]]) -> None:
        """The polyomino `shape` ((x, y) offsets) as a block picture centered on the cell center."""
        assert shape, "empty polyomino shape"
        cx, cy = self.cell_center(cell)
        min_x, max_x = min(x for x, _ in shape), max(x for x, _ in shape)
        min_y, max_y = min(y for _, y in shape), max(y for _, y in shape)
        ex, ey = max_x - min_x + 1, max_y - min_y + 1
        block = max(1, min(2, (2 * self.half + 1) // max(ex, ey)))
        x0, y0 = cx - (ex * block) // 2, cy - (ey * block) // 2
        for sx, sy in shape:
            bx, by = x0 + (sx - min_x) * block, y0 + (sy - min_y) * block
            paint(frame, ((bx + dx, by + dy) for dx in range(block) for dy in range(block)), POLYOMINO)

    def draw_eraser(self, frame: Frame, cell: Cell) -> None:
        """A 7-pixel Y-shaped eraser glyph on the cell center."""
        cx, cy = self.cell_center(cell)
        paint(frame, ((cx + dx, cy + dy) for dx, dy in _ERASER_PIXELS), ERASER)

    def draw_cell_ring(self, frame: Frame, cell: Cell, colour: int) -> None:
        """A 1-pixel ring on the outermost pixels of a cell's interior, just inside its lattice lines."""
        x, y = self.node_xy(cell)
        n = self.cell_size
        paint(frame, [(x + i, y + j) for i in range(1, n + 1) for j in (1, n)]
              + [(x + i, y + j) for j in range(1, n + 1) for i in (1, n)], colour)
