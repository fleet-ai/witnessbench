"""Markers, config parsing, marker drawing and region rules shared by the tw games, and their base class `TwGame`.

A path splits the board's cells into regions (`witness_env.grid.split_regions`). The rule functions below count
the errors of one rule in one region; a region satisfies a rule when it has no error of that rule. In the games
with erasers, each eraser cancels one error of its own region, and a region is satisfied only when it holds
exactly as many erasers as it has errors.

Polyomino pieces are placed by `can_tile`: a region holding pieces is satisfied when its cells can be partitioned
into translated copies of exactly those pieces, each used once, rotated by multiples of 90 degrees only if the
piece allows rotation, never mirrored. The answer depends only on the region and the pieces, never on the order
in which either is listed.
"""
from __future__ import annotations

import dataclasses
import enum
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence, Set
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, ClassVar

from .base import Board, WitnessGame, parse_cell_map, parse_coords, parse_int_map
from .grid import DOT, VISITED_DOT, Frame, count_sides
from .truth import BoardTruth, Cell, Edge, Node, Polyomino


@dataclass(frozen=True)
class TwMarkers:
    """The markers of one tw level, parsed from its config; mappings are read-only and keep config order.

    Attributes:
        dots: Nodes the path must visit (config `dots`).
        squares: Painted color of each square (config `squares`).
        stars: Color of each star (config `stars`).
        triangles: Number of triangles (1 to 3) in each cell (config `triangles`).
        polyominoes: Polyomino piece shown in each cell (config `tetris`).
        erasers: Cells holding an eraser (config `erasers`).
        filters: For each framed cell, the color its square counts as (config `filters`); the frame is drawn,
            this color never is.
    """
    dots: tuple[Node, ...] = ()
    squares: Mapping[Cell, int] = field(default_factory=dict)
    stars: Mapping[Cell, int] = field(default_factory=dict)
    triangles: Mapping[Cell, int] = field(default_factory=dict)
    polyominoes: Mapping[Cell, Polyomino] = field(default_factory=dict)
    erasers: frozenset[Cell] = frozenset()
    filters: Mapping[Cell, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("squares", "stars", "triangles", "polyominoes", "filters"):
            object.__setattr__(self, name, MappingProxyType(dict(getattr(self, name))))


class Layer(enum.Enum):
    """A kind of marker drawn by `TwGame._draw_markers`; a game lists its layers in drawing order."""
    DOTS = "dots"
    SQUARES = "squares"
    STARS = "stars"
    TRIANGLES = "triangles"
    POLYOMINOES = "polyominoes"
    ERASERS = "erasers"


# -------------------------------------------------------------------------------------------------- the rules
def dots_visited(dots: Sequence[Node], path: Sequence[Node]) -> bool:
    """Whether every dot is a node of `path`."""
    nodes = set(path)
    return all(dot in nodes for dot in dots)


def squares_errors(region: Set[Cell], squares: Mapping[Cell, int]) -> int:
    """Errors of the squares rule in `region`: k - 1 when its squares show k >= 2 different colors, else 0."""
    colours = {colour for cell, colour in squares.items() if cell in region}
    return max(len(colours) - 1, 0)


def stars_errors(region: Set[Cell], stars: Mapping[Cell, int]) -> int:
    """Errors of the stars rule in `region`: |n - 2| for every color that has n >= 1 stars in the region."""
    counts = Counter(colour for cell, colour in stars.items() if cell in region)
    return sum(abs(n - 2) for n in counts.values())


def triangles_errors(region: Set[Cell], triangles: Mapping[Cell, int], edges: Set[Edge]) -> int:
    """Errors of the triangles rule in `region`: the triangle cells whose number of sides in `edges` (the edges
    of the whole path) differs from their number of triangles."""
    return sum(1 for cell, count in triangles.items() if cell in region and count_sides(cell, edges) != count)


def _normalise(cells: Iterable[tuple[int, int]]) -> frozenset[tuple[int, int]]:
    """`cells` translated so that their minimum x and minimum y are 0."""
    points = tuple(cells)
    min_x, min_y = min(x for x, _ in points), min(y for _, y in points)
    return frozenset((x - min_x, y - min_y) for x, y in points)


def orientations(piece: Polyomino) -> tuple[frozenset[tuple[int, int]], ...]:
    """The distinct orientations `piece` may be placed in, each translated so its minimum x and y are 0: the
    shape as given, then (for a rotatable piece) its successive quarter turns `(x, y) -> (y, -x)`."""
    out = [_normalise(piece.shape)]
    if piece.rotatable:
        for _ in range(3):
            turned = _normalise((y, -x) for x, y in out[-1])
            if turned not in out:
                out.append(turned)
    return tuple(out)


def can_tile(region: Set[Cell], pieces: Sequence[Polyomino]) -> bool:
    """Whether the cells of `region` are exactly partitioned by translated copies of `pieces`, each piece used
    once in one of its `orientations`.

    The search covers the smallest uncovered cell (in `(col, row)` order) with every unused piece, in every
    orientation, with every cell of the orientation on that cell, so it finds a partition whenever one exists.
    """
    if sum(len(piece.shape) for piece in pieces) != len(region):
        return False
    shapes = tuple(orientations(piece) for piece in pieces)

    def fill(uncovered: frozenset[Cell], unused: tuple[int, ...]) -> bool:
        if not uncovered:
            return not unused
        ax, ay = min(uncovered)
        for i in unused:
            rest = tuple(j for j in unused if j != i)
            for shape in shapes[i]:
                for rx, ry in shape:
                    placed = frozenset((x + ax - rx, y + ay - ry) for x, y in shape)
                    if placed <= uncovered and fill(uncovered - placed, rest):
                        return True
        return False

    return fill(frozenset(region), tuple(range(len(pieces))))


def tiling_error(region: Set[Cell], polyominoes: Mapping[Cell, Polyomino]) -> int:
    """Errors of the tiling rule in `region`: 1 when the pieces shown in the region cannot tile it (`can_tile`),
    else 0; a region without pieces has no error."""
    pieces = [polyominoes[cell] for cell in sorted(region) if cell in polyominoes]
    return int(bool(pieces) and not can_tile(region, pieces))


def region_errors(region: Set[Cell], markers: TwMarkers, edges: Set[Edge]) -> int:
    """The errors of the squares, stars, triangles and tiling rules in `region`, added up."""
    return (squares_errors(region, markers.squares) + stars_errors(region, markers.stars)
            + triangles_errors(region, markers.triangles, edges) + tiling_error(region, markers.polyominoes))


def erasers_balance(regions: Iterable[Set[Cell]], markers: TwMarkers, edges: Set[Edge]) -> bool:
    """Whether every region holds exactly as many erasers as it has errors (`region_errors`)."""
    return all(region_errors(region, markers, edges) == len(region & markers.erasers) for region in regions)


# --------------------------------------------------------------------------------------------- config parsing
def _parse_polyominoes(value: object) -> dict[Cell, Polyomino]:
    """The `tetris` config entry: per cell an object with a non-empty `shape` of distinct `[x, y]` pairs and
    the optional booleans `rotated` (default false) and `negative` (only false is accepted)."""
    pieces: dict[Cell, Polyomino] = {}
    for (c, r), entry in parse_cell_map(value, "tetris").items():
        where = f"tetris: '{c},{r}'"
        if not isinstance(entry, dict):
            raise ValueError(f"{where}: expected an object with a 'shape', got {entry!r}")
        unknown = set(entry) - {"shape", "rotated", "negative"}
        if unknown:
            raise ValueError(f"{where}: unknown key {min(unknown)!r}")
        if "shape" not in entry:
            raise ValueError(f"{where}: missing 'shape'")
        shape = parse_coords(entry["shape"], f"{where} shape")
        if not shape or len(set(shape)) != len(shape):
            raise ValueError(f"{where} shape: expected a non-empty list of distinct [x, y] pairs, "
                             f"got {entry['shape']!r}")
        for flag in ("rotated", "negative"):
            if flag in entry and type(entry[flag]) is not bool:
                raise ValueError(f"{where}: '{flag}' must be true or false, got {entry[flag]!r}")
        if entry.get("negative", False):
            raise ValueError(f"{where}: negative polyomino pieces are not supported")
        pieces[(c, r)] = Polyomino(shape, entry.get("rotated", False))
    return pieces


# --------------------------------------------------------------------------------------------- the base class
class TwGame(WitnessGame[TwMarkers]):
    """Base of the tw games: parses, draws and reports the markers a game declares.

    Class attributes (set by every game besides `GAME_ID` and `CONFIG_KEYS`):
        REQUIRED: Marker keys every level config must contain; the other marker keys of `CONFIG_KEYS` are
            optional.
        LAYERS: The marker kinds drawn, in drawing order.

    Each game implements `_wins` as a static method.
    """
    REQUIRED: ClassVar[frozenset[str]]
    LAYERS: ClassVar[tuple[Layer, ...]]

    @classmethod
    def _parse_markers(cls, config: Mapping[str, Any], board: Board) -> TwMarkers:
        """The markers of one level config; reads only the game's own marker keys. Raises KeyError for the
        smallest missing required key and ValueError for a malformed marker entry."""
        missing = sorted(cls.REQUIRED - config.keys())
        if missing:
            raise KeyError(missing[0])

        def has(key: str) -> bool:
            return key in cls.CONFIG_KEYS and key in config

        return TwMarkers(
            dots=parse_coords(config["dots"], "dots") if has("dots") else (),
            squares=parse_int_map(config["squares"], "squares") if has("squares") else {},
            stars=parse_int_map(config["stars"], "stars") if has("stars") else {},
            triangles=parse_int_map(config["triangles"], "triangles", allowed={1, 2, 3}) if has("triangles") else {},
            polyominoes=_parse_polyominoes(config["tetris"]) if has("tetris") else {},
            erasers=frozenset(parse_coords(config["erasers"], "erasers")) if has("erasers") else frozenset(),
            filters=parse_int_map(config["filters"], "filters") if has("filters") else {},
        )

    @classmethod
    def _draw_markers(cls, frame: Frame, board: Board, markers: TwMarkers, visited: Set[Node]) -> None:
        """Draw the layers of `LAYERS` in order, each layer's markers in config order (erasers sorted). A dot is
        drawn in the visited-dot color when its node is in `visited`."""
        grid = board.grid
        for layer in cls.LAYERS:
            match layer:
                case Layer.DOTS:
                    for node in markers.dots:
                        grid.draw_dot(frame, node, VISITED_DOT if node in visited else DOT)
                case Layer.SQUARES:
                    for cell, colour in markers.squares.items():
                        grid.draw_square(frame, cell, colour)
                case Layer.STARS:
                    for cell, colour in markers.stars.items():
                        grid.draw_star(frame, cell, colour)
                case Layer.TRIANGLES:
                    for cell, count in markers.triangles.items():
                        grid.draw_triangles(frame, cell, count)
                case Layer.POLYOMINOES:
                    for cell, piece in markers.polyominoes.items():
                        grid.draw_polyomino(frame, cell, piece.shape)
                case Layer.ERASERS:
                    for cell in sorted(markers.erasers):
                        grid.draw_eraser(frame, cell)

    @staticmethod
    def _truth_markers(truth: BoardTruth, markers: TwMarkers) -> BoardTruth:
        """`truth` with the dots, squares (painted colors), stars, triangles, polyominoes and erasers of
        `markers`, and the filter cells as the framed cells."""
        return dataclasses.replace(truth, dots=markers.dots, squares=markers.squares, stars=markers.stars,
                                   triangles=markers.triangles, polyominoes=markers.polyominoes,
                                   erasers=markers.erasers, framed_cells=frozenset(markers.filters))
