"""The line-puzzle game base on arcengine: level files, config parsing, movement, CONFIRM and frames.

Every game is a `WitnessGame` subclass that declares its id and config keys and implements four hooks (marker
parsing, truth markers, marker drawing, win predicate). The base does everything the games share: it parses
each level once at construction, keeps the drawn path, moves it on ACTION1-ACTION4 (up, down, left, right),
checks the level's rule on ACTION5 (CONFIRM) and renders every frame.
"""
from __future__ import annotations

import dataclasses
import importlib.resources
import json
import os
import re
from abc import abstractmethod
from collections.abc import Mapping, Sequence, Set
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, ClassVar, Final, Generic, TypeVar

from arcengine import ARCBaseGame, BlockingMode, Camera, GameAction, InteractionMode, Level, Sprite

from .grid import BACKGROUND, BLACK, CURSOR, END, FLAGGED, FRAME_SIZE, PATH, SOLVED_PATH, START, Frame, Grid, edge
from .truth import BoardTruth, Cell, Edge, Node

# The type of a game's parsed, immutable markers of one level.
M = TypeVar("M")

# The keys Board.from_config reads; games build CONFIG_KEYS from these plus their marker keys (tw09 and tw10
# accept only a subset of these, without "starts" and "breakpoints").
STRUCTURAL_KEYS: Final[frozenset[str]] = frozenset({"cols", "rows", "start", "starts", "end", "breakpoints"})

# The smallest cell size, in pixels, that the marker glyphs of the games calling `require_cell_size` fit in; it
# limits their boards to at most 5 cells per side.
GLYPH_MIN_CELL_SIZE: Final = 9

# (dc, dr) of the four moves: up, down, left, right.
_MOVES: Final[Mapping[GameAction, tuple[int, int]]] = MappingProxyType(
    {GameAction.ACTION1: (0, -1), GameAction.ACTION2: (0, 1), GameAction.ACTION3: (-1, 0),
     GameAction.ACTION4: (1, 0)})

_CELL_KEY: Final = re.compile(r"(-?[0-9]+),(-?[0-9]+)")

# Names of the two sprites of every level.
_BACKGROUND_SPRITE: Final = "grid_bg"
_CURSOR_SPRITE: Final = "cursor"


# --------------------------------------------------------------------------------------------- config helpers
def load_level_configs(game_id: str, package: str = "witness_env") -> tuple[dict[str, Any], ...]:
    """The level configs of `<package>/levels/<game_id>.json`, in play order.

    Raises FileNotFoundError if the file is missing and ValueError unless it holds a non-empty JSON array of
    JSON objects.
    """
    text = (importlib.resources.files(package) / "levels" / f"{game_id}.json").read_text(encoding="utf-8")
    try:
        configs = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{game_id}.json: not valid JSON: {exc}") from exc
    if not (isinstance(configs, list) and configs and all(isinstance(c, dict) for c in configs)):
        raise ValueError(f"{game_id}.json: expected a non-empty JSON array of level objects")
    return tuple(configs)


def parse_coord(value: object, label: str) -> tuple[int, int]:
    """A `[col, row]` list of two ints as a tuple; ValueError starting with `label` otherwise."""
    if isinstance(value, list) and len(value) == 2 and all(type(v) is int for v in value):
        return (value[0], value[1])
    raise ValueError(f"{label}: expected [col, row] with two ints, got {value!r}")


def parse_coords(value: object, label: str) -> tuple[tuple[int, int], ...]:
    """A list of `[col, row]` pairs as a tuple of tuples, order kept (may be empty)."""
    if not isinstance(value, list):
        raise ValueError(f"{label}: expected a list of [col, row] pairs, got {value!r}")
    return tuple(parse_coord(v, label) for v in value)


def parse_cell_key(key: str, label: str) -> Cell:
    """A cell key `"c,r"` (two int literals separated by one comma) as `(c, r)`."""
    match = _CELL_KEY.fullmatch(key) if isinstance(key, str) else None
    if match is None:
        raise ValueError(f"{label}: expected a cell key 'c,r', got {key!r}")
    return (int(match[1]), int(match[2]))


def parse_cell_map(value: object, label: str) -> dict[Cell, Any]:
    """A JSON object with `"c,r"` keys as `{(c, r): value}` in insertion order; values are not checked."""
    if not isinstance(value, dict):
        raise ValueError(f"{label}: expected an object with 'c,r' keys, got {value!r}")
    cells: dict[Cell, Any] = {}
    for key, v in value.items():
        cell = parse_cell_key(key, label)
        if cell in cells:
            raise ValueError(f"{label}: cell {cell[0]},{cell[1]} is listed twice")
        cells[cell] = v
    return cells


def parse_int_map(value: object, label: str, allowed: Set[int] | None = None) -> dict[Cell, int]:
    """`parse_cell_map` whose values must be ints (booleans rejected) and, if given, members of `allowed`."""
    cells = parse_cell_map(value, label)
    for (c, r), v in cells.items():
        if type(v) is not int:
            raise ValueError(f"{label}: cell {c},{r} has value {v!r}, expected an int")
        if allowed is not None and v not in allowed:
            raise ValueError(f"{label}: cell {c},{r} has value {v!r}, expected one of {sorted(allowed)}")
    return cells


def require_cell_size(board: Board, minimum: int) -> None:
    """ValueError unless the board's cells are at least `minimum` pixels wide."""
    size = board.grid.cell_size
    if size < minimum:
        raise ValueError(f"board {board.cols}x{board.rows}: cell size {size} is below {minimum}")


# -------------------------------------------------------------------------------------------------- the board
@dataclass(frozen=True)
class Board:
    """The structural part of one level: grid geometry, starts (level-file order), end and broken edges."""
    grid: Grid
    starts: tuple[Node, ...]
    end: Node
    breakpoints: frozenset[Edge]

    @property
    def cols(self) -> int:
        """Board width in cells."""
        return self.grid.cols

    @property
    def rows(self) -> int:
        """Board height in cells."""
        return self.grid.rows

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> Board:
        """Parse `cols`, `rows`, exactly one of `start` (a node) / `starts` (distinct nodes), `end` and the
        optional `breakpoints` (a list of lattice-adjacent node pairs). Every node must lie on the board. Other
        keys are ignored."""
        grid = Grid(config["cols"], config["rows"])
        if ("start" in config) == ("starts" in config):
            raise ValueError("expected exactly one of 'start' and 'starts'")
        if "start" in config:
            starts: tuple[Node, ...] = (parse_coord(config["start"], "start"),)
            _require_on_board(grid, "start", starts)
        else:
            starts = parse_coords(config["starts"], "starts")
            if not starts or len(set(starts)) != len(starts):
                raise ValueError(f"starts: expected a non-empty list of distinct nodes, got {config['starts']!r}")
            _require_on_board(grid, "starts", starts)
        end = parse_coord(config["end"], "end")
        _require_on_board(grid, "end", (end,))
        pairs = config.get("breakpoints", [])
        if not isinstance(pairs, list):
            raise ValueError(f"breakpoints: expected a list of node pairs, got {pairs!r}")
        breakpoints = frozenset(_parse_broken_edge(grid, pair) for pair in pairs)
        return cls(grid, starts, end, breakpoints)


def _require_on_board(grid: Grid, label: str, nodes: Sequence[Node]) -> None:
    for c, r in nodes:
        if not (0 <= c <= grid.cols and 0 <= r <= grid.rows):
            raise ValueError(f"{label}: node ({c}, {r}) is off the {grid.cols}x{grid.rows} board")


def _parse_broken_edge(grid: Grid, pair: object) -> Edge:
    nodes = parse_coords(pair, "breakpoints")
    if len(nodes) != 2 or abs(nodes[0][0] - nodes[1][0]) + abs(nodes[0][1] - nodes[1][1]) != 1:
        raise ValueError(f"breakpoints: expected two lattice-adjacent nodes, got {pair!r}")
    _require_on_board(grid, "breakpoints", nodes)
    return edge(nodes[0], nodes[1])


@dataclass(frozen=True)
class _Layout(Generic[M]):
    """What stays the same while one level is played: its board, its parsed markers and its entry snapshot."""
    board: Board
    markers: M
    truth: BoardTruth


def _background_sprite(frame: Frame) -> Sprite:
    return Sprite(pixels=frame, name=_BACKGROUND_SPRITE, x=0, y=0, layer=-10, blocking=BlockingMode.NOT_BLOCKED,
                  interaction=InteractionMode.INTANGIBLE)


def _cursor_sprite(xy: tuple[int, int]) -> Sprite:
    return Sprite(pixels=[[CURSOR]], name=_CURSOR_SPRITE, x=xy[0], y=xy[1], layer=10,
                  blocking=BlockingMode.NOT_BLOCKED, interaction=InteractionMode.TANGIBLE)


# --------------------------------------------------------------------------------------------------- the game
class WitnessGame(ARCBaseGame, Generic[M]):
    """A grid line-puzzle game: draw a path from a start node to the end node that satisfies the level's rule.

    Actions: ACTION1-ACTION4 move the path tip up, down, left, right; ACTION5 (CONFIRM) checks the rule;
    ACTION6 and ACTION7 are counted no-ops. A move extends the path, or retracts it when it goes back onto the
    previous node; a move onto an older path node leaves the path unchanged; a move off the board or across a
    broken edge is blocked and leaves the frame unchanged. On a level with several starts, a move that is blocked
    while the path is a single node is made from the first listed start (level-file order) that can make it.
    A CONFIRM with the tip on the end node and the rule satisfied paints the path green and advances to the next
    level; any other CONFIRM resets the path to the first start. Level resets, full resets and the WIN state
    follow arcengine's rules.

    Flagged cells (only in games that override `_flagged_cells`): a CONFIRM rejected with the tip on the end node
    also flags the cells `_flagged_cells` returns that lie on the board. The frame draws a red ring on the
    outermost pixels of each flagged cell, and `truth().flagged_cells` reports them until the next action, which
    clears them; with no further action the snapshot keeps them. A move right after the flagging CONFIRM redraws
    the frame without the rings, also when it is blocked. Any other action (ACTION6, ACTION7, ...) clears the
    flags without redrawing, so the rings stay in the frame, though not in the snapshot, until the next redraw (a
    move that is not blocked, a CONFIRM, a RESET or a level change). RESET and every level change clear the flags;
    a CONFIRM away from the end node flags nothing.

    Visit record: the base keeps the first start and every node the path has covered since the level was entered.
    A failed CONFIRM keeps it; a level reset or a level change restarts it. It is passed to `_flagged_cells`,
    `_draw_reveals` and `_truth_reveals`.

    Class attributes:
        GAME_ID: The game id; also the name of the game's module and level file.
        CONFIG_KEYS: Every key a level config of the game may contain; any other key fails construction.
        CURSOR_SPRITE_FOLLOWS_TIP: Move the 1x1 cursor sprite to the path tip after every redraw.
        LEVELS_PACKAGE: The package whose `levels/<GAME_ID>.json` holds the game's level configs.

    Hooks: `_parse_markers`, `_truth_markers`, `_draw_markers` and `_wins` are pure functions of their
    arguments (they may be static methods); `_move_target` may be overridden to change how moves land.
    `_flagged_cells`, `_draw_reveals` and `_truth_reveals` (pure as well) flag nothing, draw nothing and leave
    the snapshot unchanged unless a game overrides them.
    """
    GAME_ID: ClassVar[str]
    CONFIG_KEYS: ClassVar[frozenset[str]]
    CURSOR_SPRITE_FOLLOWS_TIP: ClassVar[bool] = False
    LEVELS_PACKAGE: ClassVar[str] = "witness_env"

    _layouts: tuple[_Layout[M], ...]
    _board: Board
    _markers: M
    _path: list[Node]
    _flagged: frozenset[Cell]
    _seen: set[Node]

    def __init__(self) -> None:
        """Build every level of the game from its level file. Raises RuntimeError if the environment variable
        ONLY_RESET_LEVELS is "true", which would make every RESET before a win a level reset. arcengine reads the
        variable again at every RESET, so it must stay unset while the game is in use; setting it after
        construction is not detected."""
        if os.environ.get("ONLY_RESET_LEVELS") == "true":
            raise RuntimeError("ONLY_RESET_LEVELS=true disables the engine's full reset; unset it")
        configs = self._level_configs()
        if not configs:
            raise ValueError(f"{self.GAME_ID}: no levels")
        layouts: list[_Layout[M]] = []
        levels: list[Level] = []
        for i, config in enumerate(configs):
            try:
                board, markers = self._parse_level(config)
            except KeyError as exc:
                raise ValueError(f"{self.GAME_ID} level {i + 1}: missing key {exc}") from exc
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{self.GAME_ID} level {i + 1}: {exc}") from exc
            start = board.starts[0]
            entry = BoardTruth(cols=board.cols, rows=board.rows, path=(start,), start=start, starts=board.starts,
                               end=board.end, breakpoints=board.breakpoints)
            layouts.append(_Layout(board, markers, self._truth_markers(entry, markers)))
            levels.append(Level(sprites=[_background_sprite(self._compose_entry(board, markers)),
                                         _cursor_sprite(board.grid.node_xy(start))],
                                grid_size=(FRAME_SIZE, FRAME_SIZE), name=f"Level {i + 1}"))
        self._layouts = tuple(layouts)
        super().__init__(game_id=self.GAME_ID, levels=levels, camera=Camera(background=BACKGROUND, letter_box=BLACK),
                         win_score=len(levels), available_actions=[1, 2, 3, 4, 5])

    def truth(self) -> BoardTruth:
        """A fresh snapshot of the current level: its entry snapshot with the current path, start and flagged
        cells, completed by `_truth_reveals`."""
        truth = dataclasses.replace(self._layouts[self.level_index].truth, path=tuple(self._path),
                                    start=self._path[0], flagged_cells=self._flagged)
        return self._truth_reveals(truth, self._markers, frozenset(self._seen))

    # -------------------------------------------------------------------------------------------------- hooks
    def _level_configs(self) -> Sequence[Mapping[str, Any]]:
        """The level configs of the game, in play order: the game's level file in `LEVELS_PACKAGE`."""
        return load_level_configs(self.GAME_ID, self.LEVELS_PACKAGE)

    @abstractmethod
    def _parse_markers(self, config: Mapping[str, Any], board: Board) -> M:
        """The game's immutable markers of one level config (KeyError for a missing key, ValueError otherwise)."""

    @abstractmethod
    def _truth_markers(self, truth: BoardTruth, markers: M) -> BoardTruth:
        """`truth` with the game's marker fields filled in from `markers`."""

    @abstractmethod
    def _draw_markers(self, frame: Frame, board: Board, markers: M, visited: Set[Node]) -> None:
        """Draw the markers on `frame`; `visited` holds the path's nodes (empty for a level's entry frame)."""

    @abstractmethod
    def _wins(self, board: Board, markers: M, path: tuple[Node, ...]) -> bool:
        """Whether `path`, which ends on the end node, satisfies the level's rule."""

    def _move_target(self, board: Board, node: Node, dc: int, dr: int) -> Node | None:
        """The node a move by (dc, dr) from `node` reaches, or None if the move is blocked."""
        target = (node[0] + dc, node[1] + dr)
        if not (0 <= target[0] <= board.cols and 0 <= target[1] <= board.rows):
            return None
        if edge(node, target) in board.breakpoints:
            return None
        return target

    def _flagged_cells(self, board: Board, markers: M, path: tuple[Node, ...], seen: Set[Node]) -> Set[Cell]:
        """The cells a rejected CONFIRM of `path`, which ends on the end node, flags; `seen` is the visit record.
        No cells by default."""
        return frozenset()

    def _draw_reveals(self, frame: Frame, board: Board, markers: M, seen: Set[Node]) -> None:
        """Draw what depends on the visit record `seen`, after the broken edges, so over their gaps (for a level's
        entry frame `seen` holds the first start only). Nothing by default."""

    def _truth_reveals(self, truth: BoardTruth, markers: M, seen: Set[Node]) -> BoardTruth:
        """`truth` with the fields that depend on the visit record `seen`. Unchanged by default."""
        return truth

    # ---------------------------------------------------------------------------------------- arcengine hooks
    def on_set_level(self, level: Level) -> None:
        """Enter the current level with the path at its first start, no flagged cells and a visit record holding
        that start (the level's sprites show its entry frame)."""
        layout = self._layouts[self.level_index]
        self._board, self._markers = layout.board, layout.markers
        self._path = [layout.board.starts[0]]
        self._flagged = frozenset()
        self._seen = {layout.board.starts[0]}

    def step(self) -> None:
        """Perform the current action and complete it in this one engine step: clear the flagged cells, then
        ACTION5 checks the path, ACTION1-ACTION4 move it (a blocked move redraws only to remove the rings of a
        flagging CONFIRM right before it), every other action does nothing."""
        action = self.action.id
        flagged, self._flagged = self._flagged, frozenset()
        if action == GameAction.ACTION5:
            self._confirm()
        elif action in _MOVES and not self._move(*_MOVES[action]) and flagged:
            self._redraw(PATH)
        self.complete_action()

    # ---------------------------------------------------------------------------------------------- internals
    def _parse_level(self, config: Mapping[str, Any]) -> tuple[Board, M]:
        """The board and the markers of one level config, after rejecting the smallest unknown key."""
        unknown = set(config) - self.CONFIG_KEYS
        if unknown:
            raise ValueError(f"unknown config key {min(unknown)!r}")
        board = Board.from_config(config)
        return board, self._parse_markers(config, board)

    def _move(self, dc: int, dr: int) -> bool:
        """Move the tip by (dc, dr), add the path's nodes to the visit record and redraw; False, with nothing
        changed, if the move is blocked."""
        target = self._move_target(self._board, self._path[-1], dc, dr)
        if target is None:
            alt = self._fallback_start(dc, dr)
            if alt is None:
                return False
            self._path = [alt]
            target = self._move_target(self._board, alt, dc, dr)
        path = self._path
        if target in path[-2:-1]:  # onto the node before the tip: retract
            path.pop()
        elif target not in path:  # onto a free node: extend; an older path node leaves the path as it is
            path.append(target)
        self._seen.update(path)
        self._redraw(PATH)
        return True

    def _fallback_start(self, dc: int, dr: int) -> Node | None:
        """On a one-node path of a multi-start level: the first start that can make the move, else None."""
        if len(self._path) != 1 or len(self._board.starts) < 2:
            return None
        return next((s for s in self._board.starts if self._move_target(self._board, s, dc, dr) is not None), None)

    def _confirm(self) -> None:
        path = tuple(self._path)
        if path[-1] == self._board.end and self._wins(self._board, self._markers, path):
            self._redraw(SOLVED_PATH)
            self.next_level()
        else:
            if path[-1] == self._board.end:
                board = self._board
                cells = self._flagged_cells(board, self._markers, path, frozenset(self._seen))
                self._flagged = frozenset(c for c in cells if 0 <= c[0] < board.cols and 0 <= c[1] < board.rows)
            self._path = [self._board.starts[0]]
            self._redraw(PATH)

    def _redraw(self, colour: int) -> None:
        """Replace the level's background sprite with the current path drawn in `colour` and a ring on every
        flagged cell."""
        level = self.current_level
        (background,) = level.get_sprites_by_name(_BACKGROUND_SPRITE)
        level.remove_sprite(background)
        frame = self._compose(self._board, self._markers, tuple(self._path), colour, frozenset(self._seen))
        for cell in sorted(self._flagged):
            self._board.grid.draw_cell_ring(frame, cell, FLAGGED)
        level.add_sprite(_background_sprite(frame))
        if self.CURSOR_SPRITE_FOLLOWS_TIP:
            (cursor,) = level.get_sprites_by_name(_CURSOR_SPRITE)
            cursor.set_position(*self._board.grid.node_xy(self._path[-1]))

    def _compose_entry(self, board: Board, markers: M) -> Frame:
        """The frame of a level before its first redraw: starts, end, markers, broken edges, and what depends on a
        visit record holding only the first start."""
        return self._compose_board(board, markers, frozenset(), frozenset(board.starts[:1]))

    def _compose(self, board: Board, markers: M, path: tuple[Node, ...], colour: int, seen: Set[Node]) -> Frame:
        """The frame of a drawn path: starts, end, markers, broken edges, what depends on the visit record `seen`,
        the path's unit steps in `colour`, the cursor dot at the tip, and the starts again on top."""
        grid = board.grid
        frame = self._compose_board(board, markers, frozenset(path), seen)
        for p, q in zip(path, path[1:]):
            if abs(p[0] - q[0]) + abs(p[1] - q[1]) == 1:
                grid.draw_edge(frame, p, q, colour)
        grid.draw_dot(frame, path[-1], CURSOR)
        for s in board.starts:
            grid.draw_dot(frame, s, START)
        return frame

    def _compose_board(self, board: Board, markers: M, visited: Set[Node], seen: Set[Node]) -> Frame:
        """A new frame with the starts, the end, the markers (drawn for the `visited` nodes), the broken edges, and
        what depends on the visit record `seen`, in this order."""
        grid = board.grid
        frame = grid.new_frame()
        for s in board.starts:
            grid.draw_dot(frame, s, START)
        grid.draw_dot(frame, board.end, END)
        self._draw_markers(frame, board, markers, visited)
        for e in sorted(board.breakpoints):
            grid.draw_gap(frame, e)
        self._draw_reveals(frame, board, markers, seen)
        return frame
