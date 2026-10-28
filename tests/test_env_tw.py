"""The twelve tw games (`witness_env.rules`, `witness_env.games.tw*`): level files, config checks, frames, rules.

Golden fixtures used: `env/entry_frames.json` (entry frame of every level), `env/traces/tw*.json` (scripted
action sequences with the engine's answer to every action; the validation games' sequences never complete a
level), `env/solutions.json` (a solving action sequence per training level) and `env/path_counts.json` (for
every training level small enough to enumerate: its end-reaching paths and how many of them CONFIRM accepts).
Frames are compared through the first 16 hex digits of the sha1 of their int8 bytes. Rule cases on small boards
are derived by hand.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import random
from types import MappingProxyType

import numpy as np
import pytest
from arcengine import ActionInput, FrameData, GameAction

import golden
from golden import sha16
from witness_env import GAME_IDS, WitnessGame, make_game
from witness_env.base import STRUCTURAL_KEYS, Board, load_level_configs
from witness_env.games.tw01 import Tw01
from witness_env.games.tw02 import Tw02
from witness_env.games.tw03 import Tw03
from witness_env.games.tw05 import Tw05
from witness_env.games.tw06 import Tw06
from witness_env.games.tw07 import Tw07
from witness_env.games.tw08 import Tw08
from witness_env.games.tw09 import Tw09
from witness_env.games.tw10 import Tw10
from witness_env.games.tw11 import Tw11
from witness_env.games.tw12 import Tw12
from witness_env.games.tw13 import Tw13
from witness_env.grid import MAROON, PATH, PURPLE, START, path_edges
from witness_env.rules import (TwGame, TwMarkers, can_tile, orientations, region_errors, squares_errors, stars_errors,
                               tiling_error, triangles_errors)
from witness_env.truth import Polyomino

TW_GAMES = tuple(g for g in GAME_IDS if g.startswith("tw"))
TRAINING_TW_GAMES = tuple(g for g in golden.TRAINING_GAMES if g.startswith("tw"))
CLASSES = {cls.GAME_ID: cls for cls in (Tw01, Tw02, Tw03, Tw05, Tw06, Tw07, Tw08, Tw09, Tw10, Tw11, Tw12, Tw13)}
LEVEL_COUNTS = golden.LEVEL_COUNTS
# md5 of json.dumps(configs, sort_keys=True, separators=(",", ":")) of each level file.
CONTENT_MD5 = {
    "tw01": "81bd467a17cd0d7842e11d40ca5b772a", "tw02": "6c61b57c23bb2119ed6c6d51c9045031",
    "tw03": "1b38c7fc4ae4a6f246ee55251e4058a5", "tw05": "4e938ce15a60dfd49cd7f18560c32dff",
    "tw06": "d9c7bd770e24fd422dbbe2c6a816d53f", "tw07": "949097cfb409e729480e60d006bc3f5a",
    "tw08": "8a494d1867060fdf36bcfedd8257eab0", "tw09": "74f20fb03d5c6e79beb95e781fea698b",
    "tw10": "76bd614a0bd0dbed86b51a8cea1c40fc", "tw11": "7fbd0339ee838f8fce01a363bfde08e5",
    "tw12": "d4686b479df30c8dd8ec58d6a00ab42b", "tw13": "b7695263f077d249e5780bf93388daf7",
}
RESET, UP, LEFT, RIGHT, CONFIRM = 0, 1, 3, 4, 5


def act(game: WitnessGame, action: int) -> FrameData:
    return game.perform_action(ActionInput(id=GameAction.from_id(action)))


def entry_frame(game_id: str, level: int) -> np.ndarray:
    """The frame of `level` (0-based) of a fresh game right after entering it."""
    game = make_game(game_id)
    game.set_level(level)
    return np.asarray(game.camera.render(game.current_level.get_sprites()), dtype=np.int8)


# --------------------------------------------------------------------------------------- level files, configs
@pytest.mark.parametrize("game_id", TW_GAMES)
def test_each_tw_game_is_registered_with_its_keys_and_level_count(game_id):
    game = make_game(game_id)
    cls = type(game)
    assert cls is CLASSES[game_id] and issubclass(cls, TwGame)
    assert game.game_id == cls.GAME_ID == game_id
    markers, required = golden.MARKER_KEYS[game_id], golden.REQUIRED_MARKER_KEYS[game_id]
    structural = {"cols", "rows", "start", "end"} if game_id in ("tw09", "tw10") else STRUCTURAL_KEYS
    assert cls.CONFIG_KEYS == structural | markers
    assert cls.REQUIRED == required
    assert act(game, RESET).win_levels == LEVEL_COUNTS[game_id]


@pytest.mark.parametrize("game_id", TW_GAMES)
def test_level_file_content_digest(game_id):
    configs = load_level_configs(game_id)
    assert len(configs) == LEVEL_COUNTS[game_id]
    text = json.dumps(list(configs), sort_keys=True, separators=(",", ":"))
    assert hashlib.md5(text.encode("utf-8")).hexdigest() == CONTENT_MD5[game_id]


def _first_level_class(game_id: str, mutate) -> type:
    """A subclass of the game whose only level is its first level config after `mutate(config)`."""
    config = json.loads(json.dumps(load_level_configs(game_id)[0]))
    mutate(config)
    return type(f"_Mutated{game_id}", (CLASSES[game_id],), {"_level_configs": lambda self: [config]})


def _set(key, value):
    return lambda config: config.__setitem__(key, value)


def _delete(key):
    return lambda config: config.pop(key)


def _set_piece(field, value):
    return lambda config: config["tetris"]["0,1"].__setitem__(field, value)


CONFIG_ERRORS = [   # (id, game, mutation of its first level config, error message)
    *[(f"{g}-unknown-key", g, _set("foo", 1), f"{g} level 1: unknown config key 'foo'") for g in TW_GAMES],
    ("tw09-starts-key", "tw09", _set("starts", [[0, 0]]), "tw09 level 1: unknown config key 'starts'"),
    ("tw10-breakpoints-key", "tw10", _set("breakpoints", []), "tw10 level 1: unknown config key 'breakpoints'"),
    *[(f"{g}-missing-{key}", g, _delete(key), f"{g} level 1: missing key '{key}'")
      for g in TW_GAMES for key in sorted(golden.REQUIRED_MARKER_KEYS[g])],
    ("tw03-negative-piece", "tw03", _set_piece("negative", True),
     "tw03 level 1: tetris: '0,1': negative polyomino pieces are not supported"),
    ("tw03-repeated-shape-cell", "tw03", _set_piece("shape", [[0, 0], [0, 0]]),
     "tw03 level 1: tetris: '0,1' shape: expected a non-empty list of distinct [x, y] pairs, got [[0, 0], [0, 0]]"),
    ("tw03-empty-shape", "tw03", _set_piece("shape", []),
     "tw03 level 1: tetris: '0,1' shape: expected a non-empty list of distinct [x, y] pairs, got []"),
    ("tw03-unknown-piece-key", "tw03", _set_piece("flip", True), "tw03 level 1: tetris: '0,1': unknown key 'flip'"),
    ("tw03-non-boolean-rotated", "tw03", _set_piece("rotated", 1),
     "tw03 level 1: tetris: '0,1': 'rotated' must be true or false, got 1"),
    ("tw03-piece-without-shape", "tw03", lambda c: c["tetris"]["0,1"].pop("shape"),
     "tw03 level 1: tetris: '0,1': missing 'shape'"),
    ("tw03-piece-not-an-object", "tw03", _set("tetris", {"0,1": [[0, 0]]}),
     "tw03 level 1: tetris: '0,1': expected an object with a 'shape', got [[0, 0]]"),
    ("tw06-triangle-count-4", "tw06", _set("triangles", {"0,0": 4}),
     "tw06 level 1: triangles: cell 0,0 has value 4, expected one of [1, 2, 3]"),
    ("tw06-triangle-count-0", "tw06", _set("triangles", {"0,0": 0}),
     "tw06 level 1: triangles: cell 0,0 has value 0, expected one of [1, 2, 3]"),
    ("tw02-boolean-colour", "tw02", _set("squares", {"0,0": True}),
     "tw02 level 1: squares: cell 0,0 has value True, expected an int"),
    ("tw02-malformed-cell-key", "tw02", _set("squares", {"1,2,3": 6}),
     "tw02 level 1: squares: expected a cell key 'c,r', got '1,2,3'"),
]


@pytest.mark.parametrize("game_id, mutate, message", [c[1:] for c in CONFIG_ERRORS],
                         ids=[c[0] for c in CONFIG_ERRORS])
def test_malformed_level_config_fails_construction_with_its_message(game_id, mutate, message):
    game_class = _first_level_class(game_id, mutate)
    with pytest.raises(ValueError) as exc:
        game_class()
    assert str(exc.value) == message


@pytest.mark.parametrize("game_id", TW_GAMES)
def test_unmodified_first_level_config_constructs(game_id):
    assert _first_level_class(game_id, lambda config: None)().win_score == 1


def test_marker_parsing_reads_only_its_own_keys():
    config = {"cols": 1, "rows": 1, "start": [0, 1], "end": [1, 0], "dots": [[1, 1]], "squares": {"0,0": 6}}
    board = Board.from_config(config)
    assert Tw01._parse_markers(config, board).squares == {}
    assert Tw02._parse_markers(config, board).dots == ()


def test_markers_keep_read_only_copies_of_their_mappings():
    sources = {"squares": {(0, 0): 6}, "stars": {(0, 0): 10}, "triangles": {(0, 0): 1},
               "polyominoes": {(0, 0): Polyomino(((0, 0),), False)}, "filters": {(0, 0): 10}}
    markers = TwMarkers(**sources)
    for name, source in sources.items():
        before = dict(source)
        source[(1, 0)] = source[(0, 0)]                                  # changing the source changes nothing
        assert type(getattr(markers, name)) is MappingProxyType and getattr(markers, name) == before, name


# ----------------------------------------------------------------------------------------------------- frames
@pytest.mark.parametrize("game_id", TW_GAMES)
def test_entry_frames_match_the_golden_fixture_of_entry_frames(game_id):
    expected = golden.STORE.entry_frames()[game_id]
    assert len(expected) == LEVEL_COUNTS[game_id]
    assert [sha16(entry_frame(game_id, level)) for level in range(len(expected))] == expected


def test_tw09_first_redraw_repaints_the_first_start_over_its_wrap_marker():
    left_arm_of_start = [(4, 3), (4, 3), (12, 4), (5, 4), (9, 3)]     # (y, x) per level
    for level, (y, x) in enumerate(left_arm_of_start):
        game = make_game("tw09")
        game.set_level(level)
        entry = np.asarray(game.camera.render(game.current_level.get_sprites()), dtype=np.int8)
        (frame,) = act(game, CONFIRM).frame
        redraw = np.asarray(frame, dtype=np.int8)
        assert np.argwhere(redraw != entry).tolist() == [[y, x]], level
        assert (entry[y, x], redraw[y, x]) == (PURPLE, START), level


@pytest.mark.parametrize("game_id", [g for g in TW_GAMES if g != "tw09"])
def test_failed_confirm_at_the_start_re_renders_the_entry_frame(game_id):
    expected = golden.STORE.entry_frames()[game_id]
    for level, digest in enumerate(expected):
        game = make_game(game_id)
        game.set_level(level)
        fd = act(game, CONFIRM)
        assert [sha16(f) for f in fd.frame] == [digest], level
        assert game.truth().path == (game.truth().starts[0],)
        assert (fd.levels_completed, game.level_index) == (0, level)


def test_tw10_framed_cell_has_a_maroon_ring_around_its_square():
    frame = entry_frame("tw10", 0)
    config = load_level_configs("tw10")[0]
    cx, cy = Board.from_config(config).grid.cell_center((0, 0))
    for d in range(-3, 4):
        for x, y in ((cx + d, cy - 3), (cx + d, cy + 3), (cx - 3, cy + d), (cx + 3, cy + d)):
            assert frame[y, x] == MAROON, (x, y)
    assert (frame[cy - 1:cy + 2, cx - 1:cx + 2] == config["squares"]["0,0"]).all()          # the painted color


def test_tw09_moves_wrap_between_the_edge_columns_and_wrap_steps_are_not_drawn():
    game = make_game("tw09")
    act(game, RESET)
    grid = Board.from_config(load_level_configs("tw09")[0]).grid
    (x0, y0), (x3, _) = grid.node_xy((0, 0)), grid.node_xy((3, 0))

    # A dot (start or cursor) covers its node's pixel and one pixel on either side of it along the row.
    (after_wrap,) = act(game, LEFT).frame
    assert game.truth().path == ((0, 0), (3, 0))
    between_start_and_cursor = slice(x0 + 2, x3 - 1)                   # the cursor dot is on (3, 0)
    assert not (np.asarray(after_wrap)[y0, between_start_and_cursor] == PATH).any()
    (after_unit,) = act(game, LEFT).frame
    assert game.truth().path == ((0, 0), (3, 0), (2, 0))
    (x2, _) = grid.node_xy((2, 0))
    right_of_cursor_through_node_3 = slice(x2 + 2, x3 + 1)             # the cursor dot is on (2, 0)
    assert (np.asarray(after_unit)[y0, right_of_cursor_through_node_3] == PATH).all()
    act(game, RIGHT)
    assert game.truth().path == ((0, 0), (3, 0))
    (back,) = act(game, RIGHT).frame
    assert game.truth().path == ((0, 0),)
    (blocked,) = act(game, UP).frame
    assert game.truth().path == ((0, 0),)
    assert sha16(blocked) == sha16(back)


# ----------------------------------------------------------------------------------------- scripted sequences
@pytest.mark.parametrize("game_id", TW_GAMES)
def test_scripted_actions_match_the_golden_fixture_of_env_traces(game_id):
    trace = golden.STORE.env_trace(game_id)
    assert trace.solution_free == (game_id not in golden.TRAINING_GAMES)
    game = make_game(game_id)
    for i, step in enumerate(trace.steps):
        fd = act(game, step.action)
        truth = game.truth()
        got = (tuple(sha16(f) for f in fd.frame), fd.levels_completed, fd.state.name, game.level_index, truth.path,
               truth.start)
        assert got == (step.frames, step.levels_completed, step.state, step.level, step.path, step.start), i


@pytest.mark.parametrize("game_id", TRAINING_TW_GAMES)
def test_stored_solutions_played_in_a_row_win_every_level(game_id):
    solutions = golden.STORE.solutions()[game_id]
    assert len(solutions) == LEVEL_COUNTS[game_id]
    game = make_game(game_id)
    assert len(act(game, RESET).frame) == 1
    for level, solution in enumerate(solutions):
        results = [act(game, a) for a in solution]
        final = level == len(solutions) - 1
        assert [len(fd.frame) for fd in results] == [1] * (len(solution) - 1) + [1 if final else 2], level
        assert [fd.levels_completed for fd in results] == [level] * (len(solution) - 1) + [level + 1], level
        assert results[-1].state.name == ("WIN" if final else "NOT_FINISHED")


@pytest.mark.parametrize("game_id", TRAINING_TW_GAMES)
def test_each_stored_solution_wins_its_level_on_its_last_action_only(game_id):
    for level, solution in enumerate(golden.STORE.solutions()[game_id]):
        game = make_game(game_id)
        game.set_level(level)
        completed = [act(game, a).levels_completed for a in solution]
        assert completed == [0] * (len(solution) - 1) + [1], level


@pytest.mark.parametrize("game_id", TRAINING_TW_GAMES)
def test_path_counts_match_the_golden_fixture_of_path_counts(game_id):
    rows = golden.STORE.path_counts()[game_id]
    assert rows
    game = make_game(game_id)
    configs = load_level_configs(game_id)
    for row in rows:
        config = configs[row["level"]]
        board = Board.from_config(config)
        markers = type(game)._parse_markers(config, board)
        assert golden.count_paths(game, board, markers) == (row["paths"], row["accepted"]), row["level"]


# -------------------------------------------------------------------------------------- rules on small boards
# Paths: TOP3 / TOP4 run along the top border of a 3x1 / 4x1 board (one region); CUT separates cell (0, 0)
# from the rest of a 2x1, 3x1 or 4x1 board; SPLIT_2X2 separates cell (0, 1) from the rest of a 2x2 board.
TOP3 = ((0, 0), (1, 0), (2, 0), (3, 0))
TOP4 = ((0, 0), (1, 0), (2, 0), (3, 0), (4, 0))
CUT = ((1, 0), (1, 1))
SPLIT_2X2 = ((0, 1), (1, 1), (1, 2))
I3_ROW = {"shape": [[0, 0], [1, 0], [2, 0]]}
I3_COLUMN = {"shape": [[0, 0], [0, 1], [0, 2]]}
MONO = {"shape": [[0, 0]]}
V_DOMINO = {"shape": [[0, 0], [0, 1]]}

RULE_CASES = [   # (id, game, cols, rows, path, config fragment, expected)
    ("tw02-one-colour-passes", "tw02", 3, 1, TOP3, {"squares": {"0,0": 6, "2,0": 6}}, True),
    ("tw02-two-colours-fail", "tw02", 3, 1, TOP3, {"squares": {"0,0": 6, "2,0": 10}}, False),
    ("tw02-colours-in-separate-regions-pass", "tw02", 2, 1, CUT, {"squares": {"0,0": 6, "1,0": 10}}, True),
    ("tw05-star-pair-passes", "tw05", 3, 1, TOP3, {"stars": {"0,0": 6, "2,0": 6}}, True),
    ("tw05-pair-split-by-path-fails", "tw05", 2, 1, CUT, {"stars": {"0,0": 6, "1,0": 6}}, False),
    ("tw05-three-stars-fail", "tw05", 3, 1, TOP3, {"stars": {"0,0": 6, "1,0": 6, "2,0": 6}}, False),
    ("tw05-two-pairs-pass", "tw05", 4, 1, TOP4, {"stars": {"0,0": 6, "1,0": 10, "2,0": 6, "3,0": 10}}, True),
    ("tw06-one-side-on-path-passes", "tw06", 2, 1, CUT, {"triangles": {"0,0": 1}}, True),
    ("tw06-too-few-sides-fail", "tw06", 2, 1, CUT, {"triangles": {"0,0": 2}}, False),
    ("tw06-three-sides-pass", "tw06", 2, 1, ((0, 1), (0, 0), (1, 0), (1, 1)), {"triangles": {"0,0": 3}}, True),
    ("tw10-frame-colour-matches-passes", "tw10", 3, 1, TOP3,
     {"squares": {"0,0": 6, "2,0": 10}, "filters": {"0,0": 10}}, True),
    ("tw10-frame-colour-differs-fails", "tw10", 3, 1, TOP3,
     {"squares": {"0,0": 6, "2,0": 6}, "filters": {"0,0": 10}}, False),
    ("tw10-frame-on-empty-cell-does-nothing", "tw10", 3, 1, TOP3,
     {"squares": {"0,0": 6, "2,0": 6}, "filters": {"1,0": 10}}, True),
    ("tw01-dot-on-path-passes", "tw01", 2, 1, ((0, 0), (1, 0), (2, 0)), {"dots": [[1, 0]]}, True),
    ("tw01-dot-off-path-fails", "tw01", 2, 1, ((0, 0), (1, 0), (2, 0)), {"dots": [[1, 1]]}, False),
    ("tw01-dot-at-the-start-passes", "tw01", 2, 1, ((0, 0), (1, 0), (2, 0)), {"dots": [[0, 0]]}, True),
    ("tw01-dot-at-the-end-passes", "tw01", 2, 1, ((0, 0), (1, 0), (2, 0)), {"dots": [[2, 0]]}, True),
    ("tw07-eraser-cancels-one-error", "tw07", 3, 1, TOP3, {"squares": {"0,0": 6, "1,0": 10}, "erasers": [[2, 0]]},
     True),
    ("tw07-unused-eraser-fails", "tw07", 3, 1, TOP3, {"squares": {"0,0": 6, "1,0": 6}, "erasers": [[2, 0]]}, False),
    ("tw07-two-errors-one-eraser-fail", "tw07", 4, 1, TOP4,
     {"squares": {"0,0": 6, "1,0": 10, "2,0": 12}, "erasers": [[3, 0]]}, False),
    ("tw07-eraser-in-another-region-fails", "tw07", 3, 1, CUT,
     {"squares": {"1,0": 6, "2,0": 10}, "erasers": [[0, 0]]}, False),
    ("tw07-eraser-in-the-same-region-passes", "tw07", 4, 1, CUT,
     {"squares": {"1,0": 6, "2,0": 10}, "erasers": [[3, 0]]}, True),
    ("tw07-two-erasers-one-error-fail", "tw07", 3, 1, TOP3, {"stars": {"0,0": 6}, "erasers": [[1, 0], [2, 0]]},
     False),
    ("tw07-lone-star-is-one-error", "tw07", 3, 1, TOP3, {"stars": {"0,0": 6}, "erasers": [[2, 0]]}, True),
    ("tw03-row-piece-tiles-row", "tw03", 3, 1, TOP3, {"tetris": {"0,0": I3_ROW}}, True),
    ("tw03-area-mismatch-fails", "tw03", 3, 1, TOP3, {"tetris": {"0,0": {"shape": [[0, 0], [1, 0]]}}}, False),
    ("tw03-fixed-column-piece-fails-on-row", "tw03", 3, 1, TOP3, {"tetris": {"0,0": I3_COLUMN}}, False),
    ("tw03-rotatable-column-piece-tiles-row", "tw03", 3, 1, TOP3, {"tetris": {"0,0": I3_COLUMN | {"rotated": True}}},
     True),
    ("tw03-region-without-pieces-passes", "tw03", 2, 1, CUT, {"tetris": {"0,0": MONO}}, True),
    # SPLIT_2X2 leaves the region {(0, 0), (1, 0), (1, 1)}, tiled only by the monomino on (0, 0) and the domino on
    # (1, 0), (1, 1); it passes whichever of the two pieces is listed first.
    ("tw03-tiling-with-the-monomino-listed-first", "tw03", 2, 2, SPLIT_2X2,
     {"tetris": {"0,0": MONO, "1,1": V_DOMINO}}, True),
    ("tw11-tiling-with-the-monomino-listed-first", "tw11", 2, 2, SPLIT_2X2,
     {"tetris": {"0,0": MONO, "1,1": V_DOMINO}}, True),
    ("tw03-tiling-with-the-monomino-listed-second", "tw03", 2, 2, SPLIT_2X2,
     {"tetris": {"0,0": V_DOMINO, "1,1": MONO}}, True),
    ("tw11-tiling-with-the-monomino-listed-second", "tw11", 2, 2, SPLIT_2X2,
     {"tetris": {"0,0": V_DOMINO, "1,1": MONO}}, True),
    ("tw13-untileable-region-is-one-error", "tw13", 4, 1, TOP4,
     {"tetris": {"0,0": V_DOMINO, "1,0": V_DOMINO}, "erasers": [[3, 0]]}, True),
    ("tw13-eraser-cannot-excuse-a-dot", "tw13", 3, 1, TOP3,
     {"squares": {"0,0": 6, "1,0": 10}, "erasers": [[2, 0]], "dots": [[1, 1]]}, False),
    ("tw13-dot-visited-and-errors-balanced", "tw13", 3, 1, TOP3,
     {"squares": {"0,0": 6, "1,0": 10}, "erasers": [[2, 0]], "dots": [[1, 0]]}, True),
    ("tw12-dot-off-path-fails", "tw12", 3, 1, TOP3, {"dots": [[1, 1]], "stars": {"0,0": 6, "2,0": 6}}, False),
    ("tw12-dot-on-path-and-star-pair-pass", "tw12", 3, 1, TOP3, {"dots": [[1, 0]], "stars": {"0,0": 6, "2,0": 6}},
     True),
    ("tw08-square-and-star-pair-pass", "tw08", 3, 1, TOP3, {"squares": {"0,0": 6}, "stars": {"1,0": 10, "2,0": 10}},
     True),
    ("tw08-star-is-not-a-square", "tw08", 3, 1, TOP3, {"squares": {"0,0": 6}, "stars": {"1,0": 10}}, False),
    ("tw11-triangle-counts-edges-of-the-whole-path", "tw11", 2, 1, ((0, 1), (0, 0), (1, 0)),
     {"triangles": {"0,0": 2}}, True),
]


def test_error_counts_of_the_region_rules():
    region = frozenset({(0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (2, 1)})
    squares = {(0, 0): 6, (1, 0): 10, (2, 0): 12, (5, 5): 8}
    stars = {(0, 0): 6, (1, 0): 6, (2, 0): 6, (0, 1): 6, (1, 1): 10}
    triangles = {(0, 0): 1, (2, 1): 2}
    edges = path_edges(((0, 0), (1, 0)))
    assert squares_errors(region, squares) == 2                        # three colors in the region
    assert squares_errors(region, {(0, 0): 6, (1, 1): 6}) == 0
    assert stars_errors(region, stars) == 3                            # |4 - 2| + |1 - 2|
    assert triangles_errors(region, triangles, edges) == 1             # (2, 1) has no side on the path
    markers = TwMarkers(squares=squares, stars=stars, triangles=triangles)
    assert region_errors(region, markers, edges) == 6


@pytest.mark.parametrize("game_id, cols, rows, path, fragment, expected", [c[1:] for c in RULE_CASES],
                         ids=[c[0] for c in RULE_CASES])
def test_rule_outcome_on_a_small_board(game_id, cols, rows, path, fragment, expected):
    config = fragment | {"cols": cols, "rows": rows, "start": list(path[0]), "end": list(path[-1])}
    cls = CLASSES[game_id]
    board = Board.from_config(config)
    assert cls._wins(board, cls._parse_markers(config, board), path) is expected


# ------------------------------------------------------------------------------------------- polyomino tiling
MONOMINO = Polyomino(((0, 0),), False)
VERTICAL_DOMINO = Polyomino(((0, 0), (0, 1)), False)
HORIZONTAL_DOMINO = Polyomino(((0, 0), (1, 0)), False)
L_SHAPE = ((0, 0), (0, 1), (0, 2), (1, 2))


@pytest.mark.parametrize("region, pieces", [
    ({(0, 0), (1, 0), (1, 1)}, [VERTICAL_DOMINO, MONOMINO]),
    ({(0, 0), (0, 1), (1, 1)}, [HORIZONTAL_DOMINO, MONOMINO]),
], ids=["vertical-domino", "horizontal-domino"])
def test_tiling_result_does_not_depend_on_piece_order(region, pieces):
    assert can_tile(region, pieces) is True
    assert can_tile(region, pieces[::-1]) is True


def test_pieces_rotate_only_when_allowed_and_never_mirror():
    fixed, rotatable = Polyomino(L_SHAPE, False), Polyomino(L_SHAPE, True)
    quarter_turn = {(0, 0), (1, 0), (2, 0), (0, 1)}
    mirrored = {(1, 0), (1, 1), (1, 2), (0, 2)}
    assert can_tile(set(L_SHAPE), [fixed]) is True
    assert can_tile(quarter_turn, [fixed]) is False
    assert can_tile(quarter_turn, [rotatable]) is True
    assert can_tile(mirrored, [rotatable]) is False


def test_orientations_are_the_successive_quarter_turns():
    assert orientations(Polyomino(L_SHAPE, True)) == (
        frozenset({(0, 0), (0, 1), (0, 2), (1, 2)}), frozenset({(0, 1), (1, 1), (2, 0), (2, 1)}),
        frozenset({(0, 0), (1, 0), (1, 1), (1, 2)}), frozenset({(0, 0), (0, 1), (1, 0), (2, 0)}))
    assert orientations(Polyomino(L_SHAPE, False)) == (frozenset(L_SHAPE),)


def test_symmetric_pieces_have_fewer_orientations():
    assert len(orientations(Polyomino(((0, 0), (1, 0)), True))) == 2
    assert len(orientations(Polyomino(((0, 0), (1, 0), (0, 1), (1, 1)), True))) == 1
    assert len(orientations(Polyomino(((0, 0),), True))) == 1


def test_disconnected_piece_tiles_a_matching_cell_set():
    assert can_tile({(1, 1), (4, 2)}, [Polyomino(((0, 0), (3, 1)), False)]) is True


def test_total_piece_area_must_equal_the_region_size():
    assert can_tile({(0, 0), (1, 0), (2, 0)}, [HORIZONTAL_DOMINO]) is False


def test_region_without_pieces_has_no_tiling_error():
    assert tiling_error(frozenset({(0, 0), (1, 0)}), {}) == 0
    assert tiling_error(frozenset({(0, 0), (1, 0)}), {(5, 5): MONOMINO}) == 0


# Shapes of the seeded tiling cases (the last one is disconnected).
CASE_SHAPES = (
    ((0, 0),), ((0, 0), (1, 0)), ((0, 0), (1, 0), (2, 0)), ((0, 0), (0, 1), (1, 1)), L_SHAPE,
    ((0, 0), (1, 0), (2, 0), (1, 1)), ((0, 0), (1, 0), (1, 1), (2, 1)), ((0, 0), (1, 0), (0, 1), (1, 1)),
    ((0, 0), (2, 1)),
)


def _allowed_shapes(piece: Polyomino) -> set[frozenset[tuple[int, int]]]:
    """Every allowed orientation of `piece`, each shifted to the origin (computed independently of `orientations`)."""
    shapes, cells = set(), list(piece.shape)
    for _ in range(4 if piece.rotatable else 1):
        min_x, min_y = min(x for x, _ in cells), min(y for _, y in cells)
        shapes.add(frozenset((x - min_x, y - min_y) for x, y in cells))
        cells = [(y, -x) for x, y in cells]
    return shapes


def _tiles_by_brute_force(region: frozenset, pieces: list[Polyomino]) -> bool:
    """Place the pieces in the given order, each in every orientation at every translation inside the region."""
    if not pieces:
        return not region
    for shape in _allowed_shapes(pieces[0]):
        for ox, oy in {(c - x, r - y) for c, r in region for x, y in shape}:
            placed = frozenset((x + ox, y + oy) for x, y in shape)
            if placed <= region and _tiles_by_brute_force(region - placed, pieces[1:]):
                return True
    return False


def _seeded_case(rng: random.Random) -> tuple[frozenset, frozenset, list[Polyomino]]:
    """1-4 random pieces placed disjointly on a 5x5 area; their union R and R with one cell moved to a cell
    outside R that touches R."""
    region: set[tuple[int, int]] = set()
    pieces = []
    for _ in range(rng.randint(1, 4)):
        piece = Polyomino(rng.choice(CASE_SHAPES), rng.random() < 0.5)
        shape = rng.choice(sorted(_allowed_shapes(piece), key=sorted))
        for _ in range(50):
            ox, oy = rng.randint(0, 4), rng.randint(0, 4)
            placed = {(x + ox, y + oy) for x, y in shape}
            if all(0 <= c <= 4 and 0 <= r <= 4 for c, r in placed) and not placed & region:
                region |= placed
                pieces.append(piece)
                break
    if not pieces:
        return _seeded_case(rng)
    outside = sorted({(c + dc, r + dr) for c, r in region for dc, dr in ((0, 1), (0, -1), (1, 0), (-1, 0))}
                     - region)
    moved = (region - {rng.choice(sorted(region))}) | {rng.choice(outside)}
    return frozenset(region), frozenset(moved), pieces


def test_tiling_agrees_with_brute_force_on_seeded_cases_in_every_piece_order():
    rng = random.Random(0)
    for case in range(500):
        region, moved, pieces = _seeded_case(rng)
        expected_moved = _tiles_by_brute_force(moved, pieces)
        for order in itertools.permutations(pieces):
            assert can_tile(region, list(order)) is True, (case, order)
            assert can_tile(moved, list(order)) is expected_moved, (case, order)
