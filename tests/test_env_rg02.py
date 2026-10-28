"""The r2 training game rg02 (`witness_env.games.rg02`, axles): level file, config checks, frames, reported markers,
rules, stored solutions, and parity with an independent reference implementation of the rule.

The expected values are frozen in this file, not in the golden files: frame digests (`golden.sha16`), the stored
solutions, rule cases on small boards, and per level the digest of the paths the reference implementation accepts.
The reference implementation, which is not part of this repository, has its own region split; it was run on every
simple start-to-end path of every level, and the digests below are its verdicts.
The class is imported directly, so this file runs whether or not rg02 is registered in `TRAIN_R2_IDS`.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest
from arcengine import ActionInput, FrameData, GameAction

import golden
from golden import sha16
from witness_env import TRAIN_R2_IDS
from witness_env.base import Board, load_level_configs
from witness_env.games.rg02 import Rg02
from witness_env.grid import CHARCOAL, CURSOR, START

RESET, UP, DOWN, LEFT, RIGHT, CONFIRM = 0, 1, 2, 3, 4, 5
LEVELS = 10
# Entry frame of every level.
ENTRY_FRAMES = ("53cb23c95d42cd32", "c8e9c523a9161eb3", "25bf67b14a899345", "7c13b44f1f925044", "7ef2f04a376800f3",
                "e35b5ae33f1e5e82", "26386e4412632522", "2a627a6a26503ac8", "89fc85b597000676", "7cbb73617f7003df")
# A shortest solving action sequence per level (breadth-first search over the engine's moves), ending in CONFIRM.
SOLUTIONS = (
    (4, 4, 4, 5),
    (2, 2, 2, 4, 1, 1, 4, 5),
    (2, 4, 1, 4, 4, 4, 5),
    (1, 1, 3, 3, 3, 3, 2, 4, 4, 4, 2, 2, 2, 3, 5),
    (3, 2, 4, 4, 4, 2, 2, 2, 3, 1, 5),
    (1, 1, 3, 3, 3, 2, 2, 2, 3, 5),
    (3, 3, 1, 1, 1, 4, 4, 4, 4, 2, 3, 3, 3, 2, 4, 4, 4, 2, 2, 5),
    (2, 3, 1, 1, 1, 4, 4, 4, 1, 3, 3, 3, 5),
    (4, 2, 2, 3, 1, 3, 3, 3, 1, 1, 4, 4, 4, 1, 3, 3, 5),
    (1, 4, 2, 4, 4, 4, 2, 2, 2, 3, 1, 3, 3, 3, 2, 4, 4, 5),
)
# Per level, from the reference: (simple start-to-end paths, accepted paths, digest of the accepted paths).
REFERENCE_VERDICTS = (
    (112, 1, "5ca137f7729a374d"), (117, 6, "a603520b9d984642"), (5198, 1, "aa2371b6723a19bc"),
    (4272, 35, "dcee96f67c72e3dc"), (4912, 16, "38510efad73278da"), (6672, 45, "e4adf6afc74ae0a9"),
    (6161, 12, "06a6a949abeca9db"), (4912, 16, "1ac7b5c27d2dc2c3"), (4990, 2, "52aeed653db1e21e"),
    (4732, 2, "ee31483c7aef90df"),
)
# The axle glyph around the cell center (row by row, top first): half-turn symmetric, not mirror symmetric.
AXLE_PICTURE = ("..##.",
                "##.##",
                "#...#",
                "##.##",
                ".##..")
# Marker layers drawn for configs that are not levels: (config, sha16 of the layer).
MARKER_LAYERS = (
    ({"cols": 5, "rows": 5, "start": [0, 0], "end": [5, 5], "axles": [[0, 0], [2, 2], [4, 1]]}, "611aec3e9e334400"),
    ({"cols": 3, "rows": 3, "start": [0, 0], "end": [3, 3], "axles": [[1, 1]]}, "9715a4207b7a68d9"),
)


def act(game: Rg02, action: int) -> FrameData:
    return game.perform_action(ActionInput(id=GameAction.from_id(action)))


def entry_frame(level: int) -> np.ndarray:
    game = Rg02()
    game.set_level(level)
    return np.asarray(game.camera.render(game.current_level.get_sprites()), dtype=np.int8)


def simple_paths(board: Board) -> list[tuple[tuple[int, int], ...]]:
    """Every simple path on the lattice from the start to the end that meets the end only at its last node."""
    out: list[tuple[tuple[int, int], ...]] = []
    path = [board.starts[0]]

    def grow() -> None:
        if path[-1] == board.end:
            out.append(tuple(path))
            return
        c, r = path[-1]
        for nxt in ((c, r - 1), (c, r + 1), (c - 1, r), (c + 1, r)):
            if 0 <= nxt[0] <= board.cols and 0 <= nxt[1] <= board.rows and nxt not in path:
                path.append(nxt)
                grow()
                path.pop()

    grow()
    return out


# --------------------------------------------------------------------------------------- level files, configs
def test_reset_reports_the_level_count_and_the_first_entry_frame():
    fd = act(Rg02(), RESET)
    assert fd.win_levels == LEVELS
    assert [sha16(f) for f in fd.frame] == list(ENTRY_FRAMES[:1])


def test_level_file_holds_ten_distinct_compact_levels_with_one_or_two_axles():
    configs = load_level_configs("rg02")
    assert len(configs) == LEVELS == len({json.dumps(c, sort_keys=True) for c in configs})
    for level, config in enumerate(configs):
        assert set(config) == {"cols", "rows", "start", "end", "axles"}, level
        assert 1 <= len(config["axles"]) <= 2, level


def _first_level_class(mutate) -> type:
    config = json.loads(json.dumps(load_level_configs("rg02")[0]))
    mutate(config)
    return type("_MutatedRg02", (Rg02,), {"_level_configs": lambda self: [config]})


def _set(key, value):
    return lambda config: config.__setitem__(key, value)


CONFIG_ERRORS = [   # (id, mutation of the first level config, error message)
    ("missing-key", lambda c: c.pop("axles"), "rg02 level 1: missing key 'axles'"),
    ("misspelled-key", _set("axle", [[1, 2]]), "rg02 level 1: unknown config key 'axle'"),
    ("board-too-large-for-the-glyphs", lambda c: c.update(cols=6, rows=6),
     "rg02 level 1: board 6x6: cell size 8 is below 9"),
    ("boolean-coordinate", _set("axles", [[True, 0]]),
     "rg02 level 1: axles: expected [col, row] with two ints, got [True, 0]"),
    ("no-axle", _set("axles", []), "rg02 level 1: axles: expected distinct cells on the board, got []"),
    ("axle-off-the-board", _set("axles", [[3, 0]]),
     "rg02 level 1: axles: expected distinct cells on the board, got [[3, 0]]"),
    ("axle-listed-twice", _set("axles", [[1, 0], [1, 0]]),
     "rg02 level 1: axles: expected distinct cells on the board, got [[1, 0], [1, 0]]"),
]


@pytest.mark.parametrize("mutate, message", [c[1:] for c in CONFIG_ERRORS], ids=[c[0] for c in CONFIG_ERRORS])
def test_malformed_level_config_fails_construction_with_its_message(mutate, message):
    with pytest.raises(ValueError) as exc:
        _first_level_class(mutate)()
    assert str(exc.value) == message


def test_unmodified_first_level_config_constructs():
    assert _first_level_class(lambda config: None)().win_score == 1


# ----------------------------------------------------------------------------------------------------- frames
def test_entry_frames_match_the_frozen_digests():
    assert [sha16(entry_frame(level)) for level in range(LEVELS)] == list(ENTRY_FRAMES)


def test_failed_confirm_at_the_start_re_renders_the_entry_frame():
    for level, digest in enumerate(ENTRY_FRAMES):
        game = Rg02()
        game.set_level(level)
        fd = act(game, CONFIRM)
        assert [sha16(f) for f in fd.frame] == [digest], level
        assert (fd.levels_completed, game.level_index) == (0, level)


@pytest.mark.parametrize("config, digest", MARKER_LAYERS)
def test_marker_layers_of_non_level_configs_match_the_frozen_digests(config, digest):
    board = Board.from_config(config)
    frame = board.grid.new_frame()
    Rg02._draw_markers(frame, board, Rg02._parse_markers(config, board), frozenset())
    assert sha16(frame) == digest


def test_every_axle_is_drawn_as_the_axle_picture_in_charcoal():
    picture = np.array([[ch == "#" for ch in row] for row in AXLE_PICTURE])
    assert (picture == picture[::-1, ::-1]).all()
    assert not (picture == picture[:, ::-1]).all() and not (picture == picture[::-1, :]).all()
    for config, _ in MARKER_LAYERS:
        board = Board.from_config(config)
        empty = board.grid.new_frame()
        frame = board.grid.new_frame()
        Rg02._draw_markers(frame, board, Rg02._parse_markers(config, board), frozenset())
        expected = np.asarray(empty).copy()
        for cell in config["axles"]:
            x, y = board.grid.cell_center(tuple(cell))
            window = expected[y - 2:y + 3, x - 2:x + 3]
            window[picture] = CHARCOAL
        assert (np.asarray(frame) == expected).all()


def test_cursor_sprite_follows_the_path_tip():
    assert Rg02.CURSOR_SPRITE_FOLLOWS_TIP is True
    game = Rg02()
    act(game, RESET)
    frame = np.asarray(act(game, RIGHT).frame[0])
    truth = game.truth()
    assert len(truth.path) == 2
    grid = Board.from_config(load_level_configs("rg02")[0]).grid
    (tip_x, tip_y), (start_x, start_y) = grid.node_xy(truth.path[-1]), grid.node_xy(truth.start)
    assert (frame[tip_y, tip_x], frame[start_y, start_x]) == (CURSOR, START)


# ------------------------------------------------------------------------------------------- reported markers
def test_truth_of_every_level_declares_the_axles_of_its_level_config():
    game = Rg02()
    for level, config in enumerate(load_level_configs("rg02")):
        game.set_level(level)
        truth = game.truth()
        assert [(m.name, m.glyph, m.values_are_colours) for m in truth.cell_markers] == [("axles", "x", False)], level
        assert dict(truth.cell_markers[0].values) == {(c, r): None for c, r in config["axles"]}, level
        assert (dict(truth.squares), truth.node_markers, truth.dots, dict(truth.stars), dict(truth.triangles),
                dict(truth.polyominoes), truth.erasers, truth.framed_cells) == (
            {}, (), (), {}, {}, {}, frozenset(), frozenset()), level
        assert truth.starts == (tuple(config["start"]),), level


# ------------------------------------------------------------------------------------------ stored solutions
def test_stored_solutions_played_in_a_row_win_every_level_and_enter_the_next():
    game = Rg02()
    act(game, RESET)
    for level, solution in enumerate(SOLUTIONS):
        results = [act(game, a) for a in solution]
        final = level == LEVELS - 1
        assert [len(fd.frame) for fd in results] == [1] * (len(solution) - 1) + [1 if final else 2], level
        assert [fd.levels_completed for fd in results] == [level] * (len(solution) - 1) + [level + 1], level
        assert results[-1].state.name == ("WIN" if final else "NOT_FINISHED")
        if not final:
            assert sha16(results[-1].frame[-1]) == ENTRY_FRAMES[level + 1], level


def test_each_stored_solution_wins_its_level_on_its_last_action_only():
    for level, solution in enumerate(SOLUTIONS):
        game = Rg02()
        game.set_level(level)
        assert [act(game, a).levels_completed for a in solution] == [0] * (len(solution) - 1) + [1], level


def test_no_stored_solution_wins_another_level():
    for level, solution in enumerate(SOLUTIONS):
        for other in range(LEVELS):
            if other == level:
                continue
            game = Rg02()
            game.set_level(other)
            assert all(act(game, a).levels_completed == 0 for a in solution), (level, other)


@pytest.mark.skipif("rg02" not in TRAIN_R2_IDS, reason="rg02 is not registered in TRAIN_R2_IDS yet")
def test_the_golden_solutions_of_rg02_are_the_ones_frozen_here():
    assert [tuple(s) for s in golden.STORE.solutions_r2()["rg02"]] == list(SOLUTIONS)


# ------------------------------------------------------------------------------------------------------ rules
def _wins(config: dict, path) -> bool:
    board = Board.from_config(config)
    return Rg02._wins(board, Rg02._parse_markers(config, board), tuple(tuple(n) for n in path))


_L1 = {"cols": 3, "rows": 3, "start": [0, 1], "end": [3, 1], "axles": [[1, 0]]}
_T = {"cols": 3, "rows": 3, "start": [0, 3], "end": [3, 0], "axles": [[1, 0]]}
_E = {"cols": 4, "rows": 4, "start": [0, 1], "end": [4, 1], "axles": [[1, 0]]}
_F = {"cols": 4, "rows": 4, "start": [0, 4], "end": [4, 0], "axles": [[1, 2]]}
_D = {"cols": 4, "rows": 4, "start": [0, 4], "end": [4, 0], "axles": [[0, 2], [3, 1]]}
_C = {"cols": 3, "rows": 3, "start": [0, 1], "end": [0, 2], "axles": [[1, 1]]}
RULE_CASES = [   # (id, config, path, expected); the first ten are the reference's own examples, same verdicts
    ("straight-cut-leaves-the-balanced-edge-strip", _L1, ((0, 1), (1, 1), (2, 1), (3, 1)), True),
    ("axleless-regions-may-have-any-shape", _T,
     ((0, 3), (1, 3), (1, 2), (0, 2), (0, 1), (1, 1), (2, 1), (3, 1), (3, 0)), True),
    ("4x4-edge-axle-balances-only-three-cells", _E, ((0, 1), (1, 1), (2, 1), (3, 1), (3, 0), (4, 0), (4, 1)), True),
    ("s-shaped-region-is-a-half-turn-not-a-mirror", _F,
     ((0, 4), (0, 3), (0, 2), (1, 2), (1, 1), (2, 1), (3, 1), (3, 2), (3, 3), (2, 3), (2, 4), (3, 4), (4, 4), (4, 3),
      (4, 2), (4, 1), (4, 0)), True),
    ("two-axles-in-two-balanced-regions", _D,
     ((0, 4), (0, 3), (0, 2), (0, 1), (1, 1), (1, 2), (1, 3), (1, 4), (2, 4), (3, 4), (4, 4), (4, 3), (3, 3), (3, 2),
      (3, 1), (3, 0), (4, 0)), True),
    ("border-hug-partner-off-the-board", _T, ((0, 3), (0, 2), (0, 1), (0, 0), (1, 0), (2, 0), (3, 0)), False),
    ("4x4-full-edge-row-is-unbalanced", _E, ((0, 1), (1, 1), (2, 1), (3, 1), (4, 1)), False),
    ("bounding-box-centred-on-the-axle-is-not-enough", _F,
     ((0, 4), (0, 3), (0, 2), (0, 1), (1, 1), (1, 2), (2, 2), (2, 3), (3, 3), (3, 4), (4, 4), (4, 3), (4, 2), (4, 1),
      (4, 0)), False),
    ("two-axles-in-one-region", _D, ((0, 4), (0, 3), (0, 2), (0, 1), (0, 0), (1, 0), (2, 0), (3, 0), (4, 0)), False),
    ("axle-walled-in-alone", _L1, ((0, 1), (0, 0), (1, 0), (1, 1), (2, 1), (2, 0), (3, 0), (3, 1)), False),
    # Hand-derived: a T-shaped region, mirror symmetric across the axle's column, but (1,0) has no partner (1,2).
    ("mirror-symmetric-region-is-not-enough", _C,
     ((0, 1), (1, 1), (1, 0), (2, 0), (2, 1), (3, 1), (3, 2), (2, 2), (1, 2), (0, 2)), False),
]


@pytest.mark.parametrize("config, path, expected", [c[1:] for c in RULE_CASES], ids=[c[0] for c in RULE_CASES])
def test_rule_outcome_on_a_small_board(config, path, expected):
    assert _wins(config, path) is expected


@pytest.mark.parametrize("level", range(LEVELS))
def test_parity_with_the_reference_on_every_simple_path_of_the_level(level):
    """The paths this module accepts are exactly the paths the reference implementation accepts (frozen digest), and
    `golden.count_paths` sees the same counts."""
    config = load_level_configs("rg02")[level]
    board = Board.from_config(config)
    markers = Rg02._parse_markers(config, board)
    paths = simple_paths(board)
    accepted = sorted(";".join(f"{c},{r}" for c, r in p) for p in paths if Rg02._wins(board, markers, p))
    digest = hashlib.sha1("\n".join(accepted).encode()).hexdigest()[:16]
    assert (len(paths), len(accepted), digest) == REFERENCE_VERDICTS[level]
    assert golden.count_paths(Rg02(), board, markers) == REFERENCE_VERDICTS[level][:2]
