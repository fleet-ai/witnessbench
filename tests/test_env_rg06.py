"""The r2 training game rg06 (`witness_env.games.rg06`, jugs: the region of a jug with value k contains exactly k of the
board's four corner cells): level file, config checks, glyph, frames, reported markers, stored solutions and the rule,
including parity with an independent reference implementation of the rule on every path of every level.

The frozen data below was recorded with tools that are not part of this repository.
`SOLUTIONS` is the lexicographically smallest shortest winning action list of each level (moves up, down, left, right,
then CONFIRM). `REFERENCE_CASES` are the reference implementation's examples with its verdicts. `REFERENCE_PARITY`
holds, per level, its own count of simple start-to-end paths, how many it accepts, and the sha1 of its verdicts (one
byte per path, 1 = accepted) in depth-first order with moves up, down, left, right and every path stopping at the end
node (the reference implementation checks its compact path enumerator against its rule on every path). This module
enumerates the same paths in the same order with `_move_target`, judges them with `Rg06._wins` and must reproduce all
three. Entry frames and marker layers are compared through `golden.sha16`.
"""
from __future__ import annotations

import hashlib
import json
import os

import numpy as np
import pytest
from arcengine import ActionInput, FrameData, GameAction

from golden import sha16
from witness_env import WitnessGame
from witness_env.base import Board, load_level_configs
from witness_env.games.rg06 import Rg06
from witness_env.grid import CURSOR, ORANGE, START

GAME_ID = "rg06"
LEVELS = 10
RESET, UP, DOWN, LEFT, RIGHT, CONFIRM = 0, 1, 2, 3, 4, 5
MOVES = ((0, -1), (0, 1), (-1, 0), (1, 0))
SLOW = os.environ.get("WITNESS_SLOW_TESTS") == "1"

SOLUTIONS = (
    (1, 1, 1, 3, 2, 2, 3, 2, 4, 5),
    (1, 1, 3, 3, 2, 4, 5),
    (4, 1, 4, 4, 1, 3, 1, 3, 3, 5),
    (1, 4, 4, 2, 2, 4, 1, 1, 4, 1, 3, 3, 3, 5),
    (2, 2, 4, 4, 2, 3, 2, 3, 3, 3, 5),
    (1, 4, 2, 2, 3, 2, 3, 2, 4, 4, 1, 4, 4, 2, 5),
    (1, 1, 1, 3, 2, 3, 2, 2, 3, 1, 1, 3, 5),
    (2, 2, 4, 1, 1, 1, 3, 1, 4, 4, 2, 2, 2, 4, 4, 5),
    (2, 2, 2, 2, 2, 4, 1, 1, 4, 4, 4, 4, 2, 2, 5),
    (1, 1, 1, 1, 1, 4, 2, 2, 4, 4, 4, 4, 1, 1, 5),
)
REFERENCE_PARITY = (   # (paths, accepted, sha1 of the verdict bytes) per level, level 1 first
    (125, 32, "57c5ae490f6c14fc74e0a15eb00f9d3d372b7c69"),
    (105, 28, "09ea11a42595625fd6648bd180ba5f55627dfead"),
    (178, 14, "7b17ba35c919bcbd2507e85ea1023104d361763b"),
    (5177, 344, "8d6f9322fe72d1b1360e09fe443e53256201bfd9"),
    (5866, 207, "59977855001d68fe6c2a62113fc2ddb3e7819602"),
    (6218, 64, "1c7e90dbd91aaa0933063d79591455fbc65e750d"),
    (5866, 70, "926c67f03c2bb1095a21b6bfe1e05360efe5af3d"),
    (4519, 34, "1b919f947b0fc7862f873430725caf9bb74028dd"),
    (1262816, 3796, "7b197360c683e7d1e1983a4806c860a33c4726a6"),
    (1262816, 3046, "e71724ebda536a6cf836f3fa7a476d1e640d4558"),
)
ENTRY_FRAMES = ("53f8558bee1d9ae8", "37cdb2e5aa6e8d78", "9ead929c7619565c", "4d164dc25289a507", "c885dab1444e1dd8",
                "ecfaa49c4fa706ee", "99b6e4aa882c1374", "b35692ea325d6401", "5ca2074664924af1", "7e0cb89bbcf661d1")
MARKER_LAYERS = (   # (config that is not a level, sha16 of its marker layer alone)
    ({"cols": 3, "rows": 3, "start": [0, 3], "end": [3, 0], "jugs": {"0,0": 1, "1,1": 2, "2,2": 3}},
     "5891f15b1ec7b425"),
    ({"cols": 4, "rows": 4, "start": [0, 4], "end": [4, 0], "jugs": {"0,1": 2, "3,3": 1}}, "bc153380b0abddaa"),
    ({"cols": 5, "rows": 5, "start": [0, 5], "end": [5, 0], "jugs": {"0,0": 3, "4,4": 1, "2,2": 2}},
     "3b03127305764252"),
    ({"cols": 2, "rows": 5, "start": [0, 0], "end": [2, 5], "jugs": {"1,4": 1}}, "f92688f3d0212186"),
)
GLYPHS = {   # the jug glyph of each value, centered on the cell center ('#' = ORANGE): pips above an open cup
    1: ("...#...", ".......", "#.....#", "#.....#", "#.....#", "#.....#", "#######"),
    2: ("..#.#..", ".......", "#.....#", "#.....#", "#.....#", "#.....#", "#######"),
    3: (".#.#.#.", ".......", "#.....#", "#.....#", "#.....#", "#.....#", "#######"),
}

_BOARD_T = {"cols": 3, "rows": 3, "start": [0, 3], "end": [3, 0], "jugs": {"1,1": 1}}
_BOARD_F = {"cols": 4, "rows": 4, "start": [0, 4], "end": [4, 0], "jugs": {"0,1": 2, "3,3": 1}}
REFERENCE_CASES = (   # (id, config, path, accepted): the reference implementation's examples
    ("A1-top-left-block-holds-one-corner", _BOARD_T,
     ((0, 3), (0, 2), (1, 2), (2, 2), (2, 1), (2, 0), (3, 0)), True),
    ("A2-three-regions-jug-region-holds-one-corner", _BOARD_T,
     ((0, 3), (0, 2), (1, 2), (1, 3), (2, 3), (2, 2), (2, 1), (2, 0), (3, 0)), True),
    ("A3-left-column-two-corners-and-a-one-corner-pocket", _BOARD_F,
     ((0, 4), (1, 4), (1, 3), (1, 2), (1, 1), (1, 0), (2, 0), (2, 1), (2, 2), (2, 3), (3, 3), (3, 2), (4, 2),
      (4, 1), (4, 0)), True),
    ("R1-frame-walk-leaves-all-four-corners", _BOARD_T,
     ((0, 3), (0, 2), (0, 1), (0, 0), (1, 0), (2, 0), (3, 0)), False),
    ("R2-jug-region-holds-two-corners", _BOARD_T,
     ((0, 3), (1, 3), (1, 2), (2, 2), (2, 1), (1, 1), (1, 0), (2, 0), (3, 0)), False),
    ("R3-full-width-cut-leaves-the-two-bottom-corners", _BOARD_T,
     ((0, 3), (0, 2), (0, 1), (1, 1), (2, 1), (3, 1), (3, 0)), False),
    ("R4-frame-walk-on-4x4", _BOARD_F, ((0, 4), (0, 3), (0, 2), (0, 1), (0, 0), (1, 0), (2, 0), (3, 0), (4, 0)), False),
)


def act(game: WitnessGame, action: int) -> FrameData:
    return game.perform_action(ActionInput(id=GameAction.from_id(action)))


def entry_frame(level: int) -> np.ndarray:
    game = Rg06()
    game.set_level(level)
    return np.asarray(game.camera.render(game.current_level.get_sprites()), dtype=np.int8)


def wins(config: dict, path: tuple) -> bool:
    board = Board.from_config(config)
    return Rg06._wins(board, Rg06._parse_markers(config, board), tuple(path))


# --------------------------------------------------------------------------------------- level file, configs
def test_the_level_file_holds_ten_levels_with_only_the_jug_key():
    configs = load_level_configs(GAME_ID)
    assert len(configs) == LEVELS
    assert all(set(c) == {"cols", "rows", "start", "end", "jugs"} for c in configs)
    assert act(Rg06(), RESET).win_levels == LEVELS


def _one_level_class(mutate) -> type:
    config = json.loads(json.dumps(load_level_configs(GAME_ID)[0]))
    mutate(config)
    return type("_MutatedRg06", (Rg06,), {"_level_configs": lambda self: [config]})


CONFIG_ERRORS = [   # (id, mutation of the first level config, error message)
    ("missing-key", lambda c: c.pop("jugs"), "rg06 level 1: missing key 'jugs'"),
    ("unknown-key", lambda c: c.__setitem__("jug", {}), "rg06 level 1: unknown config key 'jug'"),
    ("board-too-large-for-the-glyph", lambda c: c.update(cols=6, rows=6),
     "rg06 level 1: board 6x6: cell size 8 is below 9"),
    *[(f"value-{v}", lambda c, v=v: c.update(jugs={"0,0": v}),
       f"rg06 level 1: jugs: cell 0,0 has value {v}, expected one of [1, 2, 3]") for v in (0, 4)],
    ("boolean-value", lambda c: c.update(jugs={"0,0": True}),
     "rg06 level 1: jugs: cell 0,0 has value True, expected an int"),
]


@pytest.mark.parametrize("mutate, message", [c[1:] for c in CONFIG_ERRORS], ids=[c[0] for c in CONFIG_ERRORS])
def test_malformed_level_config_fails_construction_with_its_message(mutate, message):
    with pytest.raises(ValueError) as exc:
        _one_level_class(mutate)()
    assert str(exc.value) == message


def test_unmodified_first_level_config_constructs():
    assert _one_level_class(lambda c: None)().win_score == 1


# ----------------------------------------------------------------------------------------------- glyph, frames
def test_each_jug_is_drawn_as_its_glyph_and_nothing_else_changes():
    config = MARKER_LAYERS[0][0]
    board = Board.from_config(config)
    frame = board.grid.new_frame()
    Rg06._draw_markers(frame, board, Rg06._parse_markers(config, board), frozenset())
    expected = board.grid.new_frame()
    for key, k in config["jugs"].items():
        x, y = board.grid.cell_center(tuple(int(n) for n in key.split(",")))
        for dy, row in enumerate(GLYPHS[k]):
            for dx, pixel in enumerate(row):
                if pixel == "#":
                    expected[y + dy - 3, x + dx - 3] = ORANGE
    assert np.array_equal(frame, expected)


def test_marker_layers_of_non_level_configs_match_their_frozen_hashes():
    for i, (config, digest) in enumerate(MARKER_LAYERS):
        board = Board.from_config(config)
        frame = board.grid.new_frame()
        Rg06._draw_markers(frame, board, Rg06._parse_markers(config, board), frozenset())
        assert sha16(frame) == digest, i


def test_reset_shows_the_first_entry_frame_and_every_level_has_its_entry_frame():
    assert [sha16(f) for f in act(Rg06(), RESET).frame] == [ENTRY_FRAMES[0]]
    assert tuple(sha16(entry_frame(level)) for level in range(LEVELS)) == ENTRY_FRAMES


def test_failed_confirm_at_the_start_re_renders_the_entry_frame():
    for level, digest in enumerate(ENTRY_FRAMES):
        game = Rg06()
        game.set_level(level)
        fd = act(game, CONFIRM)
        assert [sha16(f) for f in fd.frame] == [digest], level
        assert (fd.levels_completed, game.level_index) == (0, level)


def test_cursor_sprite_follows_the_path_tip():
    assert Rg06.CURSOR_SPRITE_FOLLOWS_TIP is True
    game = Rg06()
    act(game, RESET)
    frame = np.asarray(act(game, SOLUTIONS[0][0]).frame[0])
    truth = game.truth()
    assert len(truth.path) == 2
    grid = Board.from_config(load_level_configs(GAME_ID)[0]).grid
    (tip_x, tip_y), (start_x, start_y) = grid.node_xy(truth.path[-1]), grid.node_xy(truth.start)
    assert (frame[tip_y, tip_x], frame[start_y, start_x]) == (CURSOR, START)


# ------------------------------------------------------------------------------------------- reported markers
def test_truth_of_every_level_declares_the_jugs_of_its_level_config():
    game = Rg06()
    for level, config in enumerate(load_level_configs(GAME_ID)):
        game.set_level(level)
        truth = game.truth()
        assert [(m.name, m.glyph, m.values_are_colours) for m in truth.cell_markers] == [("jugs", "j", False)]
        assert dict(truth.cell_markers[0].values) == {
            tuple(int(n) for n in key.split(",")): k for key, k in config["jugs"].items()}, level
        assert (dict(truth.squares), truth.dots, dict(truth.stars), dict(truth.triangles), dict(truth.polyominoes),
                truth.erasers, truth.framed_cells, truth.node_markers) == ({}, (), {}, {}, {}, frozenset(),
                                                                           frozenset(), ()), level
        assert truth.starts == (tuple(config["start"]),), level


# ------------------------------------------------------------------------------------------ stored solutions
def test_stored_solutions_played_in_a_row_win_every_level_and_enter_the_next():
    game = Rg06()
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
        game = Rg06()
        game.set_level(level)
        results = [act(game, a) for a in solution]
        assert [fd.levels_completed for fd in results] == [0] * (len(solution) - 1) + [1], level


def test_no_stored_solution_wins_another_level():
    for level, solution in enumerate(SOLUTIONS):
        for other in range(LEVELS):
            if other != level:
                game = Rg06()
                game.set_level(other)
                assert [act(game, a).levels_completed for a in solution][-1] == 0, (level, other)


# ------------------------------------------------------------------------------------------------------ rules
@pytest.mark.parametrize("config, path, accepted", [c[1:] for c in REFERENCE_CASES],
                         ids=[c[0] for c in REFERENCE_CASES])
def test_the_reference_examples(config, path, accepted):
    assert wins(config, path) is accepted


_CUT_OFF_TOP_LEFT = ((1, 0), (1, 1), (0, 1))   # on a 2x2 board: cell (0,0) alone, the other three cells together
RULE_CASES = [   # (id, cols, rows, path, jugs, expected), derived by hand on 2x2 boards (four distinct corner cells)
    # The line runs down the middle column: each half holds two corners.
    ("two-k2-jugs-split-the-corners-two-and-two", 2, 2, ((1, 0), (1, 1), (1, 2)), {"0,0": 2, "1,1": 2}, True),
    # A line along the top edge cuts nothing: one region with all four corners.
    ("a-board-in-one-piece-fails-every-jug", 2, 2, ((0, 0), (1, 0), (2, 0)), {"0,0": 3}, False),
    # The jug's own cell is a corner and counts: {(0,0)} holds one corner, the rest three.
    ("the-jugs-own-corner-cell-counts", 2, 2, _CUT_OFF_TOP_LEFT, {"0,0": 1, "1,1": 3}, True),
    # Two jugs in one three-corner region are each checked on their own: k=3 holds, k=2 fails.
    ("jugs-sharing-a-region-are-checked-each", 2, 2, _CUT_OFF_TOP_LEFT, {"1,0": 3, "0,1": 2}, False),
    # The lone corner cell has no jug, so its region is free.
    ("a-region-without-a-jug-is-free", 2, 2, _CUT_OFF_TOP_LEFT, {"1,1": 3}, True),
]


@pytest.mark.parametrize("cols, rows, path, jugs, expected", [c[1:] for c in RULE_CASES],
                         ids=[c[0] for c in RULE_CASES])
def test_rule_outcome_on_a_small_board(cols, rows, path, jugs, expected):
    config = {"cols": cols, "rows": rows, "start": list(path[0]), "end": list(path[-1]), "jugs": jugs}
    assert wins(config, path) is expected


def _verdicts(game: Rg06, config: dict) -> bytes:
    """One byte per simple start-to-end path (1 = `_wins`), depth first with moves up, down, left, right; a path
    stops at the end node."""
    board = Board.from_config(config)
    markers = Rg06._parse_markers(config, board)
    end, out = board.end, bytearray()
    path, on_path = [board.starts[0]], {board.starts[0]}

    def extend() -> None:
        tip = path[-1]
        if tip == end:
            out.append(Rg06._wins(board, markers, tuple(path)))
            return
        for dc, dr in MOVES:
            target = game._move_target(board, tip, dc, dr)
            if target is not None and target not in on_path:
                path.append(target)
                on_path.add(target)
                extend()
                on_path.discard(path.pop())

    extend()
    return bytes(out)


# The 5x5 levels (L9-L10, 1,262,816 paths each, about 45 s per level) run only with WITNESS_SLOW_TESTS=1.
_PARITY_LEVELS = [pytest.param(level, marks=pytest.mark.skipif(
    not SLOW and config["cols"] * config["rows"] >= 25, reason="5x5 level: set WITNESS_SLOW_TESTS=1"))
    for level, config in enumerate(load_level_configs(GAME_ID))]


@pytest.mark.parametrize("level", _PARITY_LEVELS)
def test_the_rule_and_the_reference_agree_on_every_simple_path_of_every_level(level):
    verdicts = _verdicts(Rg06(), load_level_configs(GAME_ID)[level])
    assert (len(verdicts), sum(verdicts), hashlib.sha1(verdicts).hexdigest()) == REFERENCE_PARITY[level]
