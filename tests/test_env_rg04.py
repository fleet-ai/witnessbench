"""The r2 training game rg04 (`witness_env.games.rg04`, urns and drums): level file, config checks, frames, reported
markers, stored solutions and the rule, including parity with an independent reference implementation of the rule on
every path of every level.

The frozen data below was recorded with tools that are not part of this repository.
`SOLUTIONS` is the first shortest accepted path of each level (moves up, down, left, right, then CONFIRM).
`REFERENCE_CASES` are the reference implementation's examples with its verdicts. `REFERENCE_PARITY` holds, per level,
its own count of simple start-to-end paths, how many it accepts, and the sha1 of its verdicts (one byte per path,
1 = accepted) in depth-first order with moves up, down, left, right and every path stopping at the end node. This
module enumerates the same paths in the same order with `_move_target`, judges them with `Rg04._wins` and must
reproduce all three. Entry frames and marker layers are compared through `golden.sha16`.
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
from witness_env.games.rg04 import Rg04
from witness_env.grid import CURSOR, MAGENTA, MAROON, START

GAME_ID = "rg04"
LEVELS = 10
RESET, UP, DOWN, LEFT, RIGHT, CONFIRM = 0, 1, 2, 3, 4, 5
MOVES = ((0, -1), (0, 1), (-1, 0), (1, 0))
SLOW = os.environ.get("WITNESS_SLOW_TESTS") == "1"

SOLUTIONS = (
    (3, 1, 1, 4, 2, 4, 2, 2, 5),
    (3, 3, 2, 2, 4, 1, 5),
    (2, 3, 2, 4, 4, 1, 4, 4, 5),
    (2, 2, 3, 2, 2, 3, 1, 3, 1, 3, 5),
    (2, 3, 2, 4, 4, 4, 1, 4, 2, 2, 5),
    (2, 2, 3, 1, 3, 2, 2, 2, 3, 1, 1, 1, 3, 5),
    (2, 4, 2, 3, 2, 3, 1, 1, 1, 3, 2, 2, 3, 5),
    (1, 1, 1, 3, 3, 2, 4, 2, 2, 2, 3, 1, 3, 1, 1, 5),
    (4, 2, 2, 3, 2, 2, 2, 4, 1, 1, 4, 4, 4, 4, 2, 2, 5),
    (4, 4, 4, 4, 2, 2, 2, 3, 3, 2, 2, 4, 1, 4, 4, 2, 5),
)
REFERENCE_PARITY = (   # (paths, accepted, sha1 of the verdict bytes) per level, level 1 first
    (114, 30, "44d64fc873d5946ada3be28116770872ff11fb85"),
    (105, 16, "1a0c8f936aff1311f5161ebde74aa20a49ebb445"),
    (5489, 717, "dc004a7c08391d0e84c104c20489d65373aa5044"),
    (5866, 248, "66ebbb88e7befa412fa1131eff3118a8aa05808c"),
    (5328, 184, "4411d69412c1cac2994e28ac28d17621da8500cd"),
    (6672, 94, "7c3c349d0494f7fd756f30c8fd979fcc26006b6f"),
    (4914, 115, "f4ffd240f17f9e12e180d4f0196504e795cab407"),
    (4914, 19, "7281f204752e23cb340693779a657ad5a29d7176"),
    (1262816, 12595, "c4b993611a6bb8ada9c745ffbc05b68d4aa29105"),
    (1262816, 8250, "ba1720a76238bfd18d038555d66ef41accaa5927"),
)
ENTRY_FRAMES = ("d144fb8acd192938", "76b87b0aace667b5", "bfc9f89fca7f7108", "295ba37e8fef73e6", "f8257b4f99af4679",
                "6f00209cedf3ddde", "8001df7721efe1bc", "fa7b2879c74ae11d", "4196e74828f620f9", "3b4755bc0808e27a")
MARKER_LAYERS = (   # (config that is not a level, sha16 of its marker layer alone)
    ({"cols": 3, "rows": 3, "start": [0, 3], "end": [3, 0], "urns": [[2, 2]], "drums": [[0, 0]]},
     "cf0d03a0ef50e4b6"),
    ({"cols": 4, "rows": 4, "start": [0, 4], "end": [4, 0], "urns": [[0, 0], [3, 3]], "drums": []},
     "9d28e74b9e1f2f8b"),
    ({"cols": 5, "rows": 5, "start": [0, 5], "end": [5, 0], "urns": [[0, 0], [2, 2], [4, 4]],
      "drums": [[4, 0], [0, 4], [1, 3]]}, "bafe5d244f21608b"),
    ({"cols": 2, "rows": 5, "start": [0, 0], "end": [2, 5], "urns": [], "drums": [[1, 4]]}, "323858bb937b6229"),
)
URN_PIXELS = ((-2, -2), (-1, -1), (0, 0), (1, 1), (2, 2), (2, -2), (1, -1), (-1, 1), (-2, 2))
DRUM_PIXELS = ((-2, -1), (-1, -1), (0, -1), (1, -1), (2, -1), (-2, 1), (-1, 1), (0, 1), (1, 1), (2, 1))

_BOARD_T = {"cols": 3, "rows": 3, "start": [0, 3], "end": [3, 0], "urns": [[2, 2]], "drums": [[0, 0]]}
_BOARD_F = {"cols": 4, "rows": 4, "start": [0, 4], "end": [4, 0], "urns": [[0, 0]], "drums": [[3, 3]]}
REFERENCE_CASES = (   # (id, config, path, accepted): the reference implementation's examples
    ("A1-two-cell-pocket-for-the-drum", _BOARD_T, ((0, 3), (0, 2), (0, 1), (1, 1), (2, 1), (2, 0), (3, 0)), True),
    ("A2-four-cell-pocket-for-the-drum", _BOARD_T, ((0, 3), (0, 2), (1, 2), (2, 2), (2, 1), (2, 0), (3, 0)), True),
    ("A3-full-width-cut", _BOARD_T, ((0, 3), (0, 2), (1, 2), (2, 2), (3, 2), (3, 1), (3, 0)), True),
    ("A4-urn-walled-in-alone", _BOARD_T, ((0, 3), (1, 3), (2, 3), (2, 2), (3, 2), (3, 1), (3, 0)), True),
    ("A5-three-regions-on-4x4", _BOARD_F,
     ((0, 4), (0, 3), (1, 3), (1, 4), (2, 4), (2, 3), (2, 2), (3, 2), (4, 2), (4, 1), (4, 0)), True),
    ("A6-shortest-win-on-4x4", _BOARD_F,
     ((0, 4), (0, 3), (0, 2), (0, 1), (1, 1), (1, 0), (2, 0), (2, 1), (3, 1), (3, 0), (4, 0)), True),
    ("R1-frame-walk-leaves-the-drum-nine-cells", _BOARD_T,
     ((0, 3), (0, 2), (0, 1), (0, 0), (1, 0), (2, 0), (3, 0)), False),
    ("R2-drum-walled-in-alone", _BOARD_T, ((0, 3), (0, 2), (0, 1), (1, 1), (1, 0), (2, 0), (3, 0)), False),
    ("R3-column-cut-swaps-the-parities", _BOARD_T, ((0, 3), (1, 3), (1, 2), (1, 1), (1, 0), (2, 0), (3, 0)), False),
    ("R4-urn-alone-but-the-drum-has-fifteen", _BOARD_F,
     ((0, 4), (0, 3), (0, 2), (0, 1), (1, 1), (1, 0), (2, 0), (3, 0), (4, 0)), False),
    ("R5-frame-walk-on-4x4-leaves-the-urn-sixteen", _BOARD_F,
     ((0, 4), (0, 3), (0, 2), (0, 1), (0, 0), (1, 0), (2, 0), (3, 0), (4, 0)), False),
)


def act(game: WitnessGame, action: int) -> FrameData:
    return game.perform_action(ActionInput(id=GameAction.from_id(action)))


def entry_frame(level: int) -> np.ndarray:
    game = Rg04()
    game.set_level(level)
    return np.asarray(game.camera.render(game.current_level.get_sprites()), dtype=np.int8)


def wins(config: dict, path: tuple) -> bool:
    board = Board.from_config(config)
    return Rg04._wins(board, Rg04._parse_markers(config, board), path)


# --------------------------------------------------------------------------------------- level files, configs
def test_reset_reports_the_level_count_and_the_first_entry_frame():
    fd = act(Rg04(), RESET)
    assert fd.win_levels == LEVELS
    assert [sha16(f) for f in fd.frame] == [ENTRY_FRAMES[0]]


def test_every_level_config_has_exactly_the_structural_keys_and_both_marker_lists():
    configs = load_level_configs(GAME_ID)
    assert len(configs) == LEVELS
    assert all(set(c) == {"cols", "rows", "start", "end", "urns", "drums"} for c in configs)


def _first_level_class(mutate) -> type:
    config = json.loads(json.dumps(load_level_configs(GAME_ID)[1]))   # level 2 has an urn and a drum
    mutate(config)
    return type("_MutatedRg04", (Rg04,), {"_level_configs": lambda self: [config]})


CONFIG_ERRORS = [   # (id, mutation of level 2's config, error message)
    ("missing-urns", lambda c: c.pop("urns"), "rg04 level 1: missing key 'urns'"),
    ("missing-drums", lambda c: c.pop("drums"), "rg04 level 1: missing key 'drums'"),
    ("unknown-key", lambda c: c.__setitem__("urn", []), "rg04 level 1: unknown config key 'urn'"),
    ("board-too-large-for-the-glyphs", lambda c: c.update(cols=6, rows=6),
     "rg04 level 1: board 6x6: cell size 8 is below 9"),
    ("no-marker-at-all", lambda c: c.update(urns=[], drums=[]),
     "rg04 level 1: urns and drums: expected at least one cell, got none"),
    ("urn-and-drum-on-one-cell", lambda c: c.update(urns=[[0, 0]], drums=[[0, 0]]),
     "rg04 level 1: urns and drums: cell 0,0 is listed twice"),
    ("one-urn-twice", lambda c: c.update(urns=[[1, 1], [1, 1]]),
     "rg04 level 1: urns and drums: cell 1,1 is listed twice"),
    ("urns-not-a-list", lambda c: c.__setitem__("urns", {"0,0": 1}),
     "rg04 level 1: urns: expected a list of [col, row] pairs, got {'0,0': 1}"),
    ("boolean-coordinate", lambda c: c.__setitem__("drums", [[True, 0]]),
     "rg04 level 1: drums: expected [col, row] with two ints, got [True, 0]"),
]


@pytest.mark.parametrize("mutate, message", [c[1:] for c in CONFIG_ERRORS], ids=[c[0] for c in CONFIG_ERRORS])
def test_malformed_level_config_fails_construction_with_its_message(mutate, message):
    with pytest.raises(ValueError) as exc:
        _first_level_class(mutate)()
    assert str(exc.value) == message


@pytest.mark.parametrize("mutate", [lambda c: None, lambda c: c.update(urns=[]), lambda c: c.update(drums=[])],
                         ids=["unmodified", "drums-only", "urns-only"])
def test_a_level_with_one_marker_kind_or_both_constructs(mutate):
    assert _first_level_class(mutate)().win_score == 1


# ----------------------------------------------------------------------------------------------------- frames
def test_entry_frames_match_the_frozen_entry_frames():
    assert [sha16(entry_frame(level)) for level in range(LEVELS)] == list(ENTRY_FRAMES)


def test_failed_confirm_at_the_start_re_renders_the_entry_frame():
    for level, digest in enumerate(ENTRY_FRAMES):
        game = Rg04()
        game.set_level(level)
        fd = act(game, CONFIRM)
        assert [sha16(f) for f in fd.frame] == [digest], level
        assert (fd.levels_completed, game.level_index) == (0, level)


def test_marker_layers_of_non_level_configs_match_their_frozen_hashes():
    for i, (config, digest) in enumerate(MARKER_LAYERS):
        board = Board.from_config(config)
        frame = board.grid.new_frame()
        Rg04._draw_markers(frame, board, Rg04._parse_markers(config, board), frozenset())
        assert sha16(frame) == digest, i


def test_the_marker_layer_paints_a_magenta_cross_per_urn_and_a_maroon_double_bar_per_drum_and_nothing_else():
    config = MARKER_LAYERS[2][0]
    board = Board.from_config(config)
    blank = board.grid.new_frame()
    frame = board.grid.new_frame()
    Rg04._draw_markers(frame, board, Rg04._parse_markers(config, board), frozenset())
    expected = {}
    for key, glyph, colour in (("urns", URN_PIXELS, MAGENTA), ("drums", DRUM_PIXELS, MAROON)):
        for c, r in config[key]:
            x, y = board.grid.cell_center((c, r))
            expected.update({(x + dx, y + dy): colour for dx, dy in glyph})
    changed = {(x, y): frame[y][x] for y in range(len(frame)) for x in range(len(frame[0]))
               if frame[y][x] != blank[y][x]}
    assert changed == expected


def test_cursor_sprite_follows_the_path_tip():
    assert Rg04.CURSOR_SPRITE_FOLLOWS_TIP is True
    game = Rg04()
    act(game, RESET)
    frame = np.asarray(act(game, SOLUTIONS[0][0]).frame[0])
    truth = game.truth()
    assert len(truth.path) == 2
    grid = Board.from_config(load_level_configs(GAME_ID)[0]).grid
    (tip_x, tip_y), (start_x, start_y) = grid.node_xy(truth.path[-1]), grid.node_xy(truth.start)
    assert (frame[tip_y, tip_x], frame[start_y, start_x]) == (CURSOR, START)


# ------------------------------------------------------------------------------------------- reported markers
def test_truth_of_every_level_declares_the_urns_and_drums_of_its_level_config():
    game = Rg04()
    for level, config in enumerate(load_level_configs(GAME_ID)):
        game.set_level(level)
        truth = game.truth()
        assert [(m.name, m.glyph, m.values_are_colours) for m in truth.cell_markers] == [
            ("urns", "u", False), ("drums", "d", False)], level
        for marker in truth.cell_markers:
            assert dict(marker.values) == {(c, r): None for c, r in config[marker.name]}, (level, marker.name)
        assert (truth.dots, dict(truth.squares), dict(truth.stars), dict(truth.triangles), dict(truth.polyominoes),
                truth.erasers, truth.framed_cells, truth.node_markers) == ((), {}, {}, {}, {}, frozenset(),
                                                                           frozenset(), ()), level
        assert truth.starts == (tuple(config["start"]),), level


# ------------------------------------------------------------------------------------------ stored solutions
def test_stored_solutions_played_in_a_row_win_every_level_and_enter_the_next():
    game = Rg04()
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
        game = Rg04()
        game.set_level(level)
        results = [act(game, a) for a in solution]
        assert [fd.levels_completed for fd in results] == [0] * (len(solution) - 1) + [1], level


def test_no_stored_solution_wins_another_level():
    for level, solution in enumerate(SOLUTIONS):
        for other in range(LEVELS):
            if other != level:
                game = Rg04()
                game.set_level(other)
                assert [act(game, a).levels_completed for a in solution][-1] == 0, (level, other)


# ------------------------------------------------------------------------------------------------------ rules
@pytest.mark.parametrize("config, path, accepted", [c[1:] for c in REFERENCE_CASES],
                         ids=[c[0] for c in REFERENCE_CASES])
def test_the_reference_examples(config, path, accepted):
    assert wins(config, path) is accepted


RULE_CASES = [   # (id, cols, rows, path, markers, expected), derived by hand
    # The line runs down column 1: left region 2 cells (even), right region 4 cells (even).
    ("two-drums-in-two-even-regions", 2, 2, ((1, 0), (1, 1), (1, 2)), {"urns": [], "drums": [[0, 0], [1, 1]]}, True),
    # Two urns may share a region: the frame walk leaves one 3-cell region (odd).
    ("two-urns-share-an-odd-region", 3, 1, ((0, 0), (1, 0), (2, 0), (3, 0)), {"urns": [[0, 0], [2, 0]], "drums": []},
     True),
    # An urn and a drum in the one region of a frame walk: a region is never both odd and even.
    ("an-urn-and-a-drum-never-share-a-region", 3, 1, ((0, 0), (1, 0), (2, 0), (3, 0)),
     {"urns": [[0, 0]], "drums": [[2, 0]]}, False),
    # The marker's own cell counts: a lone urn cell is a region of 1 (odd), the drum's rest is 2 (even).
    ("the-markers-own-cell-counts", 3, 1, ((1, 0), (1, 1)), {"urns": [[0, 0]], "drums": [[2, 0]]}, True),
    # A region without a marker is free: the cut leaves (0,0) alone (1 cell, odd, no marker) and the drum 2 cells.
    ("a-region-without-a-marker-is-free", 3, 1, ((1, 0), (1, 1)), {"urns": [], "drums": [[1, 0]]}, True),
]


@pytest.mark.parametrize("cols, rows, path, markers, expected", [c[1:] for c in RULE_CASES],
                         ids=[c[0] for c in RULE_CASES])
def test_rule_outcome_on_a_small_board(cols, rows, path, markers, expected):
    config = markers | {"cols": cols, "rows": rows, "start": list(path[0]), "end": list(path[-1])}
    assert wins(config, path) is expected


def _verdicts(game: Rg04, config: dict) -> bytes:
    """One byte per simple start-to-end path (1 = `_wins`), depth first with moves up, down, left, right; a path
    stops at the end node."""
    board = Board.from_config(config)
    markers = Rg04._parse_markers(config, board)
    end, out = board.end, bytearray()
    path, on_path = [board.starts[0]], {board.starts[0]}

    def extend() -> None:
        tip = path[-1]
        if tip == end:
            out.append(Rg04._wins(board, markers, tuple(path)))
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
    verdicts = _verdicts(Rg04(), load_level_configs(GAME_ID)[level])
    assert (len(verdicts), sum(verdicts), hashlib.sha1(verdicts).hexdigest()) == REFERENCE_PARITY[level]
