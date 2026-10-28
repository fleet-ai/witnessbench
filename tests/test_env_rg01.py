"""The r2 training game rg01 (`witness_env.games.rg01`, manors: the region of a manor with value k borders exactly k
other regions): config checks, glyph, frames, reported markers, stored solutions, rule and feedback (a rejected CONFIRM
flags no cell).

The rule is checked against an independent reference implementation of the rule, which is not part of this
repository. Its verdicts are frozen here as data: its accept and reject examples (RULE_CASES)
and, per level, its verdict on every simple path from the start that ends at the end node (REFERENCE_VERDICTS).
Frames are compared through the first 16 hex digits of the sha1 of their int8 bytes.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest
from arcengine import ActionInput, FrameData, GameAction

from golden import sha16
from witness_env import WitnessGame
from witness_env.base import Board, load_level_configs
from witness_env.games.rg01 import Rg01

RESET, UP, DOWN, LEFT, RIGHT, CONFIRM = 0, 1, 2, 3, 4, 5
LEVELS = 10
MOVES = {(0, -1): "U", (0, 1): "D", (-1, 0): "L", (1, 0): "R"}
# The shortest solving action sequence of each level (breadth-first search, ties broken by the smallest action
# codes).
SOLUTIONS = (
    (2, 2, 4, 2, 3, 5),
    (4, 1, 3, 1, 4, 1, 3, 5),
    (2, 4, 1, 4, 2, 2, 2, 3, 3, 5),
    (2, 2, 3, 1, 1, 3, 1, 4, 1, 4, 2, 4, 2, 4, 2, 3, 5),
    (1, 3, 2, 3, 3, 2, 2, 4, 2, 4, 4, 1, 4, 1, 3, 3, 3, 5),
    (1, 4, 2, 4, 1, 1, 4, 1, 3, 3, 1, 3, 2, 5),
    (2, 2, 2, 4, 1, 1, 1, 1, 4, 2, 2, 4, 2, 3, 2, 5),
    (1, 3, 2, 2, 4, 4, 2, 3, 2, 3, 1, 3, 3, 1, 4, 5),
    (1, 3, 1, 4, 1, 3, 1, 3, 2, 2, 2, 2, 3, 1, 1, 3, 5),
    (2, 4, 2, 3, 3, 1, 1, 3, 2, 2, 3, 2, 4, 2, 4, 1, 4, 5),
)
ENTRY_FRAMES = ("f92dc82b35ccbb7e", "2103ee42dfa81df1", "e23b1beeb77a398d", "367999a462225b4d", "0820da2a220f587c",
                "37a52c442c5a7c38", "418bf5dc5e874607", "c5c36f2713409767", "1b3e1802e03ebc94", "608ae821ae21f666")
# Per level, the reference implementation's verdicts: (simple paths from the start that end at the end node,
# accepted ones, sha16 of the accepted paths written as move letters, sorted, joined by newlines).
REFERENCE_VERDICTS = (
    (178, 74, "3c2bed6765ed74cb"), (178, 39, "489d41577f8b1a0b"), (178, 44, "ca3e538f8992f4c9"),
    (3762, 162, "7c5564aa519e5698"), (4178, 24, "76b8d11378345edc"), (4866, 69, "9a664e6f8628150d"),
    (4914, 107, "eb4a92688add7fbe"), (4178, 48, "952c4460efbff3bf"), (5866, 27, "7c43d994820841e5"),
    (4866, 29, "573a4e220f78d4f4"),
)


def act(game: WitnessGame, action: int) -> FrameData:
    return game.perform_action(ActionInput(id=GameAction.from_id(action)))


def entry_frame(level: int) -> np.ndarray:
    game = Rg01()
    game.set_level(level)
    return np.asarray(game.camera.render(game.current_level.get_sprites()), dtype=np.int8)


def wins(config: dict, path: tuple) -> bool:
    board = Board.from_config(config)
    return Rg01._wins(board, Rg01._parse_markers(config, board), tuple(path))


# --------------------------------------------------------------------------------------- level file, configs
def test_the_level_file_holds_ten_levels_with_only_the_manor_key():
    configs = load_level_configs("rg01")
    assert len(configs) == LEVELS
    assert all(set(c) == {"cols", "rows", "start", "end", "manors"} for c in configs)
    assert act(Rg01(), RESET).win_levels == LEVELS


def _one_level_class(mutate) -> type:
    config = json.loads(json.dumps(load_level_configs("rg01")[0]))
    mutate(config)
    return type("_MutatedRg01", (Rg01,), {"_level_configs": lambda self: [config]})


CONFIG_ERRORS = [   # (id, mutation of the first level config, error message)
    ("missing-key", lambda c: c.pop("manors"), "rg01 level 1: missing key 'manors'"),
    ("board-too-large-for-the-glyph", lambda c: c.update(cols=6, rows=6),
     "rg01 level 1: board 6x6: cell size 8 is below 9"),
    *[(f"value-{v}", lambda c, v=v: c.update(manors={"0,0": v}),
       f"rg01 level 1: manors: cell 0,0 has value {v}, expected one of [1, 2, 3]") for v in (0, 4)],
    ("boolean-value", lambda c: c.update(manors={"0,0": True}),
     "rg01 level 1: manors: cell 0,0 has value True, expected an int"),
]


@pytest.mark.parametrize("mutate, message", [c[1:] for c in CONFIG_ERRORS], ids=[c[0] for c in CONFIG_ERRORS])
def test_malformed_level_config_fails_construction_with_its_message(mutate, message):
    with pytest.raises(ValueError) as exc:
        _one_level_class(mutate)()
    assert str(exc.value) == message


def test_unmodified_first_level_config_constructs():
    assert _one_level_class(lambda c: None)().win_score == 1


# ----------------------------------------------------------------------------------------------- glyph, frames
GLYPHS = {   # the manor glyph of each value, centered on the cell center ('#' = BLACK)
    1: ("#######", "#.....#", "#..#..#", "#.....#", "#.....#", "#.....#", "###.###"),
    2: ("#######", "#.....#", "#.#.#.#", "#.....#", "#.....#", "#.....#", "###.###"),
    3: ("#######", "#.....#", "##.#.##", "#.....#", "#.....#", "#.....#", "###.###"),
}


def test_each_manor_is_drawn_as_its_glyph_and_nothing_else_changes():
    config = {"cols": 3, "rows": 3, "start": [0, 3], "end": [3, 0], "manors": {"0,0": 1, "1,1": 2, "2,2": 3}}
    board = Board.from_config(config)
    empty = board.grid.new_frame()
    frame = board.grid.new_frame()
    Rg01._draw_markers(frame, board, Rg01._parse_markers(config, board), frozenset())
    expected = empty.copy()
    for cell, k in ((0, 0), 1), ((1, 1), 2), ((2, 2), 3):
        x, y = board.grid.cell_center(cell)
        for dy, row in enumerate(GLYPHS[k]):
            for dx, pixel in enumerate(row):
                if pixel == "#":
                    expected[y + dy - 3, x + dx - 3] = 5          # BLACK
    assert np.array_equal(frame, expected)


def test_reset_shows_the_first_entry_frame_and_every_level_has_its_entry_frame():
    assert [sha16(f) for f in act(Rg01(), RESET).frame] == [ENTRY_FRAMES[0]]
    assert tuple(sha16(entry_frame(level)) for level in range(LEVELS)) == ENTRY_FRAMES


def test_failed_confirm_at_the_start_re_renders_the_entry_frame():
    for level, digest in enumerate(ENTRY_FRAMES):
        game = Rg01()
        game.set_level(level)
        fd = act(game, CONFIRM)
        assert [sha16(f) for f in fd.frame] == [digest], level
        assert (fd.levels_completed, game.level_index) == (0, level)


# ------------------------------------------------------------------------------------------- reported markers
def test_truth_of_every_level_declares_the_manors_of_its_level_config():
    game = Rg01()
    for level, config in enumerate(load_level_configs("rg01")):
        game.set_level(level)
        truth = game.truth()
        assert [(m.name, m.glyph, m.values_are_colours) for m in truth.cell_markers] == [("manors", "m", False)]
        assert dict(truth.cell_markers[0].values) == {
            tuple(int(n) for n in key.split(",")): k for key, k in config["manors"].items()}, level
        assert (dict(truth.squares), truth.dots, dict(truth.stars), dict(truth.triangles), dict(truth.polyominoes),
                truth.erasers, truth.framed_cells, truth.node_markers) == ({}, (), {}, {}, {}, frozenset(),
                                                                           frozenset(), ()), level
        assert truth.starts == (tuple(config["start"]),), level


# ---------------------------------------------------------------------------------------------- solutions
def test_stored_solutions_played_in_a_row_win_every_level_and_enter_the_next():
    game = Rg01()
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
        game = Rg01()
        game.set_level(level)
        results = [act(game, a) for a in solution]
        assert [fd.levels_completed for fd in results] == [0] * (len(solution) - 1) + [1], level


# ------------------------------------------------------------------------------------------------------ rules
_T = {"cols": 3, "rows": 3, "start": [0, 3], "end": [3, 0], "manors": {"0,1": 2, "2,2": 1}}
_F = {"cols": 4, "rows": 4, "start": [0, 4], "end": [4, 0], "manors": {"0,3": 1, "2,1": 3}}
_L1 = {"cols": 3, "rows": 3, "start": [0, 3], "end": [3, 3], "manors": {"1,1": 1}}
RULE_CASES = [   # (id, config, path, expected): the reference implementation's examples
    ("three-regions-degrees-2-and-1", _T, ((0, 3), (0, 2), (1, 2), (1, 3), (2, 3), (2, 2), (2, 1), (2, 0), (3, 0)),
     True),
    ("same-verdict-other-partition", _T, ((0, 3), (0, 2), (1, 2), (1, 3), (2, 3), (2, 2), (2, 1), (3, 1), (3, 0)),
     True),
    ("4x4-degree-3-and-pocket-degree-1", _F,
     ((0, 4), (0, 3), (1, 3), (1, 4), (2, 4), (2, 3), (2, 2), (3, 2), (4, 2), (4, 1), (3, 1), (3, 0), (4, 0)), True),
    # The manor's region shares three wall segments with one region: neighboring regions count once.
    ("long-border-counts-once", _L1, ((0, 3), (0, 2), (1, 2), (2, 2), (3, 2), (3, 3)), True),
    ("border-hug-leaves-one-region", _T, ((0, 3), (0, 2), (0, 1), (0, 0), (1, 0), (2, 0), (3, 0)), False),
    # Both manors share a region of degree 2: each manor is checked on its own, so k=1 fails.
    ("co-housed-manors-degree-2", _T, ((0, 3), (0, 2), (0, 1), (1, 1), (1, 0), (2, 0), (2, 1), (3, 1), (3, 0)),
     False),
    # The k=1 manor cell touches one foreign region, but its whole region borders two.
    ("whole-region-not-manor-cell", _T,
     ((0, 3), (1, 3), (1, 2), (0, 2), (0, 1), (1, 1), (2, 1), (2, 2), (3, 2), (3, 1), (3, 0)), False),
    ("straight-along-the-border", _L1, ((0, 3), (1, 3), (2, 3), (3, 3)), False),
    ("two-pockets-give-degree-2", _L1, ((0, 3), (0, 2), (1, 2), (1, 3), (2, 3), (2, 2), (3, 2), (3, 3)), False),
]


@pytest.mark.parametrize("config, path, expected", [c[1:] for c in RULE_CASES], ids=[c[0] for c in RULE_CASES])
def test_rule_outcome_matches_the_reference_example(config, path, expected):
    assert wins(config, path) is expected


def test_a_rejected_confirm_at_the_end_rings_no_cell():
    """As in the r1 training games: level 1 (start (0,0), end (0,3), manor k=1 in cell (1,1)), the straight line down
    the left border leaves one region and is rejected; no cell is flagged and the frame is the entry frame."""
    game = Rg01()
    entry = np.asarray(act(game, RESET).frame[-1], dtype=np.int8)
    for a in (DOWN, DOWN, DOWN):
        act(game, a)
    assert game.truth().path[-1] == Board.from_config(load_level_configs("rg01")[0]).end
    rejected = act(game, CONFIRM)
    assert (game.truth().flagged_cells, rejected.levels_completed, game.level_index) == (frozenset(), 0, 0)
    assert np.array_equal(np.asarray(rejected.frame[-1], dtype=np.int8), entry)


@pytest.mark.parametrize("level", range(LEVELS))
def test_the_rule_agrees_with_the_reference_on_every_path_of_the_level(level):
    """Parity with the independent implementation: the same paths, and the same ones accepted."""
    config = load_level_configs("rg01")[level]
    board = Board.from_config(config)
    markers = Rg01._parse_markers(config, board)
    paths, accepted, stack = 0, [], [(board.starts[0],)]
    while stack:                                              # every simple path; it may pass through the end
        path = stack.pop()
        if len(path) > 1 and path[-1] == board.end:
            paths += 1
            if Rg01._wins(board, markers, path):
                accepted.append("".join(MOVES[(b[0] - a[0], b[1] - a[1])] for a, b in zip(path, path[1:])))
        for dc, dr in MOVES:
            target = (path[-1][0] + dc, path[-1][1] + dr)          # rg01 levels have no broken edges
            if 0 <= target[0] <= board.cols and 0 <= target[1] <= board.rows and target not in path:
                stack.append(path + (target,))
    digest = hashlib.sha1("\n".join(sorted(accepted)).encode("ascii")).hexdigest()[:16]
    assert (paths, len(accepted), digest) == REFERENCE_VERDICTS[level]
