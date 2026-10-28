"""The optional parts of the game base (`witness_env.base`): the levels package, flagged cells, the visit record and the
snapshot's legend field.

Every game of the registry leaves them at their defaults (checked first). The behavior is checked on small
test-local games with the levels of `test_env_engine.py`: level 1 is a 2x2 board with the starts (0, 2) and (2, 2),
a broken edge above (0, 2) and one mandatory dot (1, 1); level 2 is a 1x1 board. Frames are compared with each other
as arrays, never through stored digests.
"""
from __future__ import annotations

import importlib
import json
import sys
from dataclasses import replace

import numpy as np
import pytest
from arcengine import ActionInput, FrameData, GameAction

from witness_env import GAME_IDS, BoardTruth, CellMarkers, LegendEntry, RevealNodes, make_game
from witness_env.base import STRUCTURAL_KEYS, WitnessGame, parse_coords
from witness_env.grid import BACKGROUND, DOT, FLAGGED, LIGHT_GRAY, VISITED_DOT, WHITE, Frame, Grid
from witness_env.truth import Node

LEVELS = [
    {"cols": 2, "rows": 2, "starts": [[0, 2], [2, 2]], "end": [2, 0], "breakpoints": [[[0, 2], [0, 1]]],
     "dots": [[1, 1]]},
    {"cols": 1, "rows": 1, "start": [0, 1], "end": [1, 0], "dots": []},
]
TO_END_MISSING_THE_DOT = [4, 4, 1, 1]   # (0,2) (1,2) (2,2) (2,1) (2,0): on the end, the dot (1,1) missed
SOLVE_LEVEL_1 = [4, 1, 1, 4, 5]
BLOCKED = 2                             # down from (0,2), and no start can move down
FLAGS = frozenset({(0, 0), (1, 1)})
LAMP: Node = (1, 2)


class _Probe(WitnessGame[tuple[Node, ...]]):
    GAME_ID = "probe"
    CONFIG_KEYS = STRUCTURAL_KEYS | {"dots"}

    def _level_configs(self):
        return LEVELS

    def _parse_markers(self, config, board):
        return parse_coords(config["dots"], "dots")

    def _truth_markers(self, truth, dots):
        return replace(truth, dots=dots)

    def _draw_markers(self, frame, board, dots, visited):
        for d in dots:
            board.grid.draw_dot(frame, d, VISITED_DOT if d in visited else DOT)

    def _wins(self, board, dots, path):
        return set(dots) <= set(path)


class _Flagging(_Probe):
    """Flags (0, 0), (1, 1) and the off-board cell (2, 0) on every rejected CONFIRM at the end."""

    def _flagged_cells(self, board, dots, path, seen):
        return {(0, 0), (1, 1), (2, 0)}


class _Revealing(_Probe):
    """The node (1, 2) is drawn light gray until the visit record holds it, white afterwards. The snapshot reports the
    whole visit record as the reveal nodes "record", and the cell (0, 1) as the revealed marker "mark" once (1, 2) is
    in the record. A rejected CONFIRM at the end flags (0, 0) only when (1, 2) is in the record."""

    def _draw_reveals(self, frame, board, dots, seen):
        if LAMP[0] <= board.cols and LAMP[1] <= board.rows:      # level 2 has no node (1, 2)
            x, y = board.grid.node_xy(LAMP)
            frame[y, x] = WHITE if LAMP in seen else LIGHT_GRAY

    def _truth_reveals(self, truth, dots, seen):
        nodes = {(c, r): (c, r) in seen for c in range(truth.cols + 1) for r in range(truth.rows + 1)}
        shown = (CellMarkers("mark", "m", {(0, 1): None}),) if LAMP in seen else ()
        return replace(truth, reveal_nodes=(RevealNodes("record", nodes),), revealed_markers=shown)

    def _flagged_cells(self, board, dots, path, seen):
        return {(0, 0)} if LAMP in seen else set()


class _OverGaps(_Probe):
    """Draws, from the visit record hook, every pixel of each broken edge's line white."""

    def _draw_reveals(self, frame, board, dots, seen):
        for a, b in board.breakpoints:
            (xa, ya), (xb, yb) = board.grid.node_xy(a), board.grid.node_xy(b)
            for y in range(min(ya, yb), max(ya, yb) + 1):
                for x in range(min(xa, xb), max(xa, xb) + 1):
                    frame[y, x] = WHITE


def play(game: WitnessGame, actions: list[int]) -> list[FrameData]:
    return [game.perform_action(ActionInput(id=GameAction.from_id(a))) for a in actions]


def last_frame(results: list[FrameData]) -> Frame:
    return np.asarray(results[-1].frame[-1], dtype=np.int8)


def ring(grid: Grid, cell: tuple[int, int]) -> set[tuple[int, int]]:
    """The (x, y) pixels of the ring on `cell`: the outermost pixels of its interior."""
    x, y = grid.node_xy(cell)
    n = grid.cell_size
    return {(x + i, y + j) for i in range(1, n + 1) for j in (1, n)} | {(x + i, y + j) for j in range(1, n + 1)
                                                                        for i in (1, n)}


def record(game: WitnessGame) -> set[Node]:
    (nodes,) = game.truth().reveal_nodes
    return {node for node, lit in nodes.lit.items() if lit}


# --------------------------------------------------------------------------------------------- defaults
def test_every_registered_game_keeps_the_optional_parts_at_their_defaults():
    for game_id in GAME_IDS:
        cls = type(make_game(game_id))
        assert cls.LEVELS_PACKAGE == "witness_env", game_id
        for hook in ("_flagged_cells", "_draw_reveals", "_truth_reveals"):
            assert getattr(cls, hook) is getattr(WitnessGame, hook), (game_id, hook)
        assert make_game(game_id).truth().legend == (), game_id


def test_a_game_without_the_optional_parts_reports_their_fields_empty():
    game = _Probe()
    play(game, [0] + TO_END_MISSING_THE_DOT + [5])          # rejected at the end
    truth = game.truth()
    assert (truth.flagged_cells, truth.reveal_nodes, truth.revealed_markers, truth.legend) == (frozenset(), (), (), ())


def test_the_new_snapshot_fields_stay_out_of_the_repr():
    truth = BoardTruth(1, 1, ((0, 1),), (0, 1), ((0, 1),), (1, 0))
    assert repr(truth).endswith("node_markers=(), cell_markers=())")
    used = replace(truth, flagged_cells=frozenset({(0, 0)}), reveal_nodes=(RevealNodes("r", {(0, 0): True}),))
    assert repr(used) == repr(truth) and used != truth
    legend = replace(truth, legend=(LegendEntry("m", 6, "bar"),))
    assert repr(legend) == repr(truth) and legend != truth       # left out of the repr, not out of ==


def test_legend_entries_are_frozen():
    entry = LegendEntry("m", 6, "bar")
    with pytest.raises(AttributeError):
        entry.colour = 12


def test_what_depends_on_the_visit_record_is_drawn_over_the_broken_edges():
    plain, over = last_frame(play(_Probe(), [0])), last_frame(play(_OverGaps(), [0]))
    grid = Grid(2, 2)
    (xa, ya), (_, yb) = grid.node_xy((0, 1)), grid.node_xy((0, 2))    # level 1's broken edge (0, 2)-(0, 1)
    k = max(1, (grid.cell_size + 1) * 2 // 10)
    gap = [(xa, y) for y in range(ya + k, yb - k + 1)]
    assert all(plain[y, x] == BACKGROUND for x, y in gap)              # the gap itself
    assert all(over[y, x] == WHITE for x, y in gap)                     # drawn over it


def test_reveal_nodes_are_read_only_copies():
    lit = {(0, 0): False}
    nodes = RevealNodes("r", lit)
    lit[(0, 0)] = True
    assert nodes.lit == {(0, 0): False}
    with pytest.raises(TypeError):
        nodes.lit[(0, 0)] = True


# ------------------------------------------------------------------------------------- levels package
def test_the_level_file_is_read_from_the_levels_folder_of_the_levels_package(tmp_path, monkeypatch):
    levels = tmp_path / "probe_levels" / "levels"
    levels.mkdir(parents=True)
    (levels.parent / "__init__.py").write_text("", encoding="utf-8")
    (levels / "probe.json").write_text(json.dumps(LEVELS[1:]), encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    importlib.invalidate_caches()
    packaged = type("_Packaged", (_Probe,), {"LEVELS_PACKAGE": "probe_levels",
                                             "_level_configs": WitnessGame._level_configs})
    try:
        game = packaged()
    finally:
        sys.modules.pop("probe_levels", None)
    assert (game.win_score, game.truth().cols, game.truth().end) == (1, 1, (1, 0))
    assert _Probe.LEVELS_PACKAGE == "witness_env"


# --------------------------------------------------------------------------------------- flagged cells
def test_a_rejected_confirm_at_the_end_rings_and_reports_the_flagged_cells_on_the_board():
    game = _Flagging()
    entry = last_frame(play(game, [0]))
    results = play(game, TO_END_MISSING_THE_DOT + [5])
    truth = game.truth()
    assert (truth.flagged_cells, truth.path, results[-1].levels_completed) == (FLAGS, ((0, 2),), 0)
    ringed = last_frame(results)
    pixels = ring(Grid(2, 2), (0, 0)) | ring(Grid(2, 2), (1, 1))
    assert {(int(x), int(y)) for y, x in zip(*np.nonzero(ringed != entry))} == pixels
    assert all(ringed[y, x] == FLAGGED for x, y in pixels)
    assert game.truth().flagged_cells == FLAGS             # no further action: the snapshot keeps them


def test_a_confirm_away_from_the_end_or_a_winning_confirm_flags_nothing():
    game = _Flagging()
    play(game, [0, 4, 1, 5])                                # rule met, tip on (1,1), not on the end
    assert game.truth().flagged_cells == frozenset()
    play(game, TO_END_MISSING_THE_DOT + [5] + SOLVE_LEVEL_1)
    assert (game.level_index, game.truth().flagged_cells) == (1, frozenset())


@pytest.mark.parametrize("after, rings_stay", [
    ([4], False),
    ([BLOCKED], False),
    ([5], False),
    ([6], True),
    ([7], True),
    ([7, BLOCKED], True),
    ([7, 7], True),
    ([7, 4], False),
    ([7, 5], False),
], ids=["move", "blocked_move", "confirm", "action6", "action7", "action7_then_blocked_move", "action7_twice",
        "action7_then_move", "action7_then_confirm"])
def test_the_next_action_clears_the_flags_and_only_a_redraw_removes_the_rings(after, rings_stay):
    game, plain = _Flagging(), _Probe()
    ringed = last_frame(play(game, [0] + TO_END_MISSING_THE_DOT + [5]))
    play(plain, [0] + TO_END_MISSING_THE_DOT + [5])
    frame, expected = last_frame(play(game, after)), last_frame(play(plain, after))
    assert game.truth().flagged_cells == frozenset()
    assert game.truth() == plain.truth()
    assert np.array_equal(frame, ringed if rings_stay else expected)
    if rings_stay:
        assert not np.array_equal(frame, expected)


def test_reset_and_a_level_change_clear_the_flags():
    game = _Flagging()
    entry = last_frame(play(game, [0]))
    play(game, TO_END_MISSING_THE_DOT + [5])
    results = play(game, [0])
    assert (game.truth().flagged_cells, results[-1].full_reset) == (frozenset(), False)
    assert np.array_equal(last_frame(results), entry)
    play(game, TO_END_MISSING_THE_DOT + [5])
    game.set_level(1)
    assert game.truth().flagged_cells == frozenset()


# --------------------------------------------------------------------------------------- visit record
def test_the_visit_record_holds_the_first_start_and_every_node_the_path_covered():
    game = _Revealing()
    assert record(game) == {(0, 2)}
    play(game, [0, 4, 1])                                   # (0,2) (1,2) (1,1)
    assert record(game) == {(0, 2), (1, 2), (1, 1)}
    play(game, [2, 3, BLOCKED])                             # retract twice, then a blocked move
    assert (game.truth().path, record(game)) == (((0, 2),), {(0, 2), (1, 2), (1, 1)})


def test_a_start_chosen_by_a_blocked_first_move_enters_the_visit_record():
    game = _Revealing()
    play(game, [0, 1])                                      # up is blocked at (0,2): the path starts at (2,2)
    assert (game.truth().path, record(game)) == (((2, 2), (2, 1)), {(0, 2), (2, 2), (2, 1)})


def test_a_failed_confirm_keeps_the_visit_record_and_reset_or_a_level_change_restarts_it():
    game = _Revealing()
    play(game, [0] + TO_END_MISSING_THE_DOT + [5])
    assert record(game) == {(0, 2), (1, 2), (2, 2), (2, 1), (2, 0)} and game.truth().path == ((0, 2),)
    play(game, [0])
    assert record(game) == {(0, 2)}
    play(game, SOLVE_LEVEL_1)
    assert (game.level_index, record(game)) == (1, {(0, 1)})


def test_the_reveal_hooks_draw_and_report_from_the_visit_record():
    game = _Revealing()
    x, y = Grid(2, 2).node_xy(LAMP)
    assert last_frame(play(game, [0]))[y, x] == LIGHT_GRAY and game.truth().revealed_markers == ()
    assert last_frame(play(game, [4, 3]))[y, x] == WHITE  # reached, then left again
    assert game.truth().revealed_markers == (CellMarkers("mark", "m", {(0, 1): None}),)
    assert last_frame(play(game, [5]))[y, x] == WHITE     # a failed CONFIRM keeps it
    assert last_frame(play(game, [0]))[y, x] == LIGHT_GRAY and game.truth().revealed_markers == ()


def test_the_flag_hook_sees_the_visit_record_at_the_confirm():
    game = _Revealing()
    play(game, [0, 1, 1, 5])                                # (2,2) (2,1) (2,0): on the end, (1,2) never reached
    assert game.truth().flagged_cells == frozenset()
    play(game, [4, 4, 1, 1, 5])                             # through (1,2) to the end
    assert game.truth().flagged_cells == frozenset({(0, 0)})
