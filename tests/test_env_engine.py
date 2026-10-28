"""Movement, CONFIRM, resets, frames and level loading of the game base (`witness_env.base`).

The behavior is checked on a small test-local game: level 1 is a 2x2 board with two starts, a broken edge and
one mandatory dot (the rule: every dot on the path); level 2 is a 1x1 board with no dots. Frames are compared
through the first 16 hex digits of the sha1 of their int8 bytes.

The last three tests cover every game: its level file, its registry entry and its rule line in docs/games.md.
"""
from __future__ import annotations

import dataclasses
import importlib
import json
import re
from dataclasses import FrozenInstanceError, replace
from importlib.resources import files
from pathlib import Path
from types import MappingProxyType

import pytest
from arcengine import ActionInput, FrameData, GameAction, GameState

from golden import LEVEL_COUNTS, MARKER_KEYS, sha16
from witness_env import GAME_IDS, TRAIN_R1_IDS, TRAIN_R2_IDS, BoardTruth, CellMarkers, Polyomino, make_game
from witness_env.base import (STRUCTURAL_KEYS, Board, WitnessGame, load_level_configs, parse_cell_key, parse_cell_map,
                              parse_coord, parse_coords, parse_int_map, require_cell_size)
from witness_env.grid import CURSOR, DOT, START, VISITED_DOT, Grid, path_edges, split_regions
from witness_env.truth import Node

LEVELS = [
    {"cols": 2, "rows": 2, "starts": [[0, 2], [2, 2]], "end": [2, 0], "breakpoints": [[[0, 2], [0, 1]]],
     "dots": [[1, 1]]},
    {"cols": 1, "rows": 1, "start": [0, 1], "end": [1, 0], "dots": []},
]


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


class _ProbeFollow(_Probe):
    CURSOR_SPRITE_FOLLOWS_TIP = True


class _WrapProbe(_Probe):
    """One 3x1 level whose horizontal moves off the board wrap between columns 0 and `cols`; always wins."""

    def _level_configs(self):
        return [{"cols": 3, "rows": 1, "start": [0, 1], "end": [3, 0], "dots": []}]

    def _move_target(self, board, node, dc, dr):
        col, row = node[0] + dc, node[1] + dr
        if col < 0:
            col = board.cols
        elif col > board.cols:
            col = 0
        return (col, row) if 0 <= row <= board.rows else None

    def _wins(self, board, dots, path):
        return True


# sha1 prefix of every expected frame, by name (L1/L2: probe levels; F_: cursor sprite follows the tip; W_: wrap).
FRAMES = {
    "L1_entry": "670c74ca0528b7c0",     # also level 1 redrawn with a one-node path at either start
    "L1_a": "cc5fbedbb7a310be",         # path (0,2),(1,2)
    "L1_b": "13835d7960f1cfd8",         # path (2,2),(1,2)
    "L1_c": "c5d762b4cf735040",         # path (2,2),(1,2),(0,2) = path (0,2),(1,2),(2,2)
    "L1_d": "586c485c92f75457",         # path (0,2),(1,2),(1,1)
    "L1_e": "f890d62f5e5092c2",         # path (0,2),(1,2),(1,1),(0,1)
    "L1_f": "e340744992f226af",         # path (0,2),(1,2),(1,1),(0,1),(0,0)
    "L1_g": "e53c3094ed078d48",         # path (0,2),(1,2),(1,1),(0,1),(0,0),(1,0)
    "L1_h": "7aba8726a19d2ffe",         # path (2,2),(2,1)
    "L1_i": "69e54ba088f4fdfb",         # path (0,2),(1,2),(2,2),(2,1)
    "L1_j": "4925a2428636ddd1",         # path (0,2),(1,2),(2,2),(2,1),(2,0)
    "L1_k": "bfbdacd9d0a02365",         # path (0,2),(1,2),(1,1),(1,0)
    "L1_l": "b113c558067bb4fe",         # path (0,2),(1,2),(1,1),(1,0),(2,0)
    "L1_won": "f37a54be50fd4528",       # path of L1_l in the solved color
    "L2_entry": "bf0c1e9539c1bb44",
    "L2_a": "d8526f3d9e44f3bc",         # path (0,1),(0,0)
    "L2_b": "dd8dc0557d984d57",         # path (0,1),(0,0),(1,0)
    "L2_won": "1812db1986ea48c1",       # path of L2_b in the solved color
    "F_L1_a": "7a042da614de43cb",
    "F_L1_d": "dc5a464a454e0ec7",
    "F_L1_k": "37e4478049b24374",
    "F_L1_l": "a293206f52a9aac5",
    "F_L1_won": "9b4db3eb4ad23438",
    "W_entry": "d966e8b00b35df25",
    "W_a": "dc5a38bcb038761c",          # path (0,1),(3,1): the wrap step is not drawn
    "W_b": "aa0bc759c8e1fe66",          # path (0,1),(3,1),(3,0): only the unit step is drawn
    "W_won": "4a5abe3879bf295d",
}
_NAMES = {digest: name for name, digest in FRAMES.items()}

SOLVE_LEVEL_1 = [0, 4, 1, 1, 4, 5]          # RESET, right, up, up, right, CONFIRM
SOLVE_BOTH = SOLVE_LEVEL_1 + [1, 4, 5]


def play(game: WitnessGame, actions: list[int]) -> list[FrameData]:
    return [game.perform_action(ActionInput(id=GameAction.from_id(a))) for a in actions]


def frames(results: list[FrameData]) -> str:
    """Frame names per action: `,` joins the frames of one action, ` | ` separates actions, `[]` = no frame."""
    return " | ".join(",".join(_NAMES.get(sha16(f), sha16(f)) for f in fd.frame) or "[]" for fd in results)


def test_construction_enters_level_one_at_the_first_start():
    game = _Probe()
    assert (game.game_id, game.level_index, game.win_score) == ("probe", 0, 2)
    assert game.truth() == BoardTruth(2, 2, ((0, 2),), (0, 2), ((0, 2), (2, 2)), (2, 0), dots=((1, 1),),
                                      breakpoints=frozenset({((0, 1), (0, 2))}))


def test_reset_on_a_fresh_game_is_a_full_reset():
    (fd,) = play(_Probe(), [0])
    assert frames([fd]) == "L1_entry"
    assert fd.full_reset is True
    assert (fd.levels_completed, fd.win_levels, fd.state) == (0, 2, GameState.NOT_FINISHED)
    assert fd.available_actions == [1, 2, 3, 4, 5]


def test_move_extends_the_path():
    game = _Probe()
    assert frames(play(game, [0, 4])) == "L1_entry | L1_a"
    assert game.truth().path == ((0, 2), (1, 2))


def test_move_off_the_board_leaves_frame_and_state_unchanged():
    game = _Probe()
    results = play(game, [0])
    before = game.truth()
    results += play(game, [2])
    assert frames(results) == "L1_entry | L1_entry"
    assert game.truth() == before
    (fd,) = play(game, [0])
    assert fd.full_reset is False                  # the blocked move counted as an action


def test_move_across_a_broken_edge_is_blocked():
    game = _Probe()
    results = play(game, [0, 3, 3])
    before = game.truth()
    results += play(game, [1])
    assert frames(results) == "L1_entry | L1_b | L1_c | L1_c"
    assert game.truth() == before
    assert (before.start, before.path) == ((2, 2), ((2, 2), (1, 2), (0, 2)))


def test_move_onto_the_previous_node_retracts_the_path():
    game = _Probe()
    assert frames(play(game, [0, 4, 3])) == "L1_entry | L1_a | L1_entry"
    assert game.truth().path == ((0, 2),)


def test_move_onto_an_older_path_node_leaves_the_path_unchanged():
    game = _Probe()
    assert frames(play(game, [0, 4, 1, 3, 1, 4, 2])) == "L1_entry | L1_a | L1_d | L1_e | L1_f | L1_g | L1_g"
    assert game.truth().path == ((0, 2), (1, 2), (1, 1), (0, 1), (0, 0), (1, 0))


@pytest.mark.parametrize("actions, expected, start, path", [
    ([0, 1], "L1_entry | L1_h", (2, 2), ((2, 2), (2, 1))),
    ([0, 3], "L1_entry | L1_b", (2, 2), ((2, 2), (1, 2))),
    ([0, 3, 4, 4], "L1_entry | L1_b | L1_entry | L1_a", (0, 2), ((0, 2), (1, 2))),
    ([0, 2], "L1_entry | L1_entry", (0, 2), ((0, 2),)),
], ids=["up_from_second_start", "left_from_second_start", "back_to_first_start", "no_start_can_move"])
def test_blocked_first_move_starts_from_the_first_start_that_can_make_it(actions, expected, start, path):
    game = _Probe()
    assert frames(play(game, actions)) == expected
    assert (game.truth().start, game.truth().path) == (start, path)


@pytest.mark.parametrize("actions, expected, level, start", [
    # The path misses the dot (1, 1) and stops at (2, 1).
    ([0, 1, 5], "L1_entry | L1_h | L1_entry", 0, (0, 2)),
    # The path visits the dot (1, 1) and stops on it: the rule holds, the tip is not on the end.
    ([0, 4, 1, 5], "L1_entry | L1_a | L1_d | L1_entry", 0, (0, 2)),
    # Level 2 has no dots, so the rule holds for every path.
    (SOLVE_LEVEL_1 + [1, 5], "L1_entry | L1_a | L1_d | L1_k | L1_l | L1_won,L2_entry | L2_a | L2_entry", 1, (0, 1)),
], ids=["rule_unmet", "rule_met", "rule_met_on_the_last_level"])
def test_confirm_away_from_the_end_resets_the_path_to_the_first_start(actions, expected, level, start):
    game = _Probe()
    results = play(game, actions)
    assert frames(results) == expected
    assert (game.truth().start, game.truth().path) == (start, (start,))
    assert (game.level_index, results[-1].levels_completed, results[-1].state) == \
        (level, level, GameState.NOT_FINISHED)


def test_confirm_at_the_end_with_the_rule_unmet_resets_the_path():
    game = _Probe()
    results = play(game, [0, 4, 4, 1, 1, 5])
    assert frames(results) == "L1_entry | L1_a | L1_c | L1_i | L1_j | L1_entry"
    assert game.truth().path == ((0, 2),)
    assert results[-1].levels_completed == 0


def test_winning_confirm_returns_the_solved_frame_and_the_next_level_entry():
    game = _Probe()
    results = play(game, SOLVE_LEVEL_1)
    assert frames(results) == "L1_entry | L1_a | L1_d | L1_k | L1_l | L1_won,L2_entry"
    assert (results[-1].levels_completed, results[-1].state) == (1, GameState.NOT_FINISHED)
    assert game.level_index == 1
    assert game.truth().path == ((0, 1),)


def test_reset_right_after_a_level_up_is_a_full_reset():
    game = _Probe()
    fd = play(game, SOLVE_LEVEL_1 + [0])[-1]
    assert frames([fd]) == "L1_entry"
    assert (fd.full_reset, fd.levels_completed, game.level_index) == (True, 0, 0)


def test_reset_after_an_action_on_the_level_is_a_level_reset():
    game = _Probe()
    results = play(game, SOLVE_LEVEL_1 + [1, 0])
    assert frames(results[-2:]) == "L2_a | L2_entry"
    assert (results[-1].full_reset, results[-1].levels_completed, game.level_index) == (False, 1, 1)
    assert game.truth().path == ((0, 1),)


def test_reset_right_after_a_rejected_confirm_is_a_level_reset():
    game = _Probe()
    results = play(game, [0, 4, 4, 1, 1, 5, 0])
    assert frames(results[-2:]) == "L1_entry | L1_entry"
    last = results[-1]
    assert (last.full_reset, last.levels_completed, game.level_index) == (False, 0, 0)
    assert game.truth().path == ((0, 2),)


def test_second_reset_in_a_row_is_a_full_reset():
    results = play(_Probe(), [0, 4, 0, 0])
    assert frames(results) == "L1_entry | L1_a | L1_entry | L1_entry"
    assert (results[2].full_reset, results[3].full_reset) == (False, True)


def test_winning_confirm_on_the_last_level_wins_the_game():
    results = play(_Probe(), SOLVE_BOTH)
    assert frames(results[-3:]) == "L2_a | L2_b | L2_won"
    assert (results[-1].state, results[-1].levels_completed, results[-1].win_levels) == (GameState.WIN, 2, 2)


def test_actions_after_the_win_return_no_frames():
    game = _Probe()
    play(game, SOLVE_BOTH)
    for action in (1, 5, 7):
        (fd,) = play(game, [action])
        assert frames([fd]) == "[]"
        assert (fd.state, fd.levels_completed, fd.win_levels) == (GameState.WIN, 0, 0)
        assert game.truth().path == ((0, 1), (0, 0), (1, 0))


def test_reset_after_the_win_restarts_the_game():
    game = _Probe()
    fd = play(game, SOLVE_BOTH + [0])[-1]
    assert frames([fd]) == "L1_entry"
    assert (fd.full_reset, fd.state, fd.levels_completed, game.level_index) == (True, GameState.NOT_FINISHED, 0, 0)


def test_action7_is_a_counted_no_op():
    game = _Probe()
    results = play(game, [0])
    before = game.truth()
    results += play(game, [7])
    assert game.truth() == before
    results += play(game, [0])
    assert frames(results) == "L1_entry | L1_entry | L1_entry"
    assert results[-1].full_reset is False


def test_cursor_sprite_follows_the_tip_only_when_the_class_asks_for_it():
    still, follow = _Probe(), _ProbeFollow()
    last_still, last_follow = play(still, [0, 4])[-1].frame[-1], play(follow, [0, 4])[-1].frame[-1]
    sx, sy = Grid(2, 2).node_xy((0, 2))            # the first start
    tx, ty = Grid(2, 2).node_xy((1, 2))            # the tip
    assert (last_still[sy][sx], last_follow[sy][sx]) == (CURSOR, START)
    assert (last_still[ty][tx], last_follow[ty][tx]) == (CURSOR, CURSOR)
    assert frames(play(follow, [5])) == "L1_entry"
    (cursor,) = follow.current_level.get_sprites_by_name("cursor")
    assert (cursor.x, cursor.y) == (sx, sy)
    assert frames(play(_ProbeFollow(), SOLVE_LEVEL_1)) == \
        "L1_entry | F_L1_a | F_L1_d | F_L1_k | F_L1_l | F_L1_won,L2_entry"


def test_move_target_hook_can_wrap_and_only_unit_steps_are_drawn():
    game = _WrapProbe()
    results = play(game, [0, 3])
    assert game.truth().path == ((0, 1), (3, 1))
    results += play(game, [1])
    path = game.truth().path
    assert path == ((0, 1), (3, 1), (3, 0))
    assert split_regions(3, 1, path_edges(path)) == (frozenset({(0, 0), (1, 0), (2, 0)}),)
    results += play(game, [5])
    assert frames(results) == "W_entry | W_a | W_b | W_won"
    assert (results[-1].state, results[-1].levels_completed) == (GameState.WIN, 1)


def test_truth_is_an_immutable_fresh_snapshot():
    game = _Probe()
    play(game, [0, 4])
    first, second = game.truth(), game.truth()
    assert isinstance(first, BoardTruth)
    assert isinstance(first.squares, MappingProxyType)
    with pytest.raises(TypeError):
        first.squares[(0, 0)] = 6
    with pytest.raises(FrozenInstanceError):
        first.path = ()
    assert first == second and first is not second


def test_truth_mappings_are_read_only_copies():
    given = {"squares": {(0, 0): 6}, "stars": {(0, 0): 10}, "triangles": {(0, 0): 2},
             "polyominoes": {(0, 0): Polyomino(((0, 0),), False)}}
    truth = BoardTruth(1, 1, ((0, 1),), (0, 1), ((0, 1),), (1, 0), **given)
    values = {(0, 0): None}
    markers = CellMarkers("posts", "p", values)
    mappings = {name: getattr(truth, name) for name in given} | {"values": markers.values}
    for name, mapping in mappings.items():
        assert isinstance(mapping, MappingProxyType), name
        with pytest.raises(TypeError):
            mapping[(1, 1)] = 1
    for source in (*given.values(), values):
        source[(1, 1)] = 1
    assert all((1, 1) not in getattr(truth, name) for name in given) and (1, 1) not in markers.values


def test_board_truth_fields_are_in_their_documented_order():
    assert [f.name for f in dataclasses.fields(BoardTruth)] == [
        "cols", "rows", "path", "start", "starts", "end", "dots", "squares", "stars", "triangles", "polyominoes",
        "erasers", "breakpoints", "framed_cells", "node_markers", "cell_markers", "flagged_cells", "reveal_nodes",
        "revealed_markers", "legend"]


def _level_1(drop: tuple[str, ...] = (), **keys) -> dict:
    """The first probe level without the keys in `drop` and with `keys` added or replaced."""
    return {**{k: v for k, v in LEVELS[0].items() if k not in drop}, **keys}


@pytest.mark.parametrize("configs, message", [
    ([], "probe: no levels"),
    ([_level_1(drop=("end",))], "probe level 1: missing key 'end'"),
    ([_level_1(start=[0, 2])], "probe level 1: expected exactly one of 'start' and 'starts'"),
    ([_level_1(starts=[])], "probe level 1: starts: expected a non-empty list of distinct nodes, got []"),
    ([_level_1(starts=[[0, 2], [0, 2]])],
     "probe level 1: starts: expected a non-empty list of distinct nodes, got [[0, 2], [0, 2]]"),
    ([_level_1(starts=[[0, 3], [2, 2]])], "probe level 1: starts: node (0, 3) is off the 2x2 board"),
    ([_level_1(drop=("starts",), start=[3, 2])], "probe level 1: start: node (3, 2) is off the 2x2 board"),
    ([_level_1(end=[3, 0])], "probe level 1: end: node (3, 0) is off the 2x2 board"),
    ([_level_1(breakpoints=[[[0, 0], [2, 0]]])],
     "probe level 1: breakpoints: expected two lattice-adjacent nodes, got [[0, 0], [2, 0]]"),
    ([_level_1(breakpoints=[[[2, 2], [3, 2]]])], "probe level 1: breakpoints: node (3, 2) is off the 2x2 board"),
    ([_level_1(cols=10)], "probe level 1: board 10x2 is outside 1..9"),
    ([_level_1(cols="2")], "probe level 1: board '2'x2 is outside 1..9"),
    ([_level_1(drop=("dots",))], "probe level 1: missing key 'dots'"),
    ([_level_1(dotz=[])], "probe level 1: unknown config key 'dotz'"),
    ([_level_1(drop=("end",), dotz=[])], "probe level 1: unknown config key 'dotz'"),
    ([_level_1(zz=[], aa=[])], "probe level 1: unknown config key 'aa'"),
    ([LEVELS[0], {**LEVELS[1], "end": [2, 0]}], "probe level 2: end: node (2, 0) is off the 1x1 board"),
], ids=["no_levels", "no_end", "start_and_starts", "empty_starts", "duplicate_starts", "a_start_off_board",
        "start_off_board", "end_off_board", "long_breakpoint", "breakpoint_off_board", "board_too_wide",
        "cols_not_int", "no_dots", "unknown_key", "unknown_key_checked_first", "smallest_unknown_key_named",
        "second_level"])
def test_malformed_level_configs_fail_construction_naming_game_and_level(configs, message):
    game_class = type("_Malformed", (_Probe,), {"_level_configs": lambda self: configs})
    with pytest.raises(ValueError) as exc:
        game_class()
    assert str(exc.value) == message


def test_only_reset_levels_true_refuses_construction(monkeypatch):
    monkeypatch.setenv("ONLY_RESET_LEVELS", "true")
    with pytest.raises(RuntimeError, match="ONLY_RESET_LEVELS"):
        _Probe()
    monkeypatch.setenv("ONLY_RESET_LEVELS", "false")
    _Probe()
    monkeypatch.delenv("ONLY_RESET_LEVELS")
    _Probe()


def test_make_game_refuses_to_build_a_game_while_only_reset_levels_is_true(monkeypatch):
    monkeypatch.setenv("ONLY_RESET_LEVELS", "true")
    with pytest.raises(RuntimeError, match="ONLY_RESET_LEVELS"):
        make_game("tw01")
    monkeypatch.delenv("ONLY_RESET_LEVELS")
    assert make_game("tw01").game_id == "tw01"


def test_unknown_game_id_raises_key_error_listing_every_id(monkeypatch):
    def no_import(name, *args, **kwargs):
        raise AssertionError(f"import of {name} attempted")

    monkeypatch.setattr("importlib.import_module", no_import)
    with pytest.raises(KeyError) as exc:
        make_game("no_such_game")
    message = exc.value.args[0]
    assert "no_such_game" in message
    assert all(game_id in message for game_id in GAME_IDS)


def test_config_helpers():
    for bad in ([1, True], [1, 2, 3]):
        with pytest.raises(ValueError, match=r"^end: "):
            parse_coord(bad, "end")
    assert parse_coords([[0, 1], [2, 3]], "dots") == ((0, 1), (2, 3))
    for bad in ("1,2,3", " 1,2", "1, 2", "+1,2"):
        with pytest.raises(ValueError, match=r"^squares: "):
            parse_cell_key(bad, "squares")
    assert parse_cell_key("1,2", "squares") == (1, 2)
    assert parse_cell_key("-1,02", "squares") == (-1, 2)
    with pytest.raises(ValueError, match=r"^squares: cell 1,2 is listed twice$"):
        parse_cell_map({"1,2": 1, "01,2": 2}, "squares")
    squares = parse_int_map({"0,1": 6, "1,0": 10}, "squares")
    assert squares == {(0, 1): 6, (1, 0): 10} and list(squares) == [(0, 1), (1, 0)]
    with pytest.raises(ValueError) as exc:
        parse_int_map({"0,0": True}, "squares")
    assert all(s in str(exc.value) for s in ("squares", "0,0", "True"))
    with pytest.raises(ValueError) as exc:
        parse_int_map({"0,0": 4}, "triangles", allowed={1, 2, 3})
    assert all(s in str(exc.value) for s in ("triangles", "0,0", "4"))
    board6 = Board.from_config({"cols": 6, "rows": 6, "start": [0, 0], "end": [6, 6]})
    board5 = Board.from_config({"cols": 5, "rows": 5, "start": [0, 0], "end": [5, 5]})
    with pytest.raises(ValueError, match=r"6x6: cell size 8 "):
        require_cell_size(board6, 9)
    assert require_cell_size(board5, 9) is None
    assert require_cell_size(board5, 10) is None                    # cell size 10: the minimum itself is enough


def test_a_level_file_must_hold_a_non_empty_array_of_level_objects(tmp_path, monkeypatch):
    monkeypatch.setattr("importlib.resources.files", lambda package: tmp_path)
    (tmp_path / "levels").mkdir()
    texts = {"empty": "[]", "numbers": "[1]", "object": '{"cols": 2}', "broken": "[{", "one": '[{"cols": 2}]'}
    for name, text in texts.items():
        (tmp_path / "levels" / f"{name}.json").write_text(text, encoding="utf-8")
    for name in ("empty", "numbers", "object"):
        with pytest.raises(ValueError) as exc:
            load_level_configs(name)
        assert str(exc.value) == f"{name}.json: expected a non-empty JSON array of level objects"
    with pytest.raises(ValueError, match=r"^broken\.json: not valid JSON: "):
        load_level_configs("broken")
    with pytest.raises(FileNotFoundError):
        load_level_configs("missing")
    assert load_level_configs("one") == ({"cols": 2},)


def test_every_game_has_a_compact_level_file_with_its_level_count_and_known_keys():
    folder = files("witness_env") / "levels"
    assert sorted(p.name for p in folder.iterdir()) == \
        sorted(f"{game_id}.json" for game_id in GAME_IDS + TRAIN_R1_IDS + TRAIN_R2_IDS)
    for game_id in GAME_IDS:
        configs = load_level_configs(game_id)
        assert len(configs) == LEVEL_COUNTS[game_id], game_id
        text = "[\n" + ",\n".join(json.dumps(c, separators=(",", ":"), ensure_ascii=True) for c in configs) + "\n]\n"
        assert (folder / f"{game_id}.json").read_bytes() == text.encode("ascii"), game_id
        structural = {"cols", "rows", "start", "end"} if game_id in ("tw09", "tw10") else STRUCTURAL_KEYS
        for i, config in enumerate(configs):
            assert set(config) <= structural | MARKER_KEYS[game_id], (game_id, i)
            assert all(config[k] not in ([], {}) for k in set(config) & MARKER_KEYS[game_id]), (game_id, i)
    assert sum(LEVEL_COUNTS.values()) == 260


def test_registry_builds_a_new_independent_game_of_its_own_class_for_every_id():
    assert GAME_IDS == ("tw01", "tw02", "tw03", "tw05", "tw06", "tw07", "tw08", "tw09", "tw10", "tw11", "tw12", "tw13",
                        "ft01", "ft02", "ft03", "ft04", "ft05", "ft06", "ft10", "ft11", "ft12", "ft13", "ft14", "ft15",
                        "ft16", "ft20", "fv01", "fv02", "fv03", "fv04")
    for game_id in GAME_IDS:
        game, other = make_game(game_id), make_game(game_id)
        cls = type(game)
        assert cls.__name__ == game_id.capitalize() and cls.GAME_ID == game.game_id == game_id
        structural = {"cols", "rows", "start", "end"} if game_id in ("tw09", "tw10") else STRUCTURAL_KEYS
        assert cls.CONFIG_KEYS == structural | MARKER_KEYS[game_id], game_id
        assert game.win_score == LEVEL_COUNTS[game_id], game_id
        truth = game.truth()
        assert truth.path == (truth.starts[0],), game_id
        assert other is not game and type(other) is cls
        for move in (1, 2, 3, 4):                              # extend the path of one game by one node
            play(game, [0, move])
            if len(game.truth().path) == 2:
                break
        assert len(game.truth().path) == 2 and other.truth() == truth, game_id


def test_docs_rule_line_of_every_game_equals_its_module_docstring():
    """The Rules section of docs/games.md lists one rule per game in registry order, each equal to the game module's
    docstring up to whitespace."""
    text = (Path(__file__).resolve().parent.parent / "docs" / "games.md").read_text(encoding="utf-8")
    section = text[text.index("\n## Rules\n"):]
    end = section.find("\n## ", 1)
    section = section if end < 0 else section[:end]
    rules = re.findall(r"^- `(\w+)`: (.+?)(?=\n- `|\n\n|\Z)", section, re.M | re.S)
    assert [game_id for game_id, _ in rules] == list(GAME_IDS)
    for game_id, rule in rules:
        doc = importlib.import_module(f"witness_env.games.{game_id}").__doc__
        assert " ".join(rule.split()) == " ".join(doc.split()), game_id
