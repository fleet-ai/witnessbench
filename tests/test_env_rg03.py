"""The r2 training game rg03 (`witness_env.games.rg03`, ingots and fulcrums: torque balance): level file, config
checks, frames, reported markers, stored solutions and the rule, including parity with an independent reference
implementation of the rule on every path of every level.

The frozen data below was recorded with tools that are not part of this repository.
`SOLUTIONS` is a shortest accepted path of each level (breadth-first search with ties broken by the smallest action
list; moves up, down, left, right, then CONFIRM). `REFERENCE_CASES` are the reference implementation's examples with
its verdicts. `REFERENCE_PARITY` holds, per level, its own count of simple start-to-end paths (a path stops at its
first arrival at the end), how many it accepts, and the sha1 of the accepted paths written as move letters U/D/L/R,
sorted and joined by newlines. This module enumerates the same paths with `_move_target`, judges them with
`Rg03._wins` and must reproduce all three. Entry frames and marker layers are compared through `golden.sha16`.
"""
from __future__ import annotations

import hashlib
import importlib.resources
import json

import numpy as np
import pytest
from arcengine import ActionInput, FrameData, GameAction

from golden import sha16
from witness_env import WitnessGame
from witness_env.base import Board, load_level_configs
from witness_env.games.rg03 import Rg03
from witness_env.grid import BLACK, CURSOR, MAROON, START

GAME_ID = "rg03"
LEVELS = 10
RESET, UP, DOWN, LEFT, RIGHT, CONFIRM = 0, 1, 2, 3, 4, 5
MOVES = {(0, -1): "U", (0, 1): "D", (-1, 0): "L", (1, 0): "R"}

SOLUTIONS = (
    (2, 2, 4, 4, 2, 5),
    (2, 3, 2, 3, 3, 1, 1, 4, 5),
    (2, 2, 4, 1, 4, 1, 4, 1, 3, 5),
    (4, 1, 1, 3, 1, 4, 1, 3, 3, 2, 2, 2, 3, 3, 2, 5),
    (2, 2, 2, 3, 3, 3, 1, 3, 1, 1, 1, 4, 4, 5),
    (4, 4, 1, 3, 3, 1, 1, 3, 3, 5),
    (2, 2, 2, 4, 1, 4, 5),
    (1, 1, 3, 1, 3, 3, 3, 2, 2, 5),
    (4, 4, 1, 1, 3, 1, 1, 3, 3, 2, 4, 2, 2, 3, 1, 3, 5),
    (1, 3, 1, 3, 2, 2, 3, 2, 3, 2, 4, 4, 4, 4, 5),
)
REFERENCE_PARITY = (   # (paths, accepted, sha1 of the sorted accepted move strings) per level, level 1 first
    (146, 28, "90968c7c5eadeb16d071d7d6202da27645269067"),
    (145, 8, "ee702589d895764e6855e4c246703203f24e8d55"),
    (117, 7, "c5eb79611d9619128bd037d954157e22c005fa1d"),
    (7071, 139, "f9b7cc34ad3188f50c88f04b420f0c63e78a8013"),
    (4793, 92, "3cf4a83edda35e9a7d0e4c0941dd8c2dce1accf6"),
    (4732, 150, "17fc90ff51fd351b622353cc0ded1868b658ee29"),
    (4272, 64, "0ff05ca15d715ea68f07d9ea6b63f13c112dda8f"),
    (6672, 30, "f4e6a4940414ee008cbddc76ad71f0070b9b9cb8"),
    (4272, 8, "2fa0c02e46d73c7d5c3f7ccc0de34bac39878ee4"),
    (5933, 28, "9b624b567f919dc2bdff4028f3d5b4c40f046d2e"),
)
ENTRY_FRAMES = ("b71d49ee4508b716", "e394523cb8e10c59", "e5d1b602b7874fda", "bdc3973e10cb7400", "2cb2b2e5e915c438",
                "384fb832cc8541d0", "326f9c38075dca7b", "952ed9eccdd9108a", "c2f49afa6a364b72", "a9e0941a0bb87d3f")
MARKER_LAYERS = (   # (config that is not a level, sha16 of its marker layer alone)
    ({"cols": 3, "rows": 3, "start": [0, 3], "end": [3, 0], "ingots": {"0,0": 1, "1,1": 2, "2,2": 3},
      "fulcrums": [[2, 0], [0, 2]]}, "2ae9ff2cfd7ed5d2"),
    ({"cols": 5, "rows": 5, "start": [0, 5], "end": [5, 0], "ingots": {"0,0": 3, "4,4": 1}, "fulcrums": [[2, 2]]},
     "a56c5a8e08e489af"),
    ({"cols": 2, "rows": 4, "start": [0, 0], "end": [2, 4], "ingots": {}, "fulcrums": [[1, 3]]}, "68973aa876bf2afc"),
)
BEAM_PIXELS = ((-2, -2), (-1, -2), (0, -2), (1, -2), (2, -2))
PIP_PIXELS = {1: ((0, 0),), 2: ((-1, 0), (1, 0)), 3: ((-2, 0), (0, 0), (2, 0))}
FULCRUM_PIXELS = ((0, -2), (0, -1), (0, 0), (0, 1), (-2, 2), (-1, 2), (0, 2), (1, 2), (2, 2))

_BOARD_S = {"cols": 3, "rows": 3, "start": [0, 1], "end": [3, 1],
            "ingots": {"0,0": 1, "2,0": 1, "0,2": 1, "2,2": 1}, "fulcrums": [[1, 0], [1, 2]]}
_BOARD_P = {"cols": 3, "rows": 3, "start": [0, 1], "end": [3, 1],
            "ingots": {"0,0": 2, "2,0": 1, "2,1": 1, "0,2": 1, "2,2": 1}, "fulcrums": [[1, 0], [1, 2]]}
_BOARD_L = {"cols": 4, "rows": 4, "start": [0, 2], "end": [4, 2],
            "ingots": {"0,0": 2, "3,0": 1, "1,3": 1, "3,3": 1}, "fulcrums": [[1, 0], [2, 3]]}
REFERENCE_CASES = (   # (id, config, path, accepted): the reference implementation's examples
    ("A1-straight-cut-two-balanced-pairs", _BOARD_S, ((0, 1), (1, 1), (2, 1), (3, 1)), True),
    ("A2-two-pips-balance-one-plus-one", _BOARD_P, ((0, 1), (0, 2), (1, 2), (2, 2), (3, 2), (3, 1)), True),
    ("A3-lever-law-two-at-one-balances-one-at-two", _BOARD_L, ((0, 2), (1, 2), (2, 2), (3, 2), (4, 2)), True),
    ("R1-border-hug-crowds-both-fulcrums", _BOARD_S, ((0, 1), (0, 0), (1, 0), (2, 0), (3, 0), (3, 1)), False),
    ("R2-fulcrum-walled-in-with-one-ingot", _BOARD_S, ((0, 1), (0, 0), (1, 0), (1, 1), (2, 1), (3, 1)), False),
    ("R3-pips-are-weights", _BOARD_P, ((0, 1), (1, 1), (2, 1), (3, 1)), False),
    ("R4-arm-length-counts", _BOARD_L,
     ((0, 2), (0, 3), (0, 4), (1, 4), (2, 4), (2, 3), (3, 3), (3, 4), (4, 4), (4, 3), (4, 2)), False),
)


def act(game: WitnessGame, action: int) -> FrameData:
    return game.perform_action(ActionInput(id=GameAction.from_id(action)))


def entry_frame(level: int) -> np.ndarray:
    game = Rg03()
    game.set_level(level)
    return np.asarray(game.camera.render(game.current_level.get_sprites()), dtype=np.int8)


def wins(config: dict, path: tuple) -> bool:
    board = Board.from_config(config)
    return Rg03._wins(board, Rg03._parse_markers(config, board), tuple(path))


# --------------------------------------------------------------------------------------- level file, configs
def test_reset_reports_the_level_count_and_the_first_entry_frame():
    fd = act(Rg03(), RESET)
    assert fd.win_levels == LEVELS
    assert [sha16(f) for f in fd.frame] == [ENTRY_FRAMES[0]]


def test_level_file_holds_ten_distinct_compact_levels_with_exactly_the_ingot_and_fulcrum_keys():
    configs = load_level_configs(GAME_ID)
    assert len(configs) == LEVELS
    assert all(set(c) == {"cols", "rows", "start", "end", "ingots", "fulcrums"} for c in configs)
    assert len({json.dumps(c, sort_keys=True) for c in configs}) == LEVELS
    raw = (importlib.resources.files("witness_env") / "levels" / f"{GAME_ID}.json").read_text(encoding="utf-8")
    assert raw == "[\n" + ",\n".join(json.dumps(c, separators=(",", ":"), ensure_ascii=True) for c in configs) + "\n]\n"


def _first_level_class(mutate) -> type:
    config = json.loads(json.dumps(load_level_configs(GAME_ID)[0]))
    mutate(config)
    return type("_MutatedRg03", (Rg03,), {"_level_configs": lambda self: [config]})


CONFIG_ERRORS = [   # (id, mutation of level 1's config, error message)
    ("missing-ingots", lambda c: c.pop("ingots"), "rg03 level 1: missing key 'ingots'"),
    ("missing-fulcrums", lambda c: c.pop("fulcrums"), "rg03 level 1: missing key 'fulcrums'"),
    ("unknown-key", lambda c: c.__setitem__("ingot", {}), "rg03 level 1: unknown config key 'ingot'"),
    ("board-too-large-for-the-glyphs", lambda c: c.update(cols=6, rows=6),
     "rg03 level 1: board 6x6: cell size 8 is below 9"),
    *[(f"weight-{v}", lambda c, v=v: c.update(ingots={"0,0": v}),
       f"rg03 level 1: ingots: cell 0,0 has value {v}, expected one of [1, 2, 3]") for v in (0, 4)],
    ("boolean-weight", lambda c: c.update(ingots={"0,0": True}),
     "rg03 level 1: ingots: cell 0,0 has value True, expected an int"),
    ("ingot-on-a-fulcrum", lambda c: c.update(ingots={"1,0": 1}),
     "rg03 level 1: ingots and fulcrums: cell 1,0 is listed twice"),
    ("one-fulcrum-twice", lambda c: c.update(fulcrums=[[2, 2], [2, 2]]),
     "rg03 level 1: ingots and fulcrums: cell 2,2 is listed twice"),
    ("fulcrums-not-a-list", lambda c: c.__setitem__("fulcrums", {"0,0": 1}),
     "rg03 level 1: fulcrums: expected a list of [col, row] pairs, got {'0,0': 1}"),
]


@pytest.mark.parametrize("mutate, message", [c[1:] for c in CONFIG_ERRORS], ids=[c[0] for c in CONFIG_ERRORS])
def test_malformed_level_config_fails_construction_with_its_message(mutate, message):
    with pytest.raises(ValueError) as exc:
        _first_level_class(mutate)()
    assert str(exc.value) == message


def test_unmodified_first_level_config_constructs():
    assert _first_level_class(lambda c: None)().win_score == 1


# ----------------------------------------------------------------------------------------------------- frames
def test_entry_frames_match_the_frozen_entry_frames():
    assert tuple(sha16(entry_frame(level)) for level in range(LEVELS)) == ENTRY_FRAMES


def test_failed_confirm_at_the_start_re_renders_the_entry_frame():
    for level, digest in enumerate(ENTRY_FRAMES):
        game = Rg03()
        game.set_level(level)
        fd = act(game, CONFIRM)
        assert [sha16(f) for f in fd.frame] == [digest], level
        assert (fd.levels_completed, game.level_index) == (0, level)


def test_marker_layers_of_non_level_configs_match_their_frozen_hashes():
    for i, (config, digest) in enumerate(MARKER_LAYERS):
        board = Board.from_config(config)
        frame = board.grid.new_frame()
        Rg03._draw_markers(frame, board, Rg03._parse_markers(config, board), frozenset())
        assert sha16(frame) == digest, i


def test_the_marker_layer_paints_a_maroon_beam_with_k_pips_per_ingot_and_a_black_t_per_fulcrum_and_nothing_else():
    config = MARKER_LAYERS[0][0]
    board = Board.from_config(config)
    blank = board.grid.new_frame()
    frame = board.grid.new_frame()
    Rg03._draw_markers(frame, board, Rg03._parse_markers(config, board), frozenset())
    expected = {}
    for key, k in config["ingots"].items():
        x, y = board.grid.cell_center(tuple(int(n) for n in key.split(",")))
        expected.update({(x + dx, y + dy): MAROON for dx, dy in BEAM_PIXELS + PIP_PIXELS[k]})
    for c, r in config["fulcrums"]:
        x, y = board.grid.cell_center((c, r))
        expected.update({(x + dx, y + dy): BLACK for dx, dy in FULCRUM_PIXELS})
    changed = {(x, y): frame[y][x] for y in range(len(frame)) for x in range(len(frame[0]))
               if frame[y][x] != blank[y][x]}
    assert changed == expected


def test_cursor_sprite_follows_the_path_tip():
    assert Rg03.CURSOR_SPRITE_FOLLOWS_TIP is True
    game = Rg03()
    act(game, RESET)
    frame = np.asarray(act(game, SOLUTIONS[0][0]).frame[0])
    truth = game.truth()
    assert len(truth.path) == 2
    grid = Board.from_config(load_level_configs(GAME_ID)[0]).grid
    (tip_x, tip_y), (start_x, start_y) = grid.node_xy(truth.path[-1]), grid.node_xy(truth.start)
    assert (frame[tip_y, tip_x], frame[start_y, start_x]) == (CURSOR, START)


# ------------------------------------------------------------------------------------------- reported markers
def test_truth_of_every_level_declares_the_ingots_and_fulcrums_of_its_level_config():
    game = Rg03()
    for level, config in enumerate(load_level_configs(GAME_ID)):
        game.set_level(level)
        truth = game.truth()
        assert [(m.name, m.glyph, m.values_are_colours) for m in truth.cell_markers] == [
            ("ingots", "i", False), ("fulcrums", "l", False)], level
        assert dict(truth.cell_markers[0].values) == {
            tuple(int(n) for n in key.split(",")): k for key, k in config["ingots"].items()}, level
        assert dict(truth.cell_markers[1].values) == {(c, r): None for c, r in config["fulcrums"]}, level
        assert (truth.dots, dict(truth.squares), dict(truth.stars), dict(truth.triangles), dict(truth.polyominoes),
                truth.erasers, truth.framed_cells, truth.node_markers) == ((), {}, {}, {}, {}, frozenset(),
                                                                           frozenset(), ()), level
        assert truth.starts == (tuple(config["start"]),), level


# ------------------------------------------------------------------------------------------ stored solutions
def test_stored_solutions_played_in_a_row_win_every_level_and_enter_the_next():
    game = Rg03()
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
        game = Rg03()
        game.set_level(level)
        results = [act(game, a) for a in solution]
        assert [fd.levels_completed for fd in results] == [0] * (len(solution) - 1) + [1], level


def test_no_stored_solution_wins_another_level():
    for level, solution in enumerate(SOLUTIONS):
        for other in range(LEVELS):
            if other != level:
                game = Rg03()
                game.set_level(other)
                assert [act(game, a).levels_completed for a in solution][-1] == 0, (level, other)


# ------------------------------------------------------------------------------------------------------ rules
@pytest.mark.parametrize("config, path, accepted", [c[1:] for c in REFERENCE_CASES],
                         ids=[c[0] for c in REFERENCE_CASES])
def test_the_reference_examples(config, path, accepted):
    assert wins(config, path) is accepted


_ROW = ((0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0))   # the top border of a 5x1 board: one region
RULE_CASES = [   # (id, cols, rows, path, markers, expected), derived by hand (the reference agrees on each)
    # Lever law along one row: 3 x (-1) + 1 x (+3) = 0.
    ("three-at-arm-one-balances-one-at-arm-three", 5, 1, _ROW, {"ingots": {"0,0": 3, "4,0": 1}, "fulcrums": [[1, 0]]},
     True),
    ("swapped-weights-tip-the-lever", 5, 1, _ROW, {"ingots": {"0,0": 1, "4,0": 3}, "fulcrums": [[1, 0]]}, False),
    # Rows never matter: -1 on the top row balances +1 on the bottom row.
    ("rows-do-not-matter", 3, 2, ((0, 0), (1, 0), (2, 0), (3, 0)),
     {"ingots": {"0,0": 1, "2,1": 1}, "fulcrums": [[1, 1]]}, True),
    # An ingot in the fulcrum's own column adds nothing, whatever its weight.
    ("an-ingot-in-the-fulcrum-column-adds-nothing", 1, 3, ((0, 0), (0, 1), (0, 2), (0, 3)),
     {"ingots": {"0,2": 3}, "fulcrums": [[0, 0]]}, True),
    # The cut down column 1 gives each bare fulcrum its own region: both balance.
    ("bare-fulcrums-apart-balance", 2, 1, ((1, 0), (1, 1)), {"ingots": {}, "fulcrums": [[0, 0], [1, 0]]}, True),
    # Without the cut the two bare fulcrums share a region.
    ("bare-fulcrums-together-are-crowded", 2, 1, ((0, 0), (1, 0), (2, 0)),
     {"ingots": {}, "fulcrums": [[0, 0], [1, 0]]}, False),
    # The cut leaves the ingot in a region with no fulcrum.
    ("an-ingot-without-a-fulcrum", 2, 1, ((1, 0), (1, 1)), {"ingots": {"1,0": 1}, "fulcrums": [[0, 0]]}, False),
    # Two balanced ingots, but their region holds two fulcrums.
    ("balanced-but-crowded", 4, 1, ((0, 0), (1, 0), (2, 0), (3, 0), (4, 0)),
     {"ingots": {"0,0": 1, "2,0": 1}, "fulcrums": [[1, 0], [3, 0]]}, False),
]


@pytest.mark.parametrize("cols, rows, path, markers, expected", [c[1:] for c in RULE_CASES],
                         ids=[c[0] for c in RULE_CASES])
def test_rule_outcome_on_a_small_board(cols, rows, path, markers, expected):
    config = markers | {"cols": cols, "rows": rows, "start": list(path[0]), "end": list(path[-1])}
    assert wins(config, path) is expected


def _accepted_moves(game: Rg03, config: dict) -> tuple[int, list[str]]:
    """(number of simple start-to-end paths, the accepted ones as move strings); a path stops at the end node."""
    board = Board.from_config(config)
    markers = Rg03._parse_markers(config, board)
    end, count, accepted = board.end, 0, []
    path, on_path = [board.starts[0]], {board.starts[0]}

    def extend() -> None:
        nonlocal count
        tip = path[-1]
        if tip == end:
            count += 1
            if Rg03._wins(board, markers, tuple(path)):
                accepted.append("".join(MOVES[(b[0] - a[0], b[1] - a[1])] for a, b in zip(path, path[1:])))
            return
        for dc, dr in MOVES:
            target = game._move_target(board, tip, dc, dr)
            if target is not None and target not in on_path:
                path.append(target)
                on_path.add(target)
                extend()
                on_path.discard(path.pop())

    extend()
    return count, accepted


@pytest.mark.parametrize("level", range(LEVELS))
def test_the_rule_and_the_reference_agree_on_every_simple_path_of_every_level(level):
    count, accepted = _accepted_moves(Rg03(), load_level_configs(GAME_ID)[level])
    digest = hashlib.sha1("\n".join(sorted(accepted)).encode("ascii")).hexdigest()
    assert (count, len(accepted), digest) == REFERENCE_PARITY[level]
