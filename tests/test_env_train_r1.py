"""The revised training games (`witness_env.TRAIN_R1_IDS`, docs/training.md): tw02r1, tw03r1, tw10r1 and ft03r1 are
their base game with a few levels replaced; ft06r1 is ft06 with the value-3 buoy drawn inside its cell on 4x4 boards.

Golden fixture used: `env/solutions_r1.json` (a solving action sequence per level; on a replaced level the shortest
one, found by breadth-first search over the engine's moves). The path counts of the replaced levels are enumerated
here with `golden.count_paths`; tw02r1 level 10 alone takes about a minute.
"""
from __future__ import annotations

import json
from importlib.resources import files

import numpy as np
import pytest
from arcengine import ActionInput, FrameData, GameAction

import golden
from witness_env import GAME_IDS, TRAIN_R1_IDS, WitnessGame, make_game
from witness_env.base import Board, load_level_configs
from witness_env.grid import GREEN

RESET, CONFIRM = 0, 5
# The levels (counted from 0) each revised game replaces, and their (paths, accepted) counts.
REPLACED = {"tw02r1": {6: (383, 62), 9: (1262816, 32394)}, "tw03r1": {1: (8, 2)}, "tw10r1": {1: (178, 46)},
            "ft03r1": {2: (178, 21), 3: (178, 18)}, "ft06r1": {}}


def act(game: WitnessGame, action: int) -> FrameData:
    return game.perform_action(ActionInput(id=GameAction.from_id(action)))


def test_the_revised_ids_are_registered_outside_the_thirty_public_games():
    assert TRAIN_R1_IDS == tuple(REPLACED) == golden.TRAIN_R1_GAMES
    assert not set(TRAIN_R1_IDS) & set(GAME_IDS)
    for game_id in TRAIN_R1_IDS:
        game = make_game(game_id)
        cls, base = type(game), type(make_game(game_id[:-2]))
        assert cls.__name__ == game_id.capitalize() and cls.GAME_ID == game.game_id == game_id
        assert cls.__bases__ == (base,) and cls.CONFIG_KEYS == base.CONFIG_KEYS and cls._wins is base._wins
        assert game.win_score == golden.LEVEL_COUNTS[game_id[:-2]]


@pytest.mark.parametrize("game_id", TRAIN_R1_IDS)
def test_the_level_file_is_the_base_games_with_only_the_replaced_levels_changed(game_id):
    configs, base = load_level_configs(game_id), load_level_configs(game_id[:-2])
    assert len(configs) == len(base)
    assert [i for i, (a, b) in enumerate(zip(configs, base)) if a != b] == sorted(REPLACED[game_id])
    text = "[\n" + ",\n".join(json.dumps(c, separators=(",", ":"), ensure_ascii=True) for c in configs) + "\n]\n"
    assert (files("witness_env") / "levels" / f"{game_id}.json").read_bytes() == text.encode("ascii")
    assert all(set(c) <= type(make_game(game_id)).CONFIG_KEYS for c in configs)


@pytest.mark.parametrize("game_id", TRAIN_R1_IDS)
def test_stored_solutions_played_in_a_row_win_every_level(game_id):
    solutions = golden.STORE.solutions_r1()[game_id]
    assert len(solutions) == len(load_level_configs(game_id))
    game = make_game(game_id)
    act(game, RESET)
    for level, solution in enumerate(solutions):
        results = [act(game, a) for a in solution]
        assert [fd.levels_completed for fd in results] == [level] * (len(solution) - 1) + [level + 1], level
    assert results[-1].state.name == "WIN"


@pytest.mark.parametrize("game_id", TRAIN_R1_IDS)
def test_each_stored_solution_wins_its_level_on_its_last_action_only(game_id):
    for level, solution in enumerate(golden.STORE.solutions_r1()[game_id]):
        game = make_game(game_id)
        game.set_level(level)
        completed = [act(game, a).levels_completed for a in solution]
        assert completed == [0] * (len(solution) - 1) + [1], level


@pytest.mark.parametrize("game_id", TRAIN_R1_IDS)
def test_unreplaced_levels_keep_the_base_games_stored_solutions(game_id):
    mine, base = golden.STORE.solutions_r1()[game_id], golden.STORE.solutions()[game_id[:-2]]
    assert [i for i, (a, b) in enumerate(zip(mine, base)) if a != b] == \
        [i for i in sorted(REPLACED[game_id]) if mine[i] != base[i]]
    assert all(mine[i] == base[i] for i in range(len(mine)) if i not in REPLACED[game_id])


@pytest.mark.parametrize(("game_id", "level"), [(g, i) for g in TRAIN_R1_IDS for i in sorted(REPLACED[g])])
def test_a_replaced_level_accepts_well_under_half_of_its_paths(game_id, level):
    config = load_level_configs(game_id)[level]
    game = make_game(game_id)
    board = Board.from_config(config)
    paths, accepted = golden.count_paths(game, board, type(game)._parse_markers(config, board))
    assert (paths, accepted) == REPLACED[game_id][level]
    assert 0 < accepted and accepted / paths < 0.3


@pytest.mark.parametrize(("game_id", "level"), [(g, i) for g in TRAIN_R1_IDS for i in sorted(REPLACED[g])])
def test_the_stored_solution_of_a_replaced_level_is_a_shortest_one(game_id, level):
    """No action sequence one action shorter wins: breadth-first search over the engine's own moves (CONFIRM is
    only ever tried at the end node, and only at the depth of the stored solution minus one)."""
    solution = golden.STORE.solutions_r1()[game_id][level]
    game = make_game(game_id)
    config = load_level_configs(game_id)[level]
    board = Board.from_config(config)
    markers = type(game)._parse_markers(config, board)
    frontier, seen = {(board.starts[0],)}, set()
    for _ in range(len(solution) - 2):                         # every path reachable in fewer moves than stored
        seen |= frontier
        nxt = set()
        for path in frontier:
            for dc, dr in ((0, -1), (0, 1), (-1, 0), (1, 0)):
                target = game._move_target(board, path[-1], dc, dr)
                if target is None:
                    continue
                if len(path) >= 2 and target == path[-2]:
                    nxt.add(path[:-1])
                elif target not in path:
                    nxt.add(path + (target,))
        frontier = nxt
    seen |= frontier
    assert not any(p[-1] == board.end and type(game)._wins(board, markers, p) for p in seen)
    assert solution[-1] == CONFIRM


def _play_frames(game_id: str, level: int, actions: list[int]) -> list[np.ndarray]:
    game = make_game(game_id)
    game.set_level(level)
    frames = [np.asarray(game.camera.render(game.current_level.get_sprites()), dtype=np.int8)]
    for a in actions[:-1]:                                     # the CONFIRM moves on to the next level
        frames.append(np.asarray(act(game, a).frame[-1], dtype=np.int8))
    return frames


@pytest.mark.parametrize("level", range(10))
def test_ft06r1_frames_differ_from_ft06_only_at_value_3_buoys_on_4x4_boards(level):
    config = load_level_configs("ft06")[level]
    board = Board.from_config(config)
    grid = board.grid
    threes = [tuple(map(int, k.split(","))) for k, v in config["buoys"].items() if v == 3]
    actions = golden.STORE.solutions_r1()["ft06r1"][level]
    old, new = _play_frames("ft06", level, actions), _play_frames("ft06r1", level, actions)
    for step, (a, b) in enumerate(zip(old, new)):
        ys, xs = np.nonzero(a != b)
        diff = set(zip(xs.tolist(), ys.tolist()))
        if (board.cols, board.rows) != (4, 4):
            assert not diff, (level, step)
            continue
        assert diff, (level, step)
        allowed, inside = set(), set()
        for cell in threes:
            cx, cy = grid.cell_center(cell)
            x0, y0 = grid.node_xy(cell)
            inside |= {(x, y) for x in range(x0 + 1, x0 + 1 + grid.cell_size)
                       for y in range(y0 + 1, y0 + 1 + grid.cell_size)}
            allowed |= {(x, y) for x in range(cx - 8, cx + 9) for y in range(cy - 2, cy + 3)}   # ft06's 17x5 glyph
        assert diff <= allowed | inside, (level, step)
        assert {(x, y) for x, y in diff if b[y, x] == GREEN} <= inside, (level, step)
