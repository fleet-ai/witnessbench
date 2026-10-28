"""The training-only games of training set r2 (`witness_env.TRAIN_R2_IDS`): new games with region rules, outside the
30 public games, played by the rows of data/train_r2.jsonl (the 21 rows of data/train_r1.jsonl, then one row per r2
game).

Golden fixture used: `env/solutions_r2.json` (the shortest solving action sequence of each level, found by
breadth-first search over the engine's moves). The path counts of every level are enumerated here with
`golden.count_paths`; the four 5x5 levels (rg04 and rg06 L9-L10, 1,262,816 paths each) only with
WITNESS_SLOW_TESTS=1, the frozen counts of every level are checked in every run.
"""
from __future__ import annotations

import json
import os
from importlib.resources import files
from pathlib import Path

import pytest
from arcengine import ActionInput, FrameData, GameAction

import golden
from witness_env import GAME_IDS, TRAIN_R1_IDS, TRAIN_R2_IDS, WitnessGame, make_game
from witness_env.base import Board, load_level_configs

RESET, CONFIRM = 0, 5
SLOW = os.environ.get("WITNESS_SLOW_TESTS") == "1"
ROOT = Path(__file__).resolve().parent.parent
LEVELS = 10
# The (paths, accepted) counts of each level of each r2 game, level 1 first.
PATH_COUNTS: dict[str, tuple[tuple[int, int], ...]] = {
    "rg01": ((178, 74), (178, 39), (178, 44), (3762, 162), (4178, 24), (4866, 69), (4914, 107), (4178, 48),
             (5866, 27), (4866, 29)),
    "rg02": ((112, 1), (117, 6), (5198, 1), (4272, 35), (4912, 16), (6672, 45), (6161, 12), (4912, 16), (4990, 2),
             (4732, 2)),
    "rg03": ((146, 28), (145, 8), (117, 7), (7071, 139), (4793, 92), (4732, 150), (4272, 64), (6672, 30), (4272, 8),
             (5933, 28)),
    "rg04": ((114, 30), (105, 16), (5489, 717), (5866, 248), (5328, 184), (6672, 94), (4914, 115), (4914, 19),
             (1262816, 12595), (1262816, 8250)),
    "rg06": ((125, 32), (105, 28), (178, 14), (5177, 344), (5866, 207), (6218, 64), (5866, 70), (4519, 34),
             (1262816, 3796), (1262816, 3046)),
}


def act(game: WitnessGame, action: int) -> FrameData:
    return game.perform_action(ActionInput(id=GameAction.from_id(action)))


def test_the_r2_ids_are_registered_outside_the_public_and_r1_games():
    assert TRAIN_R2_IDS == tuple(PATH_COUNTS) == golden.TRAIN_R2_GAMES
    assert not set(TRAIN_R2_IDS) & set(GAME_IDS + TRAIN_R1_IDS)
    assert sorted(golden.STORE.solutions_r2()) == sorted(TRAIN_R2_IDS)
    for game_id in TRAIN_R2_IDS:
        game = make_game(game_id)
        assert type(game).__name__ == game_id.capitalize() and type(game).GAME_ID == game.game_id == game_id
        assert game.win_score == LEVELS


def test_the_r2_task_file_is_the_r1_rows_then_one_row_per_r2_game():
    r1 = (ROOT / "data" / "train_r1.jsonl").read_text(encoding="utf-8").splitlines()
    r2 = (ROOT / "data" / "train_r2.jsonl").read_text(encoding="utf-8").splitlines()
    assert r2[:len(r1)] == r1
    assert [json.loads(line) for line in r2[len(r1):]] == [
        {"prompt": g, "metadata": {"game_id": g, "max_levels": LEVELS, "max_reflection_calls": 300}}
        for g in TRAIN_R2_IDS]


@pytest.mark.parametrize("game_id", TRAIN_R2_IDS)
def test_the_level_file_is_compact_with_distinct_levels_and_known_keys(game_id):
    configs = load_level_configs(game_id)
    assert len(configs) == LEVELS
    assert len({json.dumps(c, sort_keys=True) for c in configs}) == LEVELS
    text = "[\n" + ",\n".join(json.dumps(c, separators=(",", ":"), ensure_ascii=True) for c in configs) + "\n]\n"
    assert (files("witness_env") / "levels" / f"{game_id}.json").read_bytes() == text.encode("ascii")
    assert all(set(c) <= type(make_game(game_id)).CONFIG_KEYS for c in configs)


@pytest.mark.parametrize("game_id", TRAIN_R2_IDS)
def test_stored_solutions_played_in_a_row_win_every_level(game_id):
    solutions = golden.STORE.solutions_r2()[game_id]
    assert len(solutions) == LEVELS
    game = make_game(game_id)
    act(game, RESET)
    for level, solution in enumerate(solutions):
        results = [act(game, a) for a in solution]
        assert [fd.levels_completed for fd in results] == [level] * (len(solution) - 1) + [level + 1], level
    assert results[-1].state.name == "WIN"


@pytest.mark.parametrize("game_id", TRAIN_R2_IDS)
def test_each_stored_solution_wins_its_level_on_its_last_action_only(game_id):
    for level, solution in enumerate(golden.STORE.solutions_r2()[game_id]):
        game = make_game(game_id)
        game.set_level(level)
        completed = [act(game, a).levels_completed for a in solution]
        assert completed == [0] * (len(solution) - 1) + [1], level


@pytest.mark.parametrize(("game_id", "level"), [(g, i) for g in TRAIN_R2_IDS for i in range(LEVELS)])
def test_every_level_accepts_well_under_half_of_its_paths(game_id, level):
    paths, accepted = PATH_COUNTS[game_id][level]
    assert 0 < accepted and accepted / paths < 0.5


def _is_5x5(game_id: str, level: int) -> bool:
    config = load_level_configs(game_id)[level]
    return config["cols"] * config["rows"] >= 25


@pytest.mark.parametrize(("game_id", "level"), [
    pytest.param(g, i, marks=pytest.mark.skipif(not SLOW and _is_5x5(g, i),
                                                reason="5x5 level: set WITNESS_SLOW_TESTS=1"))
    for g in TRAIN_R2_IDS for i in range(LEVELS)])
def test_the_path_counts_of_every_level_are_enumerated(game_id, level):
    config = load_level_configs(game_id)[level]
    game = make_game(game_id)
    board = Board.from_config(config)
    assert golden.count_paths(game, board, type(game)._parse_markers(config, board)) == PATH_COUNTS[game_id][level]


def _symmetries(n: int):
    """The 8 dihedral maps of the node lattice 0..n of an n x n board, each with the cell map it induces."""
    out = []
    for k in range(8):
        def node(p, k=k):
            c, r = p
            for _ in range(k % 4):
                c, r = n - r, c
            return (n - c, r) if k >= 4 else (c, r)

        def cell(c, node=node):
            corners = [node(p) for p in ((c[0], c[1]), (c[0] + 1, c[1]), (c[0], c[1] + 1), (c[0] + 1, c[1] + 1))]
            return (min(x for x, _ in corners), min(y for _, y in corners))
        out.append((node, cell))
    return out


def _marker_cells(config: dict, cell) -> dict:
    """The markers of a level config with every cell mapped by `cell`: dicts "c,r" -> value, lists of [c, r]."""
    out = {}
    for key in sorted(set(config) - {"cols", "rows", "start", "end"}):
        value = config[key]
        if isinstance(value, dict):
            out[key] = frozenset((cell(tuple(int(n) for n in k.split(","))), json.dumps(v)) for k, v in value.items())
        else:
            out[key] = frozenset(cell(tuple(c)) for c in value)
    return out


@pytest.mark.parametrize("game_id", TRAIN_R2_IDS)
def test_no_level_is_a_board_symmetry_copy_of_another_level_of_its_game(game_id):
    """Under none of the 8 rotations and reflections, also with start and end swapped, does a level's config
    (board, start, end, every marker cell and value) map onto another level's."""
    configs = load_level_configs(game_id)
    assert all(c["cols"] == c["rows"] for c in configs)
    for i, a in enumerate(configs):
        for j in range(i + 1, LEVELS):
            b = configs[j]
            if a["cols"] != b["cols"]:
                continue
            ends_b, markers_b = (tuple(b["start"]), tuple(b["end"])), _marker_cells(b, lambda c: c)
            for k, (node, cell) in enumerate(_symmetries(a["cols"])):
                s, e = node(tuple(a["start"])), node(tuple(a["end"]))
                assert not ({(s, e), (e, s)} & {ends_b} and _marker_cells(a, cell) == markers_b), (i + 1, j + 1, k)


@pytest.mark.parametrize("game_id", TRAIN_R2_IDS)
def test_no_stored_solution_wins_a_later_level(game_id):
    solutions = golden.STORE.solutions_r2()[game_id]
    for level, solution in enumerate(solutions):
        for later in range(level + 1, LEVELS):
            game = make_game(game_id)
            game.set_level(later)
            assert not any(act(game, a).levels_completed for a in solution), (level + 1, later + 1)


@pytest.mark.parametrize(("game_id", "level"), [(g, i) for g in TRAIN_R2_IDS for i in range(LEVELS)])
def test_the_stored_solution_of_every_level_is_a_shortest_one(game_id, level):
    """No action sequence one action shorter wins: breadth-first search over the engine's own moves (CONFIRM is
    only ever tried at the end node, and only at the depth of the stored solution minus one)."""
    solution = golden.STORE.solutions_r2()[game_id][level]
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
