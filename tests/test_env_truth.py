"""Board snapshots reported by the games (`WitnessGame.truth()`).

Golden fixtures used: `env/traces/<game>.json` (scripted action sequences) and `env/truth/<game>.json` (the snapshot
right after entering each level of a training game, in the form of `golden.encode_truth`). At every state of a
scripted sequence the snapshot equals its level's entry snapshot with the path and start of that state. The
structure checks compare the snapshot of every level with the level's config in the level file.
"""
from __future__ import annotations

from types import MappingProxyType

import pytest
from arcengine import ActionInput, GameAction

import golden
from witness_env import GAME_IDS, BoardTruth, Polyomino, make_game
from witness_env.base import load_level_configs

TW_GAMES = tuple(g for g in GAME_IDS if g.startswith("tw"))


def _cell(key: str) -> tuple[int, int]:
    c, r = key.split(",")
    return (int(c), int(r))


def _level_truths(game_id: str) -> list[BoardTruth]:
    """The snapshot right after entering each level of a fresh game."""
    truths = []
    for level in range(len(load_level_configs(game_id))):
        game = make_game(game_id)
        game.set_level(level)
        truths.append(game.truth())
    return truths


@pytest.mark.parametrize("game_id", golden.TRAINING_GAMES)
def test_truth_along_scripted_actions_matches_the_golden_fixture_of_level_snapshots(game_id):
    levels = golden.STORE.truth_levels(game_id)
    assert len(levels) == len(load_level_configs(game_id))
    game = make_game(game_id)
    for i, step in enumerate(golden.STORE.env_trace(game_id).steps):
        game.perform_action(ActionInput(id=GameAction.from_id(step.action)))
        expected = dict(levels[step.level], path=[list(n) for n in step.path], start=list(step.start))
        assert golden.encode_truth(game.truth()) == expected, i


@pytest.mark.parametrize("game_id", TW_GAMES)
def test_tw_truth_of_every_level_reports_the_markers_of_its_level_config(game_id):
    for level, (config, truth) in enumerate(zip(load_level_configs(game_id), _level_truths(game_id))):
        starts = tuple(tuple(s) for s in config["starts"]) if "starts" in config else (tuple(config["start"]),)
        pieces = {_cell(k): Polyomino(tuple(tuple(x) for x in v["shape"]), v.get("rotated", False))
                  for k, v in config.get("tetris", {}).items()}
        breakpoints = frozenset(tuple(sorted(map(tuple, pair))) for pair in config.get("breakpoints", []))
        assert (truth.cols, truth.rows, truth.end) == (config["cols"], config["rows"], tuple(config["end"])), level
        assert (truth.starts, truth.start, truth.path) == (starts, starts[0], (starts[0],)), level
        assert truth.breakpoints == breakpoints, level
        assert truth.dots == tuple(tuple(d) for d in config.get("dots", [])), level
        for name in ("squares", "stars", "triangles"):
            assert dict(getattr(truth, name)) == {_cell(k): v for k, v in config.get(name, {}).items()}, (level, name)
        assert dict(truth.polyominoes) == pieces, level
        assert all(type(piece.rotatable) is bool for piece in truth.polyominoes.values()), level
        assert truth.erasers == frozenset(tuple(e) for e in config.get("erasers", [])), level
        assert truth.framed_cells == frozenset(_cell(k) for k in config.get("filters", {})), level
        assert (truth.node_markers, truth.cell_markers) == ((), ()), level
        assert all(isinstance(getattr(truth, name), MappingProxyType)
                   for name in ("squares", "stars", "triangles", "polyominoes")), level
        assert type(truth.erasers) is frozenset and type(truth.framed_cells) is frozenset, level
