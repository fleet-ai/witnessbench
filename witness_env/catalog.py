"""The registry of the 30 games and of the revised training games: their ids and a constructor by id.

`make_game` imports `witness_env.games.<id>` on first use and instantiates its one class, named
`id.capitalize()` (`Tw01`, `Ft20`, `Fv04`, `Tw02r1`, ...). Every call returns a new, independent game.

`TRAIN_R1_IDS` are the games of the revised training set r1 (docs/training.md): each is a public training game
with some levels replaced or its drawing fixed, under its own id and level file. They are not among the 30
`GAME_IDS`, which the published splits and the paper use; the harness and the evaluators accept them too
(`witness_agent.catalog.game_ids`).

`TRAIN_R2_IDS` are the training-only games of training set r2: new games with region rules, likewise outside
`GAME_IDS`.
"""
from __future__ import annotations

import importlib
from typing import Final

from .base import WitnessGame

GAME_IDS: Final[tuple[str, ...]] = (
    "tw01", "tw02", "tw03", "tw05", "tw06", "tw07", "tw08", "tw09", "tw10", "tw11", "tw12", "tw13",
    "ft01", "ft02", "ft03", "ft04", "ft05", "ft06", "ft10", "ft11", "ft12", "ft13", "ft14", "ft15", "ft16", "ft20",
    "fv01", "fv02", "fv03", "fv04",
)
TRAIN_R1_IDS: Final[tuple[str, ...]] = ("tw02r1", "tw03r1", "tw10r1", "ft03r1", "ft06r1")
TRAIN_R2_IDS: Final[tuple[str, ...]] = ("rg01", "rg02", "rg03", "rg04", "rg06")


def make_game(game_id: str) -> WitnessGame:
    """Return a new game for `game_id` (an id of `GAME_IDS`, `TRAIN_R1_IDS` or `TRAIN_R2_IDS`). Raises KeyError for
    an unknown id and RuntimeError if ONLY_RESET_LEVELS is "true" (see WitnessGame)."""
    if game_id not in GAME_IDS + TRAIN_R1_IDS + TRAIN_R2_IDS:
        raise KeyError(f"unknown game id {game_id!r}; known ids: {' '.join(GAME_IDS + TRAIN_R1_IDS + TRAIN_R2_IDS)}")
    module = importlib.import_module(f"witness_env.games.{game_id}")
    return getattr(module, game_id.capitalize())()
