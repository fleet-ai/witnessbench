"""The games the harness plays: the 30 public games of `witness_env`, its training-only games (`TRAIN_R1_IDS`,
`TRAIN_R2_IDS`) and, when the environment variable WITNESS_GAME_CATALOG names a module, that module's games.

The named module must define `PRIVATE_GAME_IDS` (a tuple of the ids of its games, none of them an id of
`witness_env`) and `make_game(game_id)` with the contract of `witness_env.make_game`; nothing else of it is read and
no other module is imported here. The variable is read on every call, and the module is imported then (an import
error is raised, never hidden). With the variable unset or empty the harness knows the public games only.
"""
from __future__ import annotations

import importlib
import os
from types import ModuleType
from typing import Final

import witness_env
from witness_env import WitnessGame

CATALOG_ENV: Final = "WITNESS_GAME_CATALOG"


def catalog_name() -> str | None:
    """The module named by the variable, or None when it is unset or empty."""
    return os.environ.get(CATALOG_ENV) or None


def _catalog() -> ModuleType | None:
    name = catalog_name()
    return None if name is None else importlib.import_module(name)


def game_ids() -> tuple[str, ...]:
    """The public ids, the training-only ids (`TRAIN_R1_IDS`, `TRAIN_R2_IDS`), then the catalog module's ids. Raises
    ValueError naming the catalog ids that are also public."""
    public_ids = witness_env.GAME_IDS + witness_env.TRAIN_R1_IDS + witness_env.TRAIN_R2_IDS
    module = _catalog()
    if module is None:
        return public_ids
    extra = tuple(module.PRIVATE_GAME_IDS)
    public = [g for g in extra if g in public_ids]
    if public:
        raise ValueError(f"the game catalog {catalog_name()} lists public game ids: {' '.join(public)}")
    return public_ids + extra


def make_game(game_id: str) -> WitnessGame:
    """A new game for an id of `game_ids()`: a public or training-only id (`TRAIN_R1_IDS`, `TRAIN_R2_IDS`) ->
    `witness_env.make_game`, a catalog id -> the module's `make_game`; any other id -> the KeyError of
    `witness_env.make_game` (its message is unchanged)."""
    public_ids = witness_env.GAME_IDS + witness_env.TRAIN_R1_IDS + witness_env.TRAIN_R2_IDS
    if game_id not in public_ids and game_id in game_ids():
        return _catalog().make_game(game_id)
    return witness_env.make_game(game_id)
