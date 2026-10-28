"""Thirty public grid line-puzzle games on arcengine, plus the ten training-only games of the revised training sets
r1 and r2 (`TRAIN_R1_IDS`, `TRAIN_R2_IDS`).

Each game is a sequence of levels. A level is a lattice of nodes with one or more start nodes, one end node
and markers; the player draws a path from a start along the lattice and confirms it, and the level is solved
when the path ends on the end node and satisfies the game's rule. Create a game with `make_game(game_id)` and
drive it with arcengine's `perform_action`: RESET, ACTION1-ACTION4 (up, down, left, right) and ACTION5
(confirm). `WitnessGame.truth()` returns an immutable `BoardTruth` snapshot of the current level.
"""
from .base import WitnessGame
from .catalog import GAME_IDS, TRAIN_R1_IDS, TRAIN_R2_IDS, make_game
from .truth import BoardTruth, CellMarkers, LegendEntry, NodeMarkers, Polyomino, RevealNodes

__all__ = ("GAME_IDS", "TRAIN_R1_IDS", "TRAIN_R2_IDS", "make_game", "WitnessGame", "BoardTruth", "CellMarkers",
           "NodeMarkers", "Polyomino", "RevealNodes", "LegendEntry")
