"""`LocalGameClient`: a game of this process as a `witness_game.GameClient`, for `run_episode` and for a
`witness_game.service.GameService` that serves the game to another process. Importing the harness
(`witness_agent`) does not import this module; `run_episode` imports it only to play a game of its own process.
"""
from arcengine import ActionInput, GameAction, GameState
from witness_game import GameFrame


class LocalGameClient:
    """`game` (a `witness_env.WitnessGame`) as a `witness_game.GameClient`: `step(action)` performs action 0-5 (0
    RESET, 1-4 moves, 5 CONFIRM) and returns a `GameFrame` with the game's `truth()`; `reset()` is `step(0)`."""

    def __init__(self, game):
        self.game = game
        self.game_id = game.game_id

    def reset(self):
        return self.step(0)

    def step(self, action):
        if type(action) is not int or not 0 <= action <= 5:
            raise ValueError(f'invalid game action: expected an int from 0 to 5, got {action!r}')
        frame = self.game.perform_action(ActionInput(id=GameAction.from_id(action)))
        return GameFrame(frame.levels_completed, frame.win_levels, frame.full_reset,
                         frame.state == GameState.WIN, self.game.truth())
