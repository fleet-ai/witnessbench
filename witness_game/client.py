"""The game side of an episode as `run_episode` sees it: `GameClient`, the `GameFrame` that every action returns, and
`RemoteGameClient` for a game that runs in another process. Prompts, rule memory and plans stay in the harness;
engine state, levels and rendering stay with the game."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .truth import BoardTruth

PROTOCOL = 'witness.game.v1'


@dataclass(frozen=True)
class GameFrame:
    """The game's state after one action.

    Attributes:
        levels_completed: Levels completed so far.
        win_levels: The number of levels of the game; completing all of them wins it.
        full_reset: Whether the action restarted the whole game.
        won: Whether the game is won.
        truth: The board snapshot after the action.
    """

    levels_completed: int
    win_levels: int
    full_reset: bool
    won: bool
    truth: BoardTruth


class GameClient(Protocol):
    """A game as `run_episode` plays it: `game_id`, `reset()` for the first RESET of an episode, and `step(action)`
    for every later action (0 RESET, 1-4 moves, 5 CONFIRM); both return a `GameFrame`."""

    game_id: str

    def reset(self) -> GameFrame: ...
    def step(self, action: int) -> GameFrame: ...


class RemoteGameClient:
    """A `GameClient` for a game served by a `witness_game.service.GameService` in another process.

    `call(name, arguments)` sends one request over a synchronous transport of your choice (for example HTTP) to the
    service and returns its reply; `description` is the service's `describe()`. The client never retries. After a
    call that raises or returns a malformed reply, the client cannot know whether the service performed the action,
    so every later action raises RuntimeError; find out on the server what happened before you start another
    episode. The service answers a repeat of its last request with the saved reply and does not perform the action
    again.
    """
    def __init__(self, call, description):
        if (not isinstance(description, dict) or description.get('protocol') != PROTOCOL
                or not isinstance(description.get('session_id'), str)
                or not description['session_id']
                or not isinstance(description.get('game_id'), str) or not description['game_id']):
            raise ValueError('invalid game description')
        self.call, self.session_id = call, description['session_id']
        self.game_id = description['game_id']
        self.sequence, self.failed = 0, False

    def reset(self):
        if self.sequence != 0:
            raise RuntimeError('initial reset already sent; use step(0) for a level reset')
        return self.step(0)

    def step(self, action):
        from .wire import decode_frame
        if self.failed:
            raise RuntimeError('game state uncertain: an earlier action may or may not have been performed; '
                               'find out on the server before you start another episode')
        if type(action) is not int or not 0 <= action <= 5:
            raise ValueError(f'invalid game action: expected an int from 0 to 5, got {action!r}')
        if self.sequence == 0 and action != 0:
            raise ValueError(f'the first action of a session must be 0 (RESET), got {action}; call reset() first')
        try:
            reply = self.call('game_action', {'session_id': self.session_id,
                              'sequence': self.sequence, 'action': action})
            if (not isinstance(reply, dict) or set(reply) != {'session_id', 'sequence', 'frame'}
                    or reply['session_id'] != self.session_id
                    or type(reply['sequence']) is not int or reply['sequence'] != self.sequence):
                raise ValueError(f'the reply does not answer this request (session id {self.session_id!r}, '
                                 f'sequence {self.sequence})')
            frame = decode_frame(reply['frame'])
        except BaseException:
            self.failed = True
            raise
        self.sequence += 1
        return frame
