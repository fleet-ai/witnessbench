"""The server side of a game played in another process: one game, one session id, actions in strict order.

The service has no authentication; run it behind an authenticated transport. Every service gets a new session id,
so after a restart a request meant for the earlier service is rejected instead of resetting or replaying the new
game. Hosts must also make sure an episode is not restarted twice.
"""
import copy
import threading
import uuid

from .client import PROTOCOL
from .wire import encode_frame


class GameService:
    """Serves one game, a `witness_game.GameClient` such as `witness_env.client.LocalGameClient`, to one
    `witness_game.RemoteGameClient`.

    `describe()` returns the protocol, the session id and the game id. `action(session_id, sequence, action)`
    performs action number `sequence` of the session (0, 1, 2, ... in order; the action of sequence 0 must be 0,
    RESET) and returns the reply with the encoded frame. A repeat of the last request gets its saved reply without the
    action being performed again; any other request out of order raises ValueError. After an action fails, the
    service performs no further action (RuntimeError). `close()` makes it refuse new actions.
    """

    def __init__(self, game):
        self.game = game
        self.session_id = uuid.uuid4().hex
        self.sequence = 0
        self.last = None
        self.failed = False
        self.closed = False
        self.lock = threading.RLock()

    def describe(self):
        return {'protocol': PROTOCOL, 'session_id': self.session_id, 'game_id': self.game.game_id}

    def action(self, session_id, sequence, action):
        with self.lock:
            if (session_id != self.session_id or type(sequence) is not int or sequence < 0
                    or type(action) is not int or not 0 <= action <= 5):
                raise ValueError('invalid game request')
            if self.failed:
                raise RuntimeError('game state uncertain: an earlier action may or may not have been performed')
            if self.last is not None and sequence == self.sequence - 1:
                prior_action, response = self.last
                if action != prior_action:
                    raise ValueError('sequence reused for a different action')
                return copy.deepcopy(response)
            if self.closed or sequence != self.sequence or (sequence == 0 and action != 0):
                raise ValueError('game request is out of order or session closed')
            # Reserve before calling the engine, including observer/encoding work.
            # If anything fails after dispatch, no request may repeat that write.
            self.failed = True
            frame = self.game.reset() if sequence == 0 else self.game.step(action)
            response = {'session_id': self.session_id, 'sequence': sequence, 'frame': encode_frame(frame)}
            self.last = action, response
            self.sequence += 1
            self.failed = False
            return copy.deepcopy(response)

    def close(self):
        with self.lock:
            self.closed = True
