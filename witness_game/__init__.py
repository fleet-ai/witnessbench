"""Engine-free game client and observation contract shared by server and harness."""
from .client import GameClient, GameFrame, RemoteGameClient

__all__ = ("GameClient", "GameFrame", "RemoteGameClient")
