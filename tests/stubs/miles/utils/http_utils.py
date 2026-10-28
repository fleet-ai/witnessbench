"""Test stand-in for the miles interfaces used by witness_rl."""
from __future__ import annotations

from typing import Any


async def post(url: str, payload: Any, max_retries: int = 60, action: str = "post", headers: Any = None) -> Any:
    """Always raises: the tests replace `witness_rl.hooks.post` with a scripted server."""
    raise RuntimeError("test stand-in: replace witness_rl.hooks.post")
