"""Per-episode journals of paid completions, and resuming an episode from its journal.

An episode is determined by its game, seed, limits and the sequence of policy answers (`run_episode`). `JournalPolicy`
therefore appends every answer to a JSON Lines journal as soon as it arrives (flushed to disk), and an interrupted
episode can be played again from the start: the journaled answers are replayed in order, without calling the inner
policy, and each replayed call must carry exactly the prompts it was recorded with (their SHA-256), or
`ResumeMismatch` is raised; after the last journaled answer the inner policy answers again, and its answers are
appended. An interruption therefore loses at most the call in flight.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict
from pathlib import Path

from .episode import Completion, Policy


class ResumeMismatch(RuntimeError):
    """A replayed call's prompts differ from the ones its journaled answer was given for."""


def prompt_digest(system: str, user: str) -> str:
    """SHA-256 of the two prompts of one call (UTF-8, separated by a NUL character)."""
    return hashlib.sha256((system + "\0" + user).encode("utf-8", errors="surrogatepass")).hexdigest()


def read_journal(path: Path) -> list[dict]:
    """The entries of the journal at `path` in order ([] when it does not exist). A last line without its newline
    (written when the process stopped) is dropped; any other malformed line raises ValueError."""
    if not path.exists():
        return []
    data = path.read_bytes()
    # an unfinished last line, possibly cut inside a character: that call was not answered for good
    text = data[:data.rfind(b"\n") + 1].decode("utf-8")
    entries = []
    for number, line in enumerate(text.split("\n"), start=1):
        if not line:
            continue
        try:
            entries.append(json.loads(line))
        except ValueError as exc:
            raise ValueError(f"{path}:{number}: not JSON") from exc
    return entries


class JournalPolicy:
    """`inner` with a journal at `path`: replays the entries of an existing journal first (see the module), then
    asks `inner` and appends each of its answers. An unfinished last line of the journal is cut off the file, so the
    first answer appended starts a line of its own. Only `Completion` answers are journaled; any other answer of
    `inner` raises TypeError. `replayed` counts the answers taken from the journal."""

    def __init__(self, inner: Policy, path: Path) -> None:
        self._inner = inner
        self._path = path
        if path.exists():
            os.truncate(path, path.read_bytes().rfind(b"\n") + 1)
        self._pending = read_journal(path)
        self._index = 0
        self.replayed = 0

    def __call__(self, system: str, user: str) -> Completion:
        digest = prompt_digest(system, user)
        if self._pending:
            entry = self._pending.pop(0)
            if entry["prompt_sha256"] != digest:
                raise ResumeMismatch(f"{self._path}: call {self._index} has other prompts than its journaled answer "
                                     "(the harness or the game changed since it was recorded)")
            self._index += 1
            self.replayed += 1
            return Completion(**entry["completion"])
        answer = self._inner(system, user)
        if not isinstance(answer, Completion):
            raise TypeError(f"the journaled policy returned {type(answer).__name__}, expected a Completion")
        entry = {"index": self._index, "prompt_sha256": digest, "completion": asdict(answer)}
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")         # ASCII: a lone surrogate of a reply is written as its escape
            f.flush()
            os.fsync(f.fileno())
        self._index += 1
        return answer
