"""The plans adopted on the current level and the "## Recent Attempts (this level)" section of the prompt.

A plan containing CONFIRM is an attempt; any other plan is a chunk, and a plan's reset count leaves out its RESETs
that were not sent to the engine (nothing to undo). The section lists the last `TRIAL_DEPTH` finished plans of the
level with how each ended, and flags two identical attempts started from the same state (the same state key). A
rejected CONFIRM noted with a detail (what it submitted and ringed) gets a line of its own under its plan's line.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Final

from .episode import CONFIRM, RESET

TRIAL_DEPTH: Final = 2              # finished plans shown in the section

# Section text (model-visible).
_HEADING: Final = "\n## Recent Attempts (this level)"
_COUNT: Final = "You have made {attempts} attempt{s} on this level so far."
_CHUNKS: Final = (" ({chunks} of the plans below ran without a submit — partial chunks, not counted as "
                  "attempts.)")
_LINE: Final = "- {label}: [{actions}] → {end_reason}{extra}"
_LABEL_ATTEMPT: Final = "attempt #{i}"
_LABEL_CHUNK: Final = "chunk before attempt #{i} (no submit)"
_RESETS: Final = "; included {k} reset(s)"
_IDENTICAL: Final = ("⚠ Your last two attempts were IDENTICAL action sequences from the same starting "
                     "state.")
_FOOTER: Final = "(Attempt history is provided automatically — do not store attempts as knowledge rules.)"
_END_REJECTED_FIRST: Final = "submit rejected at action {k}/{n}"
_END_REJECTED_MORE: Final = "{m} submits rejected (last at action {k}/{n})"
_END_CHUNK: Final = "ran to its end — continue from the resulting state"
_CAP_SUFFIX: Final = " ({cap_note})"
_DETAIL: Final = "    submit at action {k}: {detail}"


@dataclass
class _Trial:
    """One adopted plan: its label number, actions, the state key its reflection saw, and how it ended."""

    number: int                     # the attempt's number; for a chunk, the number the next attempt gets
    actions: tuple[int, ...]
    start_key: str
    is_attempt: bool
    cap_note: str
    rejected: int = 0               # CONFIRMs of this plan that did not complete the level
    noop_resets: int = 0            # RESETs of this plan that were not sent to the engine (left out of the count)
    end_reason: str = ""
    details: list[tuple[int, str]] = field(default_factory=list)    # (action number, detail) per detailed CONFIRM


class TrialMemory:
    """Plans adopted on the current level: the open one and the last TRIAL_DEPTH finished ones."""

    def __init__(self) -> None:
        self._open: _Trial | None = None
        self._done: list[_Trial] = []      # oldest first
        self._attempts = 0                 # every attempt opened on this level, shown or not

    def open(self, plan: Sequence[int], start_key: str, cap_note: str) -> None:
        """Open a trial for `plan` (the plan after the plan cap) adopted by a reflection that saw the state
        `start_key`; `cap_note` is the plan-cap note ("" when the cap did not apply). No trial may be open: the
        previous one must have been finished by `close_if_exhausted` (or dropped by `reset`)."""
        assert self._open is None
        is_attempt = CONFIRM in plan
        if is_attempt:
            self._attempts += 1
        number = self._attempts if is_attempt else self._attempts + 1
        self._open = _Trial(number, tuple(plan), start_key, is_attempt, cap_note)

    def note_rejected_confirm(self, remaining: int, detail: str = "") -> None:
        """A CONFIRM of the open plan did not complete the level; `remaining` actions of the plan are left. A
        trial must be open and `0 <= remaining <` its plan length. A non-empty `detail` is shown on a line of its
        own under the plan's line."""
        trial = self._open
        assert trial is not None and 0 <= remaining < len(trial.actions)
        n = len(trial.actions)
        k = n - remaining
        trial.rejected += 1
        trial.end_reason = (_END_REJECTED_FIRST.format(k=k, n=n) if trial.rejected == 1
                            else _END_REJECTED_MORE.format(m=trial.rejected, k=k, n=n))
        if detail:
            trial.details.append((k, detail))

    def note_noop_reset(self) -> None:
        """A RESET of the open plan was not sent to the engine (it is left out of the reset count). A trial must be
        open."""
        assert self._open is not None
        self._open.noop_resets += 1

    def close_if_exhausted(self, remaining: int) -> None:
        """Finish the open trial when its plan has no action left (`remaining == 0`); only the last
        `TRIAL_DEPTH` finished trials are kept."""
        trial = self._open
        if trial is None or remaining != 0:
            return
        if not trial.end_reason:
            trial.end_reason = _END_CHUNK           # an attempt always ends with a rejected submit, which sets it
        if trial.cap_note:
            trial.end_reason += _CAP_SUFFIX.format(cap_note=trial.cap_note)
        self._done.append(trial)
        del self._done[:-TRIAL_DEPTH]
        self._open = None

    def reset(self) -> None:
        """Forget every trial of the level (a new level starts; the plan that solved the level is not kept)."""
        self._done.clear()
        self._open = None
        self._attempts = 0

    def section(self) -> str:
        """The "## Recent Attempts (this level)" block with a leading and a trailing newline, or "" before the
        first trial of the level has finished. An open trial is not shown."""
        if not self._done:
            return ""
        header = _COUNT.format(attempts=self._attempts, s="" if self._attempts == 1 else "s")
        chunks = sum(1 for trial in self._done if not trial.is_attempt)
        if chunks:
            header += _CHUNKS.format(chunks=chunks)
        lines = [_HEADING, header]
        for trial in self._done:
            label = (_LABEL_ATTEMPT if trial.is_attempt else _LABEL_CHUNK).format(i=trial.number)
            resets = trial.actions.count(RESET) - trial.noop_resets
            lines.append(_LINE.format(label=label, actions=",".join(map(str, trial.actions)),
                                      end_reason=trial.end_reason, extra=_RESETS.format(k=resets) if resets else ""))
            lines += [_DETAIL.format(k=k, detail=detail) for k, detail in trial.details]
        last_two = self._done[-2:]
        first, last = last_two[0], last_two[-1]
        if (len(last_two) == 2 and first.is_attempt and last.is_attempt and first.actions == last.actions
                and first.start_key == last.start_key):
            lines.append(_IDENTICAL)
        lines.append(_FOOTER)
        return "\n".join(lines) + "\n"
