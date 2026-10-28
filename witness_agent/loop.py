"""The episode loop: one game played by one policy under the harness, from its first level to the first stop.

After the engine's RESET, every step takes the next action of the current plan or, when the plan is used up,
makes one reflection call: the policy gets a system prompt and a user message built from the harness state and
answers with rule edits and a plan of at most `parse.PLAN_CAP` actions, whose first action is taken at once. An
answer without a plan is followed by one random move from `RANDOM_ACTIONS`, drawn from a random source seeded by
the game id and the episode seed; the next step reflects again. An answer cut off at the output limit (a
`Completion` with finish reason "length") counts as a call but is not applied: no action follows, and the next
prompt says why (`TRUNCATED_NOTICE`).

A RESET chosen when no action has been sent to the engine since the level started or since the last executed
RESET is not sent (the engine would restart the whole game); it still counts as an action and is shown as an
unchanged board. Every level has a budget of `MAX_STEPS_PER_LEVEL` actions.

The episode stops at the first of: `max_levels` levels completed, the game won, a level's budget used up, the
evaluation call cap reached (checked when a new call is due, so the last allowed call's plan runs to its end, as in
training), or the policy declining a call.

A game state is identified by the state key of its board view (`view.state_key`: everything the view shows but the
ringed cells).
"""
from __future__ import annotations

import random
import zlib
from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, Literal

from witness_game import GameClient, GameFrame

if TYPE_CHECKING:
    from witness_env import WitnessGame

from .actions import ActionTracker, move_vote
from .board import board_block, budget_line
from .episode import (CONFIRM, RESET, Actor, CallRecord, Completion, EpisodeAborted, EpisodeLimits, EpisodeResult,
                      Policy, ResetRecord, StepRecord, StopReason)
from .history import ObservationWindow, TransitionTable
from .knowledge import RuleStore
from .parse import parse_plan, parse_response
from .prompts import kas_text, system_prompt, user_message
from .transitions import MovementTracker, breadcrumb, showcase, submit_details, transition_line
from .trials import TrialMemory
from .view import BoardView, board_view, state_key

MAX_STEPS_PER_LEVEL: Final = 300          # actions allowed on one level; running out ends the episode
RANDOM_ACTIONS: Final = (1, 2, 3, 4)      # the moves the random fallback chooses from (never RESET or CONFIRM)
# The notice that follows a reply cut off at the output limit (`Completion.truncated`).
TRUNCATED_NOTICE: Final = ("your previous reply was cut off at the output limit before it ended, so none of it was "
                           "applied: no rule edits, no plan, no action")
_TRUNCATED: Final = object()              # `_answer`'s result for a truncated reply: take no action


def run_episode(game: GameClient | WitnessGame, *, seed: int, limits: EpisodeLimits, policy: Policy) -> EpisodeResult:
    """Play one episode of `game`, freshly made (`witness_agent.catalog.make_game`), and return its record.

    The episode is determined by the game id, `seed`, `limits` and the sequence of policy answers. Every exception
    raised inside the episode, including one raised by the policy, ends it with `EpisodeAborted`, which holds
    everything recorded so far and chains the exception. Episodes of distinct games and policies may run
    concurrently on threads.
    """
    run = _EpisodeRun(game, seed, limits, policy)
    try:
        return run.play()
    except Exception as exc:
        raise EpisodeAborted(run.result("aborted")) from exc


@dataclass(frozen=True)
class _Observation:
    """One game state as the harness sees it."""

    key: str                    # the state's identity: the view's state key
    view: BoardView


class _EpisodeRun:
    """The whole state of one episode.

    Per episode: the rule store, the trial memory, the action and movement trackers and the random source; per
    level: the transition table, the observation window and the views of the level's states by state identity
    (`_Observation.key`).
    """

    rng: random.Random                      # the random source of the fallback moves
    fd: GameFrame                           # the engine's latest answer
    cur: _Observation                       # the current state
    level_total: int                        # the M of "Level: K/M": min(the game's levels, max_levels)
    table: TransitionTable
    window: ObservationWindow
    snapshots: dict[str, BoardView]

    def __init__(self, game: GameClient, seed: int, limits: EpisodeLimits, policy: Policy) -> None:
        # An in-process game is wrapped in a LocalGameClient; a remote GameClient is used as is, so the harness side
        # never imports the engine.
        if hasattr(game, "perform_action"):
            from witness_env.client import LocalGameClient
            game = LocalGameClient(game)
        self.game, self.seed, self.limits, self.policy = game, seed, limits, policy
        self.store, self.trials = RuleStore(), TrialMemory()
        self.tracker, self.movement = ActionTracker(), MovementTracker()
        self.queue: deque[int] = deque()
        self.notices: list[str] = []
        self.calls: list[CallRecord] = []
        self.steps: list[StepRecord] = []
        self.resets: list[ResetRecord] = []
        self.level = self.level_step = self.acts_since_reset = 0

    def result(self, stop_reason: StopReason | Literal["aborted"]) -> EpisodeResult:
        """Everything recorded so far, as an episode that ended for `stop_reason`."""
        return EpisodeResult(game_id=self.game.game_id, seed=self.seed, max_levels=self.limits.max_levels,
                             calls=tuple(self.calls), steps=tuple(self.steps), resets=tuple(self.resets),
                             levels_completed=self.level, stop_reason=stop_reason)

    def play(self) -> EpisodeResult:
        """Play from the engine's RESET to the first stop."""
        self.rng = random.Random((self.seed * 1_000_003) ^ zlib.crc32(self.game.game_id.encode("utf-8")))
        self.fd = self.game.reset()
        if self.fd.levels_completed != 0:
            raise ValueError("run_episode needs a freshly made game")
        self.level_total = min(self.fd.win_levels, self.limits.max_levels)
        self.cur = self._observe()
        self._begin_level()
        cap = self.limits.max_reflection_calls
        while True:
            if cap is not None and len(self.calls) >= cap and not self.queue:
                return self.result("call_budget")
            self.trials.close_if_exhausted(remaining=len(self.queue))
            actor: Actor = "plan"
            if self.queue:
                action = self.queue.popleft()
            else:
                self.notices.extend(self.store.check(self.table, self.tracker.observed_directions()))
                system, user = self._prompts()
                response = self.policy(system, user)
                if response is None:
                    return self.result("policy_stop")
                plan = self._answer(system, user, response)
                if plan is _TRUNCATED:
                    continue
                if plan is None:
                    action, actor = self.rng.choice(RANDOM_ACTIONS), "random"
                else:
                    action = plan[0]
                    self.queue.extend(plan[1:])
            stop = self._step(action, actor)
            if stop is not None:
                return self.result(stop)

    # --------------------------------------------------------------------------------------------- reflection
    def _prompts(self) -> tuple[str, str]:
        """The system prompt and user message of a reflection call made now."""
        semantics = self.tracker.semantics()
        kas = kas_text(semantics)
        budget = budget_line(self.level_step, MAX_STEPS_PER_LEVEL, len(self.calls), self.limits.max_reflection_calls)
        board = board_block(self.cur.view, level=self.level, level_total=self.level_total, semantics=semantics,
                            movement=self.movement.text, budget=budget)
        user = user_message(board=board, kas=kas,
                            showcase=showcase(self.table.transitions(), self.snapshots, self.cur.view.dims),
                            window=self.window.text(), rules=self.store.rules, notices=self.notices,
                            attempts=self.trials.section())
        return system_prompt(kas), user

    def _answer(self, system: str, user: str, response: object) -> tuple[int, ...] | None | object:
        """Record and apply one answer: the notices of the call replace the pending ones (rule-list notices,
        then plan notices), and a plan opens a trial. Returns the plan, or None for an answer without one.

        A `Completion` is recorded with its metadata and read as its text, except when it was cut off at the output
        limit: then nothing of it is applied, the only pending notice is `TRUNCATED_NOTICE`, and the result is
        `_TRUNCATED` (the call counts; no action follows)."""
        meta = None
        if isinstance(response, Completion):
            meta = response.meta()
            if response.truncated:
                self.calls.append(CallRecord(index=len(self.calls), level=self.level, level_step=self.level_step,
                                             system=system, user=user, response=response.text, plan=None,
                                             meta=meta))
                self.notices.clear()
                self.notices.append(TRUNCATED_NOTICE)
                return _TRUNCATED
            response = response.text
        if not isinstance(response, str):
            raise TypeError(f"policy returned {type(response).__name__}, expected str, Completion or None")
        parsed = parse_response(response)
        planned = parse_plan(parsed.plan_blocks)
        self.calls.append(CallRecord(index=len(self.calls), level=self.level, level_step=self.level_step,
                                     system=system, user=user, response=response, plan=planned.actions, meta=meta))
        self.notices.clear()
        self.notices.extend(self.store.apply_reflection(parsed.deletes, parsed.adds))
        self.notices.extend(planned.notices)
        if planned.actions is None:
            return None
        self.trials.open(planned.actions, start_key=self.cur.key, cap_note=planned.cap_note)
        return planned.actions

    # -------------------------------------------------------------------------------------------------- steps
    def _observe(self) -> _Observation:
        view = board_view(self.fd.truth)
        return _Observation(state_key(view), view)

    def _step(self, action: int, actor: Actor) -> StopReason | None:
        """Take one action (a RESET with nothing to undo is recorded but not sent to the engine) and update the
        harness state; return the stop reason if the episode ends with it."""
        noop = action == RESET and self.acts_since_reset == 0
        if noop:
            self.trials.note_noop_reset()
        self.steps.append(StepRecord(self.level, self.level_step, action, actor, noop))
        pre = self.cur
        completed = False
        if not noop:
            fd = self.game.step(action)
            previous = self.fd.levels_completed
            if fd.full_reset or fd.levels_completed not in (previous, previous + 1):
                raise RuntimeError(f"engine state out of step with the harness at level {self.level} "
                                   f"step {self.level_step}: full_reset={fd.full_reset}, "
                                   f"levels_completed {previous} -> {fd.levels_completed}")
            completed = fd.levels_completed > previous
            self.fd = fd
            self.cur = self._observe()
            self.acts_since_reset = 0 if completed or action == RESET else self.acts_since_reset + 1
        self.level_step += 1
        if action == RESET and not noop:
            self.resets.append(ResetRecord(self.level, self.level_step))
        if completed:
            return self._complete_level(completed_in=self.level_step)
        self._record(pre, action, self.cur, noop=noop, actor=actor)
        if action == CONFIRM:
            detail = "; ".join(submit_details(pre.view, self.cur.view))
            self.trials.note_rejected_confirm(remaining=len(self.queue), detail=detail)
        if self.level_step >= MAX_STEPS_PER_LEVEL:
            return "level_budget"
        return None

    def _record(self, pre: _Observation, action: int, post: _Observation, *, noop: bool, actor: Actor) -> None:
        """Update the level's memory after a step from `pre` to `post` that did not complete the level."""
        self.table.record(pre.key, action, post.key)
        self.window.add(self.level_step, action,
                        transition_line(action, pre.view, post.view, state_changed=pre.key != post.key,
                                        random_fallback=actor == "random"))
        self.snapshots[post.key] = post.view
        if noop:
            self.movement.clear()
        else:
            self.movement.update(post.view, action)
        self.tracker.record(action, pre.key, post.key, move_vote(pre.view, post.view), post.view.dims)

    # ------------------------------------------------------------------------------------------------- levels
    def _complete_level(self, completed_in: int) -> StopReason | None:
        """Count the completed level; stop, or enter the next level (`self.cur` is already its first state)."""
        self.level += 1
        if self.level >= self.limits.max_levels:
            return "max_levels"
        if self.fd.won:
            return "game_won"
        notice = self.store.carry_over()
        if notice is not None:
            self.notices.append(notice)
        self.tracker.new_level()
        self.trials.reset()
        self.queue.clear()
        self.level_step = 0
        self._begin_level()
        self.window.add_breadcrumb(breadcrumb(self.level, completed_in, self.level_total))
        return None

    def _begin_level(self) -> None:
        """Fresh per-level memory whose first state is `self.cur`."""
        self.table, self.window = TransitionTable(), ObservationWindow()
        self.snapshots = {self.cur.key: self.cur.view}
        self.movement.reset(self.cur.view)
