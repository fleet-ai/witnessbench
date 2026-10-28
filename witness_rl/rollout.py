"""One training episode: the harness plays a game and the inference server answers every reflection call.

`run_rollout` runs the synchronous harness (`witness_agent.run_episode`) in a thread of its own. For each reflection
call the prompt is rendered with the model's chat template, tokenized and sent to the server from the event loop that
awaits `run_rollout`; only the episode's thread waits for the answer, so many episodes share one loop. Every episode
gets a thread of its own rather than a pool worker: all episodes of a training step run at the same time and each
holds its thread until it ends. The policy declines the call after `Task.max_reflection_calls` answered calls, which
ends the episode with stop reason "policy_stop". An exception raised while the episode runs (a request that failed
every attempt, a template error, a harness error) ends it early: the rollout keeps the calls answered so far, with
their rewards, and names the failure in `Rollout.abort`.
"""
from __future__ import annotations

import asyncio
import logging
import threading
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from witness_agent import EpisodeAborted, EpisodeLimits, EpisodeResult, Task, run_episode, write_trace
from witness_agent.catalog import make_game
from witness_rl.rewards import CallReward, call_rewards, episode_reward

logger = logging.getLogger(__name__)

AbortReason = Literal["agent_error", "inference_error"]
SglangRequest = Callable[[list[int]], Awaitable[dict]]      # prompt token ids -> the server's raw response


class InferenceError(Exception):
    """The inference server did not answer a request after all attempts."""


@dataclass(frozen=True)
class RolloutCall:
    """One answered reflection call: its prompt, the server's answer and its reward."""

    system: str
    user: str
    prompt_ids: list[int]           # token ids of the rendered prompt (`prompt_ids`)
    output: dict                    # the server's raw response: "text" and "meta_info" (sampled token log-probs,
                                    # finish reason, ...)
    finish_reason: str              # output["meta_info"]["finish_reason"]["type"]: "stop" (ended on a stop token),
                                    # "length" (token limit) or "abort"; any other value is carried as is
    level: int                      # levels completed when the call was made
    reward: CallReward

    @property
    def kept(self) -> bool:
        """True iff the call enters the gradient, i.e. the answer ended on a stop token."""
        return self.finish_reason == "stop"


@dataclass(frozen=True)
class Rollout:
    """One training episode as the trainer sees it."""

    task: Task
    seed: int                       # episode seed
    calls: tuple[RolloutCall, ...]  # every answered call in call order, kept or not
    levels_completed: int
    reward: float                   # episode reward: the call rewards added in call order
    stop_reason: str                # the episode's stop reason; "aborted" whenever `abort` is set
    abort: AbortReason | None       # "inference_error" when a request failed every attempt, "agent_error" for any
                                    # other failure, None for an episode that ended normally


@dataclass(frozen=True)
class _Answer:
    """One reflection call as the policy answered it."""

    system: str
    user: str
    prompt_ids: list[int]
    output: dict


def prompt_ids(tokenizer: Any, system: str, user: str) -> list[int]:
    """Token ids of one reflection call's prompt.

    The system prompt and the user message are rendered with the tokenizer's chat template, ending in the opening of
    the assistant turn, with thinking disabled (the template variable `enable_thinking=False`); the text is then
    tokenized without adding special tokens.
    """
    text = tokenizer.apply_chat_template(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False)
    return tokenizer(text, add_special_tokens=False)["input_ids"]


class _SglangPolicy:
    """The harness policy of one training episode; called in the episode's thread.

    Rendering, tokenization and the request run on the event loop, so the tokenizer is only ever used from that
    loop's thread. Answers call number `max_calls` + 1, and every call after `stop()`, with None (declined).
    """

    def __init__(self, loop: asyncio.AbstractEventLoop, tokenizer: Any, request: SglangRequest,
                 max_calls: int) -> None:
        self._loop, self._tokenizer, self._request, self._max_calls = loop, tokenizer, request, max_calls
        self._stopped = False
        self.answered: list[_Answer] = []

    def stop(self) -> None:
        """Decline every later call."""
        self._stopped = True

    def __call__(self, system: str, user: str) -> str | None:
        if self._stopped or len(self.answered) == self._max_calls:
            return None
        ids, output = asyncio.run_coroutine_threadsafe(self._complete(system, user), self._loop).result()
        self.answered.append(_Answer(system, user, ids, output))
        return output["text"]

    async def _complete(self, system: str, user: str) -> tuple[list[int], dict]:
        ids = prompt_ids(self._tokenizer, system, user)
        return ids, await self._request(ids)


async def run_rollout(task: Task, seed: int, *, tokenizer: Any, request: SglangRequest,
                      trace_dir: Path | None = None) -> Rollout:
    """Play one training episode of `task` with episode seed `seed` and return it.

    The episode stops after `task.max_levels` levels, when the game is won, when a level's step budget runs out, or
    when call number `task.max_reflection_calls` + 1 would be needed. `request` sends the prompt ids of one call to
    the inference server and returns its raw response; it raises `InferenceError` when it gives up. With
    `trace_dir`, the episode's trace (`witness_agent.write_trace`) is written to `<trace_dir>/<game_id>/seed<seed>/`;
    a trace that cannot be written (`OSError`) is logged and changes nothing else.

    Failures of the episode never raise: they become `Rollout.abort`, with the calls answered before the failure
    and their rewards (the last call's level gain is counted up to the failure). Only an inconsistency found when
    the rollout is assembled raises: `RuntimeError` when the harness's call records do not match the answered calls,
    and `KeyError`/`TypeError` for a response without `meta_info.finish_reason.type`. When the awaiting task is
    cancelled, the episode declines its next reflection call, which ends it, and its result is discarded.
    """
    loop = asyncio.get_running_loop()
    policy = _SglangPolicy(loop, tokenizer, request, task.max_reflection_calls)
    done: asyncio.Future[tuple[EpisodeResult | None, BaseException | None]] = loop.create_future()

    def play() -> None:                                     # the episode's thread
        result: EpisodeResult | None = None
        cause: BaseException | None = None
        try:
            result = run_episode(make_game(task.game_id), seed=seed, limits=EpisodeLimits(task.max_levels),
                                 policy=policy)
        except EpisodeAborted as exc:
            result, cause = exc.partial, exc.__cause__ or exc
        except Exception as exc:                            # the game could not be made: no episode at all
            cause = exc
        if trace_dir is not None and result is not None:
            directory = trace_dir / task.game_id / f"seed{seed}"
            try:
                write_trace(result, directory)
            except OSError:
                logger.error("trace not written game=%s seed=%s dir=%s", task.game_id, seed, directory,
                             exc_info=True)
            except Exception as exc:
                cause = cause or exc

        def deliver() -> None:                              # on the loop
            if not done.cancelled():
                done.set_result((result, cause))

        loop.call_soon_threadsafe(deliver)

    threading.Thread(target=play, name=f"episode-{task.game_id}-seed{seed}", daemon=True).start()
    try:
        result, cause = await done
    except asyncio.CancelledError:
        policy.stop()
        raise
    return _assemble(task, seed, result, cause, policy.answered)


def _assemble(task: Task, seed: int, result: EpisodeResult | None, cause: BaseException | None,
              answered: list[_Answer]) -> Rollout:
    """The rollout of a finished episode thread (`result` is None when no episode could start)."""
    if result is not None:
        records, levels_completed, stop_reason = result.calls, result.levels_completed, result.stop_reason
    else:
        records, levels_completed, stop_reason = (), 0, "aborted"
    if len(records) != len(answered) or any((r.system, r.user) != (a.system, a.user)
                                            for r, a in zip(records, answered, strict=True)):
        raise RuntimeError(f"game {task.game_id} seed {seed}: the {len(records)} call records of the harness do "
                           f"not match the {len(answered)} answered calls")
    rewards = call_rewards([r.level for r in records], levels_completed, task.max_levels)
    calls = tuple(RolloutCall(a.system, a.user, a.prompt_ids, a.output, a.output["meta_info"]["finish_reason"]["type"],
                              r.level, reward) for r, a, reward in zip(records, answered, rewards, strict=True))
    abort: AbortReason | None = None
    if cause is not None:
        abort = "inference_error" if isinstance(cause, InferenceError) else "agent_error"
        logger.warning("episode aborted game=%s seed=%s calls=%d abort=%s", task.game_id, seed, len(calls), abort,
                       exc_info=cause)
        stop_reason = "aborted"
    return Rollout(task, seed, calls, levels_completed, episode_reward(rewards), stop_reason, abort)
