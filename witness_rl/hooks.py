"""The two functions the miles trainer calls: the per-sample rollout function plays one training episode,
`reward_post_process` turns the episodes of a training step into broadcast advantages.

miles hands the rollout function one sample per episode: its metadata is the task row and its index is the episode
seed. Every answered reflection call becomes one training sample (prompt ids + sampled ids) carrying the episode
reward; calls whose answer did not end on a stop token are left out, and all samples of an episode share one rollout
id, so the trainer normalizes the loss per episode. `reward_post_process` receives the flat sample list of the step
and returns, per sample, the episode reward and the episode's advantage (`witness_rl.advantages`).

This is the only module of the package that imports miles.
"""
from __future__ import annotations

import asyncio
import logging
from argparse import Namespace
from collections import Counter
from pathlib import Path
from typing import Final

from miles.rollout.base_types import GenerateFnInput, GenerateFnOutput
from miles.rollout.generate_utils.generate_endpoint_utils import compute_request_payload, update_sample_from_response
from miles.utils.http_utils import post
from miles.utils.types import Sample

from witness_agent import Task
from witness_rl.advantages import EpisodeScore, broadcast_advantages
from witness_rl.rollout import AbortReason, InferenceError, Rollout, SglangRequest, run_rollout

logger = logging.getLogger(__name__)

REQUEST_TIMEOUT_S: Final = 1200.0       # time limit of one request attempt
REQUEST_ATTEMPTS: Final = 3             # attempts per reflection call before the episode ends with an inference error
                                        # (each attempt is one call of miles' post, which already retries every
                                        # exception 60 times, 1 s apart)


async def generate(input: GenerateFnInput) -> GenerateFnOutput:     # parameter named as in miles' interface
    """Play the training episode of one miles sample and return its training samples.

    The episode plays `Task.from_metadata(input.sample.metadata)` with seed `input.sample.index` against the
    trainer's inference server. It returns one sample per answered call whose answer ended on a stop token (the
    sample's `index` is the call's number among all answered calls); if every call was cut off, one sample of the
    first call with an all-zero loss mask, so the episode still counts in its game's group. Every sample has
    `rollout_id = input.sample.index` and reward = the episode reward. Traces are written when the trainer's
    arguments carry `witness_trace_dir`.

    Never raises: an episode that failed after at least one answered call is returned with its samples and its
    abort type in the metadata; an episode without any answered call (or any unexpected error) returns
    `[input.sample]` as a marker (reward 0.0, no tokens, the abort type in the metadata), which
    `reward_post_process` rejects.
    """
    sample = input.sample
    try:
        task = Task.from_metadata(sample.metadata)
        trace_root = getattr(input.args, "witness_trace_dir", None)
        rollout = await run_rollout(task, sample.index, tokenizer=input.state.tokenizer,
                                    request=_sglang_request(input.args, input.sampling_params),
                                    trace_dir=Path(trace_root) if trace_root else None)
        samples = await _episode_samples(input.args, rollout, rollout_id=sample.index)
    except Exception:
        logger.exception("episode seed=%s failed; returned as an episode without an answered call", sample.index)
        return GenerateFnOutput(samples=[_no_call_marker(sample, "agent_error")])
    if not samples:                                         # only an abort ends an episode before its first call
        return GenerateFnOutput(samples=[_no_call_marker(sample, rollout.abort or "agent_error")])
    dropped = samples[0].metadata["dropped_calls"]
    logger.info("episode game=%s seed=%s calls=%d kept=%d dropped=%d levels=%d reward=%.3f stop=%s%s",
                rollout.task.game_id, rollout.seed, len(rollout.calls), len(rollout.calls) - dropped, dropped,
                rollout.levels_completed, rollout.reward, rollout.stop_reason,
                f" abort={rollout.abort}" if rollout.abort else "")
    return GenerateFnOutput(samples=samples)


def reward_post_process(args: Namespace, samples: list[Sample]) -> tuple[list[float], list[float]]:
    """(raw rewards, advantages) of the flat sample list of one training step, one entry per sample.

    Samples are grouped into episodes by `rollout_id` in order of first appearance. The raw reward of a sample is
    its episode reward; its advantage is the episode's broadcast advantage (`witness_rl.advantages`): an episode is
    valid iff its metadata `episode_abort` is empty, and its advantage group is its game. Aborted episodes get 0.0,
    also when every episode of a game or of the step aborted (a warning names each game without a valid episode).

    Raises ValueError for a sample without `rollout_id` and when a game does not have exactly
    `args.n_samples_per_prompt` episodes, and RuntimeError naming game, seed and abort type of every episode without
    an answered call (the marker sample returned for such an episode), which cannot be trained on.
    """
    episodes: dict[int, list[int]] = {}                     # rollout id -> sample positions, first appearance
    for pos, s in enumerate(samples):
        if s.rollout_id is None:
            raise ValueError(f"sample {pos} has no rollout_id")
        episodes.setdefault(s.rollout_id, []).append(pos)
    heads = [samples[positions[0]] for positions in episodes.values()]
    aborts: dict[str, list[str]] = {}                       # game -> `episode_abort` of each of its episodes
    for h in heads:
        aborts.setdefault(h.metadata["game_id"], []).append(h.metadata["episode_abort"])
    wrong = {game: len(types) for game, types in aborts.items() if len(types) != args.n_samples_per_prompt}
    if wrong:
        raise ValueError(f"episodes per game != {args.n_samples_per_prompt}: {wrong} (every game needs "
                         f"n_samples_per_prompt = {args.n_samples_per_prompt} episodes in a step)")
    _check_trainable(heads)
    scores = [EpisodeScore(h.metadata["game_id"], h.reward, h.metadata["episode_abort"] == "") for h in heads]
    advantages = [0.0] * len(samples)
    for advantage, positions in zip(broadcast_advantages(scores), episodes.values(), strict=True):
        for pos in positions:
            advantages[pos] = advantage
    logger.info("advantages episodes=%d valid=%d aborted=%s games=%d", len(heads), sum(e.valid for e in scores),
                dict(Counter(h.metadata["episode_abort"] for h in heads if h.metadata["episode_abort"])), len(aborts))
    for game in sorted(aborts):
        if all(aborts[game]):
            logger.warning("no valid episode game=%s aborted=%s; its samples get advantage 0", game,
                           dict(Counter(aborts[game])))
    return [s.reward for s in samples], advantages


def _sglang_request(args: Namespace, sampling_params: dict) -> SglangRequest:
    """The request function of one episode: the prompt ids go to the trainer's inference router with miles'
    request body; up to REQUEST_ATTEMPTS attempts of at most REQUEST_TIMEOUT_S each (any exception is retried),
    then InferenceError. Each attempt is one call of miles' post, which already retries every exception 60 times,
    1 s apart; this loop adds two more rounds, each bounded by REQUEST_TIMEOUT_S."""
    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"

    async def request(prompt_ids: list[int]) -> dict:
        payload, _ = compute_request_payload(args, prompt_ids, sampling_params=sampling_params)
        attempt = 1
        while True:
            try:
                return await asyncio.wait_for(post(url, payload), timeout=REQUEST_TIMEOUT_S)
            except Exception as exc:
                if attempt == REQUEST_ATTEMPTS:
                    raise InferenceError(f"{url}: {type(exc).__name__}: {exc}") from exc
                logger.warning("request attempt %d/%d failed: %s: %s", attempt, REQUEST_ATTEMPTS,
                               type(exc).__name__, exc)
            attempt += 1

    return request


async def _episode_samples(args: Namespace, rollout: Rollout, rollout_id: int) -> list[Sample]:
    """The training samples of an episode: one per kept call, or one zero-mask sample of call 0 when no call was
    kept; [] for an episode without calls."""
    kept = [i for i, call in enumerate(rollout.calls) if call.kept]
    dropped = len(rollout.calls) - len(kept)
    if rollout.calls and not kept:
        representative = await _call_sample(args, rollout, 0, rollout_id, dropped)
        representative.loss_mask = [0] * representative.response_length
        return [representative]
    return [await _call_sample(args, rollout, i, rollout_id, dropped) for i in kept]


async def _call_sample(args: Namespace, rollout: Rollout, index: int, rollout_id: int, dropped: int) -> Sample:
    """The training sample of call number `index`: its prompt ids followed by the sampled ids, the sampled
    log-probabilities, loss mask 1 on every sampled token, the status from the finish reason and the episode
    reward."""
    call = rollout.calls[index]
    sample = Sample(index=index, rollout_id=rollout_id, prompt="")
    await update_sample_from_response(args, sample, payload={"input_ids": list(call.prompt_ids)}, output=call.output,
                                      update_loss_mask=True)
    sample.reward = rollout.reward
    sample.metadata = {"game_id": rollout.task.game_id, "call_index": index, "episode_reward": rollout.reward,
                       "episode_abort": rollout.abort or "", "finish_reason": call.finish_reason,
                       "levels_completed": rollout.levels_completed, "dropped_calls": dropped}
    return sample


def _no_call_marker(sample: Sample, abort: AbortReason) -> Sample:
    """`sample` (miles' input sample) as the marker of an episode without an answered call: rollout id = its index,
    reward 0.0 and, added to its metadata, the abort type; it has no tokens."""
    sample.rollout_id = sample.index
    sample.reward = 0.0
    sample.metadata = {**sample.metadata, "episode_reward": 0.0, "episode_abort": abort, "levels_completed": 0,
                       "dropped_calls": 0}
    return sample


def _check_trainable(heads: list[Sample]) -> None:
    """Raise RuntimeError naming every episode (by its first sample) that is a marker: no answered call, no
    tokens."""
    no_call = [h for h in heads if not h.tokens]
    if no_call:
        raise RuntimeError(
            f"{len(no_call)} episode(s) have no answered call and cannot be trained on "
            "(fix the cause, then re-run to resume from the last checkpoint): "
            + ", ".join(f"{h.metadata['game_id']} seed {h.index} ({h.metadata['episode_abort']})" for h in no_call))
