"""Rewards of the reflection calls of one training episode.

The reward of every answered call is STEP_PENALTY (-0.005), plus the number of levels completed between it and the
next call (or the end of the episode), plus COMPLETION_BONUS if the episode reaches its task row's `max_levels`
before the next call. The episode reward is the sum of the call rewards, added in call order. Calls that the
trainer later drops from the gradient are still part of this sum.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

STEP_PENALTY: Final = -0.005
COMPLETION_BONUS: Final = 0.5


@dataclass(frozen=True)
class CallReward:
    """The reward of one answered call and its three parts."""

    step_penalty: float          # always STEP_PENALTY
    level_gain: int              # levels completed between this call and the next call (or the episode end)
    completion_bonus: float      # COMPLETION_BONUS on the call after which max_levels is reached, else 0.0
    total: float                 # STEP_PENALTY + (float(level_gain) + completion_bonus), added in this order


def call_rewards(levels: Sequence[int], final_level: int, max_levels: int) -> list[CallReward]:
    """The reward of every call, in call order.

    `levels[i]` is the number of levels completed when call i was made, `final_level` the number completed when
    the episode ended, and `max_levels` the task row's level limit.
    """
    rewards = []
    for i, before in enumerate(levels):
        after = levels[i + 1] if i + 1 < len(levels) else final_level
        gain = after - before
        bonus = COMPLETION_BONUS if (after >= max_levels and before < max_levels) else 0.0
        rewards.append(CallReward(STEP_PENALTY, gain, bonus, STEP_PENALTY + (float(gain) + bonus)))
    return rewards


def episode_reward(rewards: Sequence[CallReward]) -> float:
    """The call totals added one at a time in call order, starting from 0.0: plain float64 addition (the built-in
    sum() compensates rounding errors from Python 3.12 on, so its result can differ in the last bits)."""
    total = 0.0
    for reward in rewards:
        total += reward.total
    return total
