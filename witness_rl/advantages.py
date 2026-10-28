"""Broadcast advantages: one advantage per episode, which the trainer applies to every token of the episode's calls.

Only valid episodes (those that ended without an error) take part. The reward of each valid episode is centered on
the mean reward of the valid episodes of its group (its game), without dividing by the group's spread. The centered
values of all valid episodes of the step are then whitened together: minus their mean, divided by their population
standard deviation plus WHITEN_EPS; all become 0.0 when that deviation is below WHITEN_EPS (so a lone valid
episode gets 0.0). Invalid episodes get advantage 0.0.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

import numpy as np

WHITEN_EPS: Final = 1e-8


@dataclass(frozen=True)
class EpisodeScore:
    """What the advantage of one episode depends on."""

    group: str           # advantage group: the game id
    reward: float        # episode reward
    valid: bool          # False for an episode that ended with an error: it is left out of the group means and
                         # the whitening and gets advantage 0.0


def broadcast_advantages(episodes: Sequence[EpisodeScore]) -> list[float]:
    """One advantage per episode, aligned with `episodes`.

    Groups are formed in order of first appearance and the whitening runs over the valid episodes in the order
    given (numpy float64 mean and standard deviation), so the result is a deterministic function of the sequence.
    """
    valid = [i for i, e in enumerate(episodes) if e.valid]
    result = [0.0] * len(episodes)
    if not valid:
        return result
    groups: dict[str, list[int]] = {}             # group -> positions k in `valid`, in order of first appearance
    for k, i in enumerate(valid):
        groups.setdefault(episodes[i].group, []).append(k)
    centred = [0.0] * len(valid)
    for ks in groups.values():
        rewards = np.array([episodes[valid[k]].reward for k in ks], dtype=np.float64)
        for k, value in zip(ks, rewards - rewards.mean()):
            centred[k] = float(value)
    values = np.array(centred, dtype=np.float64)
    mean, std = values.mean(), values.std()
    if std < WHITEN_EPS:
        whitened = [0.0] * len(values)
    else:
        whitened = [float(x) for x in (values - mean) / (std + WHITEN_EPS)]
    for k, i in enumerate(valid):
        result[i] = whitened[k]
    return result
