"""Broadcast advantages of the episodes of a training step (`witness_rl.advantages`).

The batches of the golden fixture `rl/advantages.json` are compared bit for bit; every other value is written
here by hand.
"""
from __future__ import annotations

import random

import pytest

from golden import STORE, TRAINING_GAMES
from witness_rl.advantages import WHITEN_EPS, EpisodeScore, broadcast_advantages

MIXED = [EpisodeScore("tw01", 1.0, valid=True), EpisodeScore("tw01", 3.0, valid=True),
         EpisodeScore("tw02", 0.5, valid=True), EpisodeScore("tw02", 0.5, valid=True),
         EpisodeScore("tw03", 2.0, valid=True), EpisodeScore("tw03", 9.0, valid=False)]
MIXED_ADVANTAGES = [-1.58113880508419, 1.58113880508419, 0.0, 0.0, 0.0, 0.0]


@pytest.mark.parametrize("case", STORE.advantage_cases(), ids=lambda case: case.name)
def test_advantages_of_every_batch_of_the_golden_fixture(case):
    episodes = [EpisodeScore(game_id, reward, valid=abort is None) for game_id, reward, abort in case.episodes]
    assert broadcast_advantages(episodes) == list(case.expected)


def test_small_batch_centred_per_game_and_whitened_over_all_valid_episodes():
    assert broadcast_advantages(MIXED) == MIXED_ADVANTAGES


def test_one_game_of_eight_episodes():
    rewards = [7.56, 2.78, 1.57, 3.71, 6.58, 10.25, 3.67, 8.45]
    assert broadcast_advantages([EpisodeScore("tw01", r, valid=True) for r in rewards]) == [
        0.6922329618029592, -0.9715626673199296, -1.3927326905916233, -0.6478534758797023, 0.35112005039282734,
        1.628553096183832, -0.6617764518556261, 1.0020191772672626]


def test_equal_rewards_within_every_game_give_zero_advantages():
    # Three equal rewards whose float mean is not the reward itself leave centered values of about 1e-16; their
    # spread is below the whitening epsilon, so every advantage is exactly zero.
    episodes = [EpisodeScore(game_id, reward, valid=True)
                for game_id, reward in (("tw01", 0.1), ("tw02", 0.7), ("ft20", 0.985)) for _ in range(3)]
    assert broadcast_advantages(episodes) == [0.0] * 9
    episodes = [EpisodeScore(game_id, reward, valid=True)
                for game_id, reward in (("tw01", 0.985), ("tw02", -0.02), ("ft20", 4.845)) for _ in range(8)]
    assert broadcast_advantages(episodes) == [0.0] * 24


def test_spread_below_the_whitening_epsilon_gives_zero_advantages():
    assert 5.0e-10 < WHITEN_EPS
    episodes = [EpisodeScore("tw01", 1.0, valid=True), EpisodeScore("tw01", 1.0 + 1e-9, valid=True)]
    assert broadcast_advantages(episodes) == [0.0, 0.0]


def test_the_reward_of_an_invalid_episode_changes_nothing():
    replaced = MIXED[:5] + [EpisodeScore("tw03", 1e6, valid=False)]
    assert broadcast_advantages(replaced) == MIXED_ADVANTAGES


def test_a_single_valid_episode_gets_advantage_zero():
    episodes = [EpisodeScore("tw05", 3.67, valid=True)] + [EpisodeScore("tw05", 8.45, valid=False)] * 7
    assert broadcast_advantages(episodes) == [0.0] * 8


def test_no_valid_episode_gives_all_zeros_and_an_empty_batch_gives_nothing():
    assert broadcast_advantages([EpisodeScore("tw01", float(i), valid=False) for i in range(5)]) == [0.0] * 5
    assert broadcast_advantages([]) == []


def test_permuting_the_episodes_permutes_the_advantages():
    rng = random.Random(0)
    episodes = [EpisodeScore(game_id, round(rng.uniform(-1.5, 10.5), 3), valid=rng.random() > 0.05)
                for game_id in TRAINING_GAMES for _ in range(8)]
    order = list(range(len(episodes)))
    rng.shuffle(order)
    advantages = broadcast_advantages(episodes)
    permuted = broadcast_advantages([episodes[i] for i in order])
    assert len(episodes) == 176 and not all(e.valid for e in episodes)
    assert permuted == pytest.approx([advantages[i] for i in order], abs=1e-12, rel=0)
