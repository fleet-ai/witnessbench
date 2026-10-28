"""Per-call rewards and episode rewards of training episodes (`witness_rl.rewards`).

Hand-derived values are written here; the recorded and scripted training episodes come from the golden episode
fixtures (`agent/episodes/`), whose expected rewards are compared bit for bit.
"""
from __future__ import annotations

import pytest

from golden import STORE
from witness_rl.rewards import COMPLETION_BONUS, STEP_PENALTY, CallReward, call_rewards, episode_reward

TRAINING_EPISODES = [name for name in STORE.episode_names() if STORE.episode(name).wiring == "rl"]


def totals(rewards: list[CallReward]) -> list[float]:
    return [r.total for r in rewards]


def test_every_call_pays_the_step_penalty_and_earns_the_levels_completed_before_the_next_call():
    rewards = call_rewards([0, 0, 1], 1, 10)
    assert totals(rewards) == [-0.005, 0.995, -0.005]
    assert [r.level_gain for r in rewards] == [0, 1, 0]
    assert [r.completion_bonus for r in rewards] == [0.0, 0.0, 0.0]
    assert [r.step_penalty for r in rewards] == [STEP_PENALTY] * 3 == [-0.005] * 3


def test_the_call_after_which_max_levels_is_reached_earns_the_completion_bonus():
    assert call_rewards([9], 10, 10) == [CallReward(-0.005, 1, 0.5, 1.495)]
    assert call_rewards([8], 10, 10) == [CallReward(-0.005, 2, 0.5, 2.495)]
    assert COMPLETION_BONUS == 0.5


def test_a_game_with_five_levels_never_earns_the_bonus_at_ten_max_levels():
    rewards = call_rewards([0, 1, 2, 3, 4], 5, 10)
    assert totals(rewards) == [0.995] * 5
    assert episode_reward(rewards) == 4.975


def test_ten_levels_in_twelve_calls_add_up_in_call_order():
    levels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10]
    rewards = call_rewards(levels, 10, 10)
    assert [r.completion_bonus for r in rewards] == [0.0] * 9 + [0.5, 0.0, 0.0]
    assert episode_reward(rewards) == 10.439999999999998


def test_call_total_adds_the_step_penalty_to_the_sum_of_level_gain_and_bonus():
    for gain in range(11):
        cases = [(call_rewards([0], gain, 100)[0], 0.0)]                # bonus out of reach
        if gain:
            cases.append((call_rewards([10 - gain], 10, 10)[0], 0.5))    # this call reaches max_levels
        for reward, bonus in cases:
            assert (reward.level_gain, reward.completion_bonus) == (gain, bonus)
            assert reward.total == STEP_PENALTY + (float(gain) + bonus)
    # The one gain/bonus pair where the order of the two additions changes the float result.
    assert (STEP_PENALTY + 8.0) + 0.5 == 8.495000000000001
    assert call_rewards([2], 10, 10)[0].total == 8.495


def test_an_episode_without_calls_has_no_call_rewards_and_reward_zero():
    assert call_rewards([], 0, 10) == []
    assert episode_reward([]) == 0.0


def test_three_hundred_planless_calls_add_up_one_at_a_time():
    assert episode_reward(call_rewards([0] * 300, 0, 10)) == -1.49999999999999


@pytest.mark.parametrize("name", TRAINING_EPISODES)
def test_rewards_of_every_training_episode_equal_the_golden_episode_fixture(name):
    ep = STORE.episode(name)
    rewards = call_rewards([c.level for c in ep.calls], ep.levels_completed, ep.max_levels)
    assert [(r.total, r.step_penalty, r.level_gain, r.completion_bonus) for r in rewards] == \
        [(e.total, e.step_penalty, e.level_gain, e.completion_bonus) for e in ep.rewards]
    reward = episode_reward(rewards)
    assert reward == ep.episode_reward
    bonus = COMPLETION_BONUS if ep.levels_completed >= ep.max_levels else 0.0
    assert abs(reward - (ep.levels_completed + bonus + STEP_PENALTY * len(ep.calls))) < 1e-9
