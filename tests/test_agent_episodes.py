"""Task rows, episode limits, the episode loop and its trace files (`witness_agent.episode`, `witness_agent.loop`,
`witness_agent.trace`).

`agent/episodes/<name>` holds episodes of the training games with their inputs (game, seed, limits, the policy's
answers in order) and every expected prompt, action, executed RESET, level count and stop reason; the replay test
plays each one through `run_episode` with `golden.ScriptedPolicy`. Episodes with a smaller call budget are derived
from a full training episode. `agent/rng.json` holds the first random-fallback moves of a few (game, seed) pairs,
and `env/solutions.json` the stored solutions of the training levels that the hand-written episodes play. The task
files are the shipped `data/train.jsonl` and `data/val.jsonl`; every other value is written by hand.
"""
from __future__ import annotations

import dataclasses
import json
import random
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from arcengine import ActionInput, GameAction

from golden import STORE, TRAINING_GAMES, VALIDATION_GAMES, Episode, ScriptedPolicy, fallback_moves
from witness_agent import EpisodeAborted, EpisodeLimits, EpisodeResult, Task, run_episode, write_trace
from witness_agent.episode import CallRecord, ResetRecord, StepRecord
from witness_agent.loop import MAX_STEPS_PER_LEVEL
from witness_env import GAME_IDS, make_game

DATA = Path(__file__).resolve().parent.parent / "data"


def read_tasks(name: str) -> list[Task]:
    return [Task.from_row(json.loads(line)) for line in (DATA / name).read_text(encoding="utf-8").splitlines()]


def test_training_rows_are_the_training_games_in_order_with_ten_levels_and_300_calls():
    assert read_tasks("train.jsonl") == [Task(game_id, 10, 300) for game_id in TRAINING_GAMES]


def test_validation_rows_are_the_validation_games_in_order_with_their_level_count_and_250_calls():
    assert read_tasks("val.jsonl") == [Task(game_id, 5, 250) for game_id in VALIDATION_GAMES]


def _row(prompt: str = "tw01", drop: tuple[str, ...] = (), **metadata) -> dict:
    """A training row for tw01 with the metadata keys in `drop` removed and `metadata` added or replaced."""
    md = {"game_id": "tw01", "max_levels": 10, "max_reflection_calls": 300, **metadata}
    return {"prompt": prompt, "metadata": {k: v for k, v in md.items() if k not in drop}}


@pytest.mark.parametrize("row, problem", [
    ({"metadata": _row()["metadata"]}, "missing keys ['prompt']"),
    ({**_row(), "comment": "x"}, "unexpected keys ['comment']"),
    (_row(drop=("max_reflection_calls",)), "metadata: missing keys ['max_reflection_calls']"),
    (_row(difficulty=1), "metadata: unexpected keys ['difficulty']"),
    (_row(max_levels=True), "max_levels must be an integer >= 1, got True"),
    (_row(max_levels=0), "max_levels must be an integer >= 1, got 0"),
    (_row(max_levels="10"), "max_levels must be an integer >= 1, got '10'"),
    (_row(max_reflection_calls=0), "max_reflection_calls must be an integer >= 1, got 0"),
    (_row(max_reflection_calls=True), "max_reflection_calls must be an integer >= 1, got True"),
    (_row(max_reflection_calls="300"), "max_reflection_calls must be an integer >= 1, got '300'"),
    (_row(prompt="no_such_game", game_id="no_such_game"), "game_id must be a known game id, got 'no_such_game' "
     "(WITNESS_GAME_CATALOG is unset: public games only)"),
    (_row(prompt="tw02"), "prompt 'tw02' differs from game_id 'tw01'"),
    (["tw01", 10, 300], "expected an object with the keys ['prompt', 'metadata'], got list"),
    ({"prompt": "tw01", "metadata": "tw01"},
     "metadata: expected an object with the keys ['game_id', 'max_levels', 'max_reflection_calls'], got str"),
], ids=["no_prompt", "extra_row_key", "no_call_budget", "extra_metadata_key", "levels_bool", "levels_zero",
        "levels_str", "calls_zero", "calls_bool", "calls_str", "unknown_game", "prompt_differs", "row_not_an_object",
        "metadata_not_an_object"])
def test_malformed_task_rows_raise_value_error_quoting_the_row(row, problem):
    with pytest.raises(ValueError) as excinfo:
        Task.from_row(row)
    assert str(excinfo.value) == f"malformed task row {row!r}: {problem}"


@pytest.mark.parametrize("args, problem", [
    ((0,), "max_levels must be an integer >= 1, got 0"),
    ((True,), "max_levels must be an integer >= 1, got True"),
    ((5, 0), "max_reflection_calls must be an integer >= 1, got 0"),
    ((5, -1), "max_reflection_calls must be an integer >= 1, got -1"),
    ((5, True), "max_reflection_calls must be an integer >= 1, got True"),
    ((5, 2.5), "max_reflection_calls must be an integer >= 1, got 2.5"),
], ids=["levels_zero", "levels_bool", "calls_zero", "calls_negative", "calls_bool", "calls_float"])
def test_episode_limits_reject_values_below_one_booleans_and_non_integers(args, problem):
    with pytest.raises(ValueError, match=problem):
        EpisodeLimits(*args)


def test_episode_limits_have_no_call_cap_unless_one_is_given():
    no_cap, capped = EpisodeLimits(5), EpisodeLimits(5, 250)
    assert (no_cap.max_levels, no_cap.max_reflection_calls) == (5, None)
    assert (capped.max_levels, capped.max_reflection_calls) == (5, 250)


# -------------------------------------------------------------------------------------------- episode helpers
SOLUTIONS = STORE.solutions()
TW01_LEVEL_1 = SOLUTIONS["tw01"][0]                     # the stored solution of tw01's first level (9 actions)


def plan(actions) -> str:
    """An answer that keeps the rule list and asks for `actions`."""
    return "<meta>x</meta><keep/><plan>" + ",".join(map(str, actions)) + "</plan>"


class AnswersThenDeclines:
    """A policy that gives `answers` in order and then declines; `prompts` keeps every (system, user) it was
    asked, the declined one included."""

    def __init__(self, answers) -> None:
        self.answers = list(answers)
        self.prompts: list[tuple[str, str]] = []

    def __call__(self, system: str, user: str) -> str | None:
        self.prompts.append((system, user))
        i = len(self.prompts) - 1
        return self.answers[i] if i < len(self.answers) else None


class AnswersThenRaises:
    """A policy that gives `answers` in order and then raises `error`."""

    def __init__(self, answers, error: Exception) -> None:
        self.answers, self.error = list(answers), error

    def __call__(self, system: str, user: str) -> str:
        if not self.answers:
            raise self.error
        return self.answers.pop(0)


def always_empty(system: str, user: str) -> str:
    return ""


def play(game_id: str, answers, *, seed: int = 0, max_levels: int = 10,
         cap: int | None = None) -> tuple[EpisodeResult, AnswersThenDeclines]:
    policy = AnswersThenDeclines(answers)
    result = run_episode(make_game(game_id), seed=seed, limits=EpisodeLimits(max_levels, cap), policy=policy)
    return result, policy


def steps_of(result: EpisodeResult) -> list[tuple]:
    return [dataclasses.astuple(s) for s in result.steps]


def resets_of(result: EpisodeResult) -> list[tuple]:
    return [dataclasses.astuple(r) for r in result.resets]


def lines_starting(text: str, prefix: str) -> list[str]:
    return [line for line in text.split("\n") if line.startswith(prefix)]


def notice_lines(user: str) -> list[str]:
    """The notice lines of a user message (none if it has no notices section)."""
    lines = user.split("\n")
    if "## Harness notices (since your last reflection)" not in lines:
        return []
    after = lines[lines.index("## Harness notices (since your last reflection)") + 1:]
    return after[:next(i for i, line in enumerate(after) if not line.startswith("- "))]


# -------------------------------------------------------------------------------------------- stored episodes
def replay(ep: Episode) -> None:
    """Play `ep` with its answers and check every prompt, call, action, executed RESET, level count and the stop."""
    limits = EpisodeLimits(ep.max_levels, ep.max_reflection_calls if ep.wiring == "eval" else None)
    policy = ScriptedPolicy(ep)
    result = run_episode(make_game(ep.game_id), seed=ep.seed, limits=limits, policy=policy)
    assert policy.mismatch is None, policy.mismatch
    assert policy.calls_made == len(ep.calls) + (ep.final_prompt is not None)
    assert [(c.index, c.system, c.user, c.response, c.level, c.level_step, c.plan) for c in result.calls] == \
           [(i, e.system, e.user, ep.responses[i], e.level, e.level_step, e.plan) for i, e in enumerate(ep.calls)]
    assert steps_of(result) == list(ep.steps)
    assert resets_of(result) == list(ep.resets)
    assert (result.levels_completed, result.stop_reason) == (ep.levels_completed, ep.stop_reason)


@pytest.mark.parametrize("name", STORE.episode_names())
def test_every_stored_episode_replays_with_identical_prompts_actions_resets_and_stop(name):
    replay(STORE.episode(name))


FULL_EPISODE = "rl_tw03_s1041_full"                     # a whole training episode: 50 calls, 10 levels


def step_index(ep: Episode, level: int, step: int) -> int:
    return next(i for i, s in enumerate(ep.steps) if s[:2] == (level, step))


def resets_within(ep: Episode, n: int) -> tuple[tuple[int, int], ...]:
    """The executed RESETs among the first `n` actions of `ep`."""
    taken = {s[:2] for s in ep.steps[:n]}
    return tuple(r for r in ep.resets if (r[0], r[1] - 1) in taken)


def with_call_budget(ep: Episode, k: int) -> Episode:
    """`ep` (a whole training episode) with a training call budget of `k` calls: the policy declines call k + 1,
    which the uncapped episode asks at the step that takes that call's first action."""
    n = step_index(ep, ep.calls[k].level, ep.calls[k].level_step)
    return dataclasses.replace(ep, responses=ep.responses[:k], calls=ep.calls[:k],
                               final_prompt=(ep.calls[k].system, ep.calls[k].user), steps=ep.steps[:n],
                               resets=resets_within(ep, n), levels_completed=ep.calls[k].level,
                               stop_reason="policy_stop")


def with_call_cap(ep: Episode, k: int) -> Episode:
    """The answers of `ep` (a whole training episode) played under an evaluation call cap of `k`: the cap is checked
    when a new call is due, so the plan of call k runs to its end and the episode stops where the uncapped episode
    asks call k + 1. The prompts are the uncapped ones with the cap in their budget line."""
    n = step_index(ep, ep.calls[k].level, ep.calls[k].level_step)
    calls = []
    for i, call in enumerate(ep.calls[:k]):
        uncapped = f"| reflection calls used {i} (no limit)\n"
        assert call.user.count(uncapped) == 1, (i, "budget line")
        calls.append(dataclasses.replace(call, user=call.user.replace(uncapped,
                                                                      f"| reflection calls used {i} of {k}\n")))
    return dataclasses.replace(ep, wiring="eval", max_reflection_calls=k, calls=tuple(calls), final_prompt=None,
                               steps=ep.steps[:n], resets=resets_within(ep, n), levels_completed=ep.calls[k].level,
                               stop_reason="call_budget")


@pytest.mark.parametrize("k, steps, levels, resets", [
    (1, 1, 0, 0), (4, 4, 0, 0), (5, 7, 1, 0), (17, 66, 4, 0), (18, 74, 4, 0), (49, 249, 9, 2)])
def test_a_training_call_budget_ends_the_episode_when_the_next_call_is_needed(k, steps, levels, resets):
    ep = with_call_budget(STORE.episode(FULL_EPISODE), k)
    assert (len(ep.steps), ep.levels_completed, len(ep.resets)) == (steps, levels, resets)
    replay(ep)


@pytest.mark.parametrize("k, steps, levels, resets", [
    (1, 1, 0, 0), (5, 7, 1, 0), (18, 74, 4, 0), (19, 82, 4, 1), (30, 146, 7, 2)])
def test_the_evaluation_call_cap_lets_the_last_plan_run_to_its_end(k, steps, levels, resets):
    ep = with_call_cap(STORE.episode(FULL_EPISODE), k)
    assert (len(ep.steps), ep.levels_completed, len(ep.resets)) == (steps, levels, resets)
    replay(ep)


# -------------------------------------------------------------------------------------------- random fallback
@pytest.mark.parametrize("case", STORE.rng_cases(), ids=lambda c: f"{c.game_id}-seed{c.seed}")
def test_random_fallback_moves_come_from_the_random_source_of_the_game_and_seed(case):
    result, _ = play(case.game_id, [""] * len(case.draws), seed=case.seed)
    assert [(s.action, s.actor) for s in result.steps] == [(a, "random") for a in case.draws]
    assert fallback_moves(case.game_id, case.seed, len(case.draws)) == list(case.draws)


def test_answers_without_a_plan_take_random_moves_until_the_level_budget_ends_the_episode():
    result = run_episode(make_game("tw01"), seed=0, limits=EpisodeLimits(10), policy=always_empty)
    assert MAX_STEPS_PER_LEVEL == 300
    assert [(c.level, c.level_step, c.response, c.plan) for c in result.calls] == [(0, i, "", None) for i in range(300)]
    assert steps_of(result) == [(0, i, a, "random", False) for i, a in enumerate(fallback_moves("tw01", 0, 300))]
    assert [s.action for s in result.steps[:12]] == [2, 1, 3, 2, 3, 2, 2, 2, 1, 3, 4, 3]
    assert (result.resets, result.levels_completed, result.stop_reason) == ((), 0, "level_budget")


# --------------------------------------------------------------------------------------- budgets and stopping
FILLER_291 = [plan([0] * 16)] * 18 + [plan([0] * 3)]    # 291 RESETs with nothing to undo: counted, never sent
FILLER_292 = [plan([0] * 16)] * 18 + [plan([0] * 4)]


def test_a_level_completed_on_its_last_budgeted_action_counts_as_completed():
    result, policy = play("tw01", FILLER_291 + [plan(TW01_LEVEL_1)])
    assert (len(result.calls), len(result.steps), {s.level for s in result.steps}) == (20, 300, {0})
    assert steps_of(result)[-1] == (0, 299, 5, "plan", False)
    assert (result.resets, result.levels_completed, result.stop_reason) == ((), 1, "policy_stop")
    assert len(policy.prompts) == 21 and lines_starting(policy.prompts[-1][1], "Level: ") == ["Level: 2/10"]


def test_the_level_budget_ends_the_episode_one_action_before_a_late_completion():
    result, _ = play("tw01", FILLER_292 + [plan(TW01_LEVEL_1)])
    assert (len(result.calls), len(result.steps)) == (20, 300)
    assert steps_of(result)[-1] == (0, 299, 4, "plan", False)
    assert (result.resets, result.levels_completed, result.stop_reason) == ((), 0, "level_budget")


def chunked_plans(actions) -> list[str]:
    """Answers that play `actions` in plans of at most 16 actions (the plan cap)."""
    return [plan(actions[i:i + 16]) for i in range(0, len(actions), 16)]


# tw09's stored solutions in a row in plans of at most 16 actions; the CONFIRM of the last level comes alone, last.
TW09_ANSWERS = ([answer for solution in SOLUTIONS["tw09"][:9] for answer in chunked_plans(solution)]
                + chunked_plans(SOLUTIONS["tw09"][9][:-1]) + [plan([5])])


@pytest.mark.parametrize("max_levels, cap, calls, steps, levels, stop", [
    (20, 14, 14, 134, 10, "game_won"),
    (10, 14, 14, 134, 10, "max_levels"),
    (20, None, 14, 134, 10, "game_won"),
    (20, 13, 13, 133, 9, "call_budget"),
], ids=["cap_reached_by_the_winning_call", "level_limit_reached_with_the_win", "no_cap",
        "cap_reached_before_the_last_plan"])
def test_the_last_level_stops_the_episode_before_the_call_cap_is_checked(max_levels, cap, calls, steps, levels,
                                                                           stop):
    result, policy = play("tw09", TW09_ANSWERS, max_levels=max_levels, cap=cap)
    assert (len(result.calls), len(result.steps), result.levels_completed, result.stop_reason) == \
           (calls, steps, levels, stop)
    assert lines_starting(policy.prompts[0][1], "Level: ") == ["Level: 1/10"]


@pytest.mark.parametrize("game_id, max_levels, header", [
    ("tw08", 10, "Level: 1/5"), ("tw01", 3, "Level: 1/3"), ("tw12", 5, "Level: 1/5"), ("ft01", 10, "Level: 1/10"),
], ids=["game_with_five_levels", "level_limit_of_three", "game_with_as_many_levels_as_the_limit",
        "game_with_ten_levels"])
def test_the_level_total_is_the_smaller_of_the_game_levels_and_the_level_limit(game_id, max_levels, header):
    _, policy = play(game_id, [], max_levels=max_levels)
    assert lines_starting(policy.prompts[0][1], "Level: ") == [header]


def test_declining_the_first_call_ends_the_episode_before_any_action():
    result, policy = play("tw01", [])
    assert (result.calls, result.steps, result.resets, result.levels_completed, result.stop_reason) == \
           ((), (), (), 0, "policy_stop")
    assert len(policy.prompts) == 1


# ------------------------------------------------------------------------------------------ RESET and CONFIRM
def test_a_reset_with_nothing_to_undo_is_not_sent_to_the_engine_but_counts_as_an_action():
    result, _ = play("tw01", [plan([0]), plan([1, 0]), plan([0])])
    assert steps_of(result) == [(0, 0, 0, "plan", True), (0, 1, 1, "plan", False), (0, 2, 0, "plan", False),
                                (0, 3, 0, "plan", True)]
    assert resets_of(result) == [(0, 3)]
    assert result.stop_reason == "policy_stop"


def test_a_rejected_confirm_leaves_the_rest_of_the_plan_running():
    result, policy = play("tw01", [plan([5]), plan([5, 1])])
    assert [(s.action, s.actor) for s in result.steps] == [(5, "plan"), (5, "plan"), (1, "plan")]
    assert len(result.calls) == 2
    assert "- attempt #1: [5] → submit rejected at action 1/1" in policy.prompts[1][1].split("\n")
    assert "- attempt #2: [5,1] → submit rejected at action 1/2" in policy.prompts[2][1].split("\n")


# ---------------------------------------------------------------------------------------------- level changes
def test_plan_actions_after_the_winning_confirm_are_dropped_and_the_next_level_opens_with_a_level_complete_line():
    result, policy = play("tw01", [plan(TW01_LEVEL_1 + [1, 1])])
    assert len(result.steps) == 9 and steps_of(result)[-1] == (0, 8, 5, "plan", False)
    assert (result.levels_completed, result.stop_reason) == (1, "policy_stop")
    user = policy.prompts[1][1]
    assert ("  #0 ACTION5: LEVEL 1 COMPLETE after 9 actions — now on level 2/10: fresh board [level_complete]"
            in user.split("\n"))
    assert lines_starting(user, "Level: ") == ["Level: 2/10"]


def test_notices_reach_the_first_prompt_of_the_next_level_and_are_shown_once():
    _, policy = play("tw01", [plan(TW01_LEVEL_1 + [1] * 11), "<meta>x</meta><keep/>", ""])
    second, third = policy.prompts[1][1], policy.prompts[2][1]
    assert lines_starting(second, "Level: ") == ["Level: 2/10"]
    assert notice_lines(second) == [
        "- plan cap: your plan had 20 actions; only the first 16 ran — the last 4 were not executed"]
    assert "## Harness notices" not in third


def test_rule_list_notices_come_before_plan_notices_and_the_level_change_notice_comes_last():
    answer = ("<meta>x</meta><delete>9</delete><add>ACTION4 moves the cursor right</add><plan>"
              + ",".join(map(str, TW01_LEVEL_1 + [6])) + "</plan>")
    result, policy = play("tw01", [answer])
    assert (result.levels_completed, len(result.steps)) == (1, 9)
    assert notice_lines(policy.prompts[1][1]) == [
        "- ignored <delete>9</delete>: there is no rule [9] (the list above numbers rules 1–0)",
        "- ignored plan token '6' — not an available action id",
        "- Level change: 1 of your rules were carried over from the previous level (tagged [carried]; confidence "
        "reduced — verify them here)."]


@pytest.mark.parametrize("game_id", GAME_IDS)
def test_the_first_level_of_every_game_has_one_start(game_id):
    """The loop's direction votes assume one start on every first level (no start switch can happen there)."""
    game = make_game(game_id)
    game.perform_action(ActionInput(id=GameAction.RESET))
    assert len(game.truth().starts) == 1


# --------------------------------------------------------------------------------------------------- failures
def aborted_after_two_calls(error: Exception) -> EpisodeAborted:
    """The abort of a tw01 episode whose policy answers "" twice and then raises `error`."""
    with pytest.raises(EpisodeAborted) as excinfo:
        run_episode(make_game("tw01"), seed=0, limits=EpisodeLimits(10), policy=AnswersThenRaises(["", ""], error))
    return excinfo.value


def test_an_exception_of_the_policy_aborts_the_episode_with_everything_recorded_so_far():
    error = RuntimeError("the server gave up")
    aborted = aborted_after_two_calls(error)
    partial = aborted.partial
    assert aborted.__cause__ is error
    assert (len(partial.calls), len(partial.steps), partial.levels_completed, partial.stop_reason) == \
           (2, 2, 0, "aborted")
    assert str(aborted) == "episode aborted: game tw01 seed 0 after 2 reflection calls"


def test_an_answer_that_is_neither_text_nor_none_aborts_the_episode():
    with pytest.raises(EpisodeAborted) as excinfo:
        run_episode(make_game("tw01"), seed=0, limits=EpisodeLimits(10), policy=lambda system, user: b"x")
    assert isinstance(excinfo.value.__cause__, TypeError)
    assert str(excinfo.value.__cause__) == "policy returned bytes, expected str, Completion or None"
    partial = excinfo.value.partial
    assert (partial.calls, partial.steps, partial.levels_completed, partial.stop_reason) == ((), (), 0, "aborted")


def test_a_game_whose_progress_survives_the_entry_reset_is_rejected_before_any_call():
    game = make_game("tw01")
    # one move after the solved level, so that the entry RESET resets the level instead of restarting the game
    for action in (0, *TW01_LEVEL_1, 1):
        fd = game.perform_action(ActionInput(id=GameAction.from_id(action)))
    assert fd.levels_completed == 1
    policy = AnswersThenDeclines([])
    with pytest.raises(EpisodeAborted) as excinfo:
        run_episode(game, seed=0, limits=EpisodeLimits(10), policy=policy)
    assert isinstance(excinfo.value.__cause__, ValueError)
    assert policy.prompts == [] and excinfo.value.partial.calls == ()


class ProxyGame:
    """tw01 played through a stand-in for its engine interface (`game_id`, `truth`, `perform_action`)."""

    def __init__(self) -> None:
        self.game = make_game("tw01")
        self.game_id = self.game.game_id

    def truth(self):
        return self.game.truth()

    def perform_action(self, action_input: ActionInput):
        return self.game.perform_action(action_input)


class CounterWipingGame(ProxyGame):
    """Zeroes the engine's action counter before every RESET after the first, so the engine answers such a RESET
    by restarting the whole game instead of the level."""

    def __init__(self) -> None:
        super().__init__()
        self.resets = 0

    def perform_action(self, action_input: ActionInput):
        if action_input.id == GameAction.RESET:
            self.resets += 1
            if self.resets > 1:
                self.game.set_level(self.game.level_index)
        return super().perform_action(action_input)


class LevelSkippingGame(ProxyGame):
    """Answers every action as if twice the completed levels were completed."""

    def perform_action(self, action_input: ActionInput):
        fd = super().perform_action(action_input)
        return fd.model_copy(update={"levels_completed": 2 * fd.levels_completed})


def test_an_engine_answer_that_breaks_the_reset_rule_aborts_the_episode():
    with pytest.raises(EpisodeAborted) as excinfo:
        run_episode(CounterWipingGame(), seed=0, limits=EpisodeLimits(10),
                    policy=AnswersThenDeclines([plan([1, 0])]))
    assert isinstance(excinfo.value.__cause__, RuntimeError)
    assert str(excinfo.value.__cause__) == ("engine state out of step with the harness at level 0 step 1: "
                                            "full_reset=True, levels_completed 0 -> 0")
    partial = excinfo.value.partial
    assert len(partial.calls) == 1
    assert steps_of(partial) == [(0, 0, 1, "plan", False), (0, 1, 0, "plan", False)]
    assert partial.resets == ()


def test_an_engine_answer_that_completes_two_levels_at_once_aborts_the_episode():
    with pytest.raises(EpisodeAborted) as excinfo:
        run_episode(LevelSkippingGame(), seed=0, limits=EpisodeLimits(10),
                    policy=AnswersThenDeclines([plan(TW01_LEVEL_1)]))
    assert isinstance(excinfo.value.__cause__, RuntimeError)
    assert str(excinfo.value.__cause__) == ("engine state out of step with the harness at level 0 step 8: "
                                            "full_reset=False, levels_completed 0 -> 2")
    partial = excinfo.value.partial
    assert (len(partial.calls), len(partial.steps), partial.levels_completed) == (1, 9, 0)


# ------------------------------------------------------------------------------------------------ concurrency
def test_concurrent_episodes_do_not_interact_and_leave_the_global_random_state_alone():
    def episode(game_id: str, seed: int) -> EpisodeResult:
        return run_episode(make_game(game_id), seed=seed, limits=EpisodeLimits(10), policy=always_empty)

    jobs = (("tw01", 0), ("tw02", 104))
    sequential = [episode(*job) for job in jobs]
    state = random.getstate()
    together = threading.Barrier(len(jobs))

    def overlapping(job: tuple[str, int]) -> EpisodeResult:
        together.wait()                                 # no episode starts before every thread is running
        return episode(*job)

    with ThreadPoolExecutor(len(jobs)) as pool:
        concurrent = list(pool.map(overlapping, jobs))
    assert concurrent == sequential
    assert random.getstate() == state


# ------------------------------------------------------------------------------------------------ trace files
TRACE_CASES = {  # name -> the episode whose trace is written
    "won_game": lambda: play("tw09", TW09_ANSWERS, max_levels=20, cap=len(TW09_ANSWERS))[0],
    "one_executed_reset": lambda: play("tw01", [plan([0]), plan([1, 0]), plan([0])])[0],
    "aborted_episode": lambda: aborted_after_two_calls(RuntimeError("the server gave up")).partial,
}


def jsonl(records: list[dict]) -> bytes:
    return "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records).encode("utf-8")


@pytest.mark.parametrize("case, stop, resets", [
    ("won_game", "game_won", 0), ("one_executed_reset", "policy_stop", 1), ("aborted_episode", "aborted", 0)])
def test_trace_files_hold_every_record_and_a_summary(case, stop, resets, tmp_path):
    result = TRACE_CASES[case]()
    assert (result.stop_reason, len(result.resets)) == (stop, resets)
    directory = tmp_path / "traces" / result.game_id / f"seed{result.seed}"
    write_trace(result, directory)
    assert sorted(p.name for p in directory.iterdir()) == ["calls.jsonl", "steps.jsonl", "summary.json"]
    calls = [{"index": c.index, "level": c.level, "level_step": c.level_step, "system": c.system, "user": c.user,
              "response": c.response, "plan": None if c.plan is None else list(c.plan)} for c in result.calls]
    steps = [{"level": s.level, "step": s.step, "action": s.action, "actor": s.actor, "noop_reset": s.noop_reset}
             for s in result.steps]
    summary = {"game_id": result.game_id, "seed": result.seed, "max_levels": result.max_levels,
               "levels_completed": result.levels_completed, "reflection_calls": len(result.calls),
               "steps": len(result.steps), "random_fallback_steps": sum(s.actor == "random" for s in result.steps),
               "stop_reason": result.stop_reason,
               "resets": [{"level": r.level, "step": r.step} for r in result.resets], "observation_version": 2}
    written = {p.name: p.read_bytes() for p in directory.iterdir()}
    assert written["calls.jsonl"] == jsonl(calls)
    assert written["steps.jsonl"] == jsonl(steps)
    assert written["summary.json"] == (json.dumps(summary, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    read_calls = [json.loads(line) for line in written["calls.jsonl"].decode("utf-8").splitlines()]
    assert tuple(CallRecord(**{**c, "plan": None if c["plan"] is None else tuple(c["plan"])})
                 for c in read_calls) == result.calls
    assert tuple(StepRecord(**json.loads(line)) for line in written["steps.jsonl"].decode("utf-8").splitlines()) \
        == result.steps
    assert tuple(ResetRecord(**r) for r in json.loads(written["summary.json"])["resets"]) == result.resets
    write_trace(result, directory)
    assert {p.name: p.read_bytes() for p in directory.iterdir()} == written


def test_a_lone_surrogate_in_a_response_is_written_as_a_json_escape_and_reads_back_as_the_same_text(tmp_path):
    result = TRACE_CASES["won_game"]()
    first = dataclasses.replace(result.calls[0], response=result.calls[0].response + " \ud83d \u00e9")
    result = dataclasses.replace(result, calls=(first, *result.calls[1:]))
    write_trace(result, tmp_path)
    raw = (tmp_path / "calls.jsonl").read_bytes()
    assert b" \\ud83d \xc3\xa9" in raw
    assert json.loads(raw.decode("utf-8").splitlines()[0])["response"] == first.response
