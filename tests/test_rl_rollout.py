"""Training episodes against a scripted inference server (`witness_rl.rollout`) and the two trainer entry points
(`witness_rl.hooks`).

The server, the tokenizer and the trainer's arguments are small stand-ins written here: the tokenizer renders
"<s>" + system + "<u>" + user + "<a>" and tokenises it to its UTF-8 bytes, and every scripted answer carries
three sampled tokens. The training episodes of the golden episode fixtures (`agent/episodes/`) are replayed with
their recorded answers and finish reasons; every other expected value is written here by hand.
"""
from __future__ import annotations

import asyncio
import inspect
import json
import logging
import os
import random
import re
import subprocess
import sys
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import NamedTuple

import pytest

from golden import STORE, TRAINING_GAMES
from miles.rollout.base_types import GenerateFnInput
from miles.utils.types import Sample
from witness_agent import EpisodeResult, Task
from witness_agent.episode import CallRecord
from witness_rl import hooks, rollout as rollout_module
from witness_rl.advantages import EpisodeScore, broadcast_advantages
from witness_rl.rewards import call_rewards, episode_reward
from witness_rl.rollout import InferenceError, Rollout, prompt_ids, run_rollout

REPOSITORY = Path(__file__).resolve().parent.parent
ARGS = SimpleNamespace(sglang_router_ip="127.0.0.1", sglang_router_port=30000, rollout_max_response_len=8192,
                       rollout_max_context_len=None, use_rollout_routing_replay=False,
                       use_rollout_indexer_replay=False, sglang_speculative_algorithm=None, n_samples_per_prompt=8)
ROUTER_URL = "http://127.0.0.1:30000/generate"  # where requests go with ARGS
SAMPLING_PARAMS = {"temperature": 1.0, "top_p": 1.0, "top_k": -1, "max_new_tokens": 8192, "stop": None,
                   "stop_token_ids": None, "skip_special_tokens": False, "no_stop_trim": True,
                   "spaces_between_special_tokens": False}
SAMPLED_IDS = [100, 101, 102]                   # the token ids of every scripted answer
SAMPLED_LOGPROB = -0.5
ROW = {"game_id": "tw01", "max_levels": 10, "max_reflection_calls": 6}
SEED = 1176
PLANLESS = "no plan"                            # an answer without a plan: one random move (never a CONFIRM) follows
METADATA_KEYS = ["game_id", "call_index", "episode_reward", "episode_abort", "finish_reason", "levels_completed",
                 "dropped_calls"]
SIX_CALLS_REWARD = episode_reward(call_rewards([0] * 6, 0, 10))


class FakeTokenizer:
    """Chat template "<s>" + system + "<u>" + user + "<a>", token ids = UTF-8 bytes. Records the messages and
    keyword arguments of every template call and the thread of every use; `fail_at` makes that template call
    (0-based) raise ValueError."""

    def __init__(self, fail_at: int | None = None) -> None:
        self.fail_at = fail_at
        self.templates: list[tuple[list[dict], dict]] = []
        self.threads: set[int] = set()

    def apply_chat_template(self, messages: list[dict], **kwargs) -> str:
        self.threads.add(threading.get_ident())
        self.templates.append((messages, kwargs))
        if len(self.templates) - 1 == self.fail_at:
            raise ValueError("the chat template cannot render this prompt")
        system, user = messages
        return "<s>" + system["content"] + "<u>" + user["content"] + "<a>"

    def __call__(self, text: str, *, add_special_tokens: bool) -> dict:
        self.threads.add(threading.get_ident())
        if add_special_tokens is not False:
            raise ValueError("special tokens must not be added")
        return {"input_ids": list(text.encode("utf-8"))}


def rendered_ids(system: str, user: str) -> list[int]:
    return list(("<s>" + system + "<u>" + user + "<a>").encode("utf-8"))


def reply(text: str, finish: str = "stop") -> dict:
    """A server response with the three sampled tokens and the given finish reason."""
    return {"text": text, "meta_info": {"output_token_logprobs": [(SAMPLED_LOGPROB, t, None) for t in SAMPLED_IDS],
                                        "finish_reason": {"type": finish}}}


class ScriptedServer:
    """Answers the i-th request with the i-th scripted item: a (text, finish reason) pair becomes `reply(...)`, a
    dict is returned as it is, an exception is raised. Records every request and the thread it ran on; with
    `pause`, every request first yields to the event loop."""

    def __init__(self, script, *, pause: bool = False) -> None:
        self.script, self.pause = list(script), pause
        self.requests: list = []
        self.threads: set[int] = set()

    async def _answer(self, record):
        self.threads.add(threading.get_ident())
        self.requests.append(record)
        if self.pause:
            await asyncio.sleep(0)
        item = self.script[len(self.requests) - 1]
        if isinstance(item, BaseException):
            raise item
        return item if isinstance(item, dict) else reply(*item)

    async def request(self, ids: list[int]) -> dict:
        """The rollout's request function: prompt ids -> response."""
        return await self._answer(ids)

    async def post(self, url: str, payload: dict) -> dict:
        """Stands in for miles' HTTP post in `witness_rl.hooks`."""
        return await self._answer((url, payload))


def play(task: Task, seed: int, server: ScriptedServer, tokenizer: FakeTokenizer | None = None,
         trace_dir: Path | None = None) -> Rollout:
    return asyncio.run(run_rollout(task, seed, tokenizer=tokenizer or FakeTokenizer(), request=server.request,
                                   trace_dir=trace_dir))


def hook_input(metadata: dict | None = None, args: SimpleNamespace = ARGS,
               tokenizer: FakeTokenizer | None = None) -> GenerateFnInput:
    return GenerateFnInput(state=SimpleNamespace(args=args, tokenizer=tokenizer or FakeTokenizer()),
                           sample=Sample(index=SEED, metadata=dict(ROW if metadata is None else metadata)),
                           sampling_params=dict(SAMPLING_PARAMS), evaluation=False)


def run_hook(monkeypatch, script, **kwargs) -> tuple[list[Sample], GenerateFnInput, ScriptedServer]:
    """The per-episode trainer hook on the six-call tw01 episode (unless `metadata` says otherwise), the server
    answering `script`."""
    server = ScriptedServer(script)
    monkeypatch.setattr(hooks, "post", server.post)
    inp = hook_input(**kwargs)
    return asyncio.run(hooks.generate(inp)).samples, inp, server


def six_answers(*finishes: str) -> list[tuple[str, str]]:
    return [(PLANLESS, finish) for finish in finishes or ("stop",) * 6]


# ------------------------------------------------------------------------------------------- prompt rendering
def test_prompt_ids_render_system_and_user_with_thinking_disabled_as_a_template_argument():
    tokenizer = FakeTokenizer()
    ids = prompt_ids(tokenizer, "S", "U")
    assert tokenizer.templates == [([{"role": "system", "content": "S"}, {"role": "user", "content": "U"}],
                                    {"tokenize": False, "add_generation_prompt": True, "enable_thinking": False})]
    assert ids == list(b"<s>S<u>U<a>")


# ------------------------------------------------------------------------------------------------ run_rollout
REPLAYABLE = [name for name in STORE.episode_names()
              if STORE.episode(name).wiring == "rl" and STORE.episode(name).responses]


@pytest.mark.parametrize("name", REPLAYABLE)
def test_every_training_episode_fixture_plays_through_the_rollout_with_its_rewards_and_kept_calls(name):
    ep = STORE.episode(name)
    server = ScriptedServer(zip(ep.responses, ep.finish_reasons, strict=True))
    # A prefix fixture ends by declining the call after its last recorded answer.
    task = Task(ep.game_id, ep.max_levels, min(len(ep.responses), ep.max_reflection_calls))
    rollout = play(task, ep.seed, server)
    assert rollout.abort is None
    assert [(c.system, c.user) for c in rollout.calls] == [(e.system, e.user) for e in ep.calls]
    assert len(server.requests) == len(ep.calls)
    assert [c.level for c in rollout.calls] == [e.level for e in ep.calls]
    assert [c.reward.total for c in rollout.calls] == [r.total for r in ep.rewards]
    assert [c.kept for c in rollout.calls] == list(ep.kept)
    assert (rollout.reward, rollout.levels_completed, rollout.stop_reason) == \
           (ep.episode_reward, ep.levels_completed, ep.stop_reason)
    samples = asyncio.run(hooks._episode_samples(ARGS, rollout, rollout_id=ep.seed))
    assert {s.metadata["levels_completed"] for s in samples} == {ep.levels_completed}
    if ep.representative is None:
        assert [s.index for s in samples] == [i for i, kept in enumerate(ep.kept) if kept]
        assert all(s.loss_mask == [1] * len(SAMPLED_IDS) for s in samples)
    else:
        (sample,) = samples
        assert sample.index == ep.representative
        assert sample.loss_mask == [0] * sample.response_length == [0] * len(SAMPLED_IDS)


def test_the_call_budget_declines_the_next_call_without_requesting_it():
    ep = STORE.episode("scripted_rl_cap")
    assert (ep.game_id, ep.seed, ep.max_reflection_calls, len(ep.responses)) == ("tw02", 7, 4, 6)
    server = ScriptedServer(zip(ep.responses, ep.finish_reasons, strict=True))
    rollout = play(Task("tw02", 10, 4), 7, server)
    assert len(server.requests) == len(rollout.calls) == 4
    assert (rollout.stop_reason, rollout.abort) == ("policy_stop", None)
    assert [c.reward.total for c in rollout.calls] == [r.total for r in ep.rewards]
    assert rollout.reward == ep.episode_reward


@pytest.mark.parametrize("name", ["rl_tw10_s1120_n3", "rl_tw01_s1064_n4"])
def test_a_request_that_fails_every_attempt_ends_the_episode_as_an_inference_error_with_partial_rewards(
        name, caplog):
    ep = STORE.episode(name)
    error = InferenceError("the server did not answer")
    server = ScriptedServer([*zip(ep.responses[:2], ep.finish_reasons[:2], strict=True), error])
    with caplog.at_level(logging.WARNING, logger="witness_rl"):
        rollout = play(Task(ep.game_id, ep.max_levels, ep.max_reflection_calls), ep.seed, server)
    assert (rollout.abort, rollout.stop_reason) == ("inference_error", "aborted")
    assert [(c.system, c.user) for c in rollout.calls] == [(e.system, e.user) for e in ep.calls[:2]]
    # The last answered call earns the levels completed up to the failed request, as the full episode's
    # reward of that call does.
    assert rollout.levels_completed == ep.calls[2].level
    assert [c.reward.total for c in rollout.calls] == [r.total for r in ep.rewards[:2]]
    assert rollout.reward == episode_reward([c.reward for c in rollout.calls])
    (record,) = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert f"game={ep.game_id} seed={ep.seed} calls=2 abort=inference_error" in record.getMessage()
    assert record.exc_info[1] is error


def test_a_template_error_ends_the_episode_as_an_agent_error_with_the_calls_answered_before():
    server = ScriptedServer(six_answers())
    rollout = play(Task(**ROW), SEED, server, tokenizer=FakeTokenizer(fail_at=2))
    assert (rollout.abort, rollout.stop_reason) == ("agent_error", "aborted")
    assert len(rollout.calls) == len(server.requests) == 2
    assert rollout.reward == episode_reward(call_rewards([0, 0], 0, 10))


def test_a_game_that_cannot_be_made_gives_an_aborted_rollout_without_calls(monkeypatch, caplog):
    monkeypatch.setenv("ONLY_RESET_LEVELS", "true")
    server = ScriptedServer([])
    with caplog.at_level(logging.WARNING, logger="witness_rl"):
        rollout = play(Task("tw01", 10, 300), 0, server)
    assert (rollout.abort, rollout.stop_reason, rollout.calls) == ("agent_error", "aborted", ())
    assert (rollout.reward, rollout.levels_completed, server.requests) == (0.0, 0, [])
    (record,) = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert isinstance(record.exc_info[1], RuntimeError)


def test_prompts_are_rendered_and_sent_on_the_event_loop_while_the_episode_runs_in_its_own_thread(monkeypatch):
    real_run_episode = rollout_module.run_episode
    episode_threads: set[int] = set()

    def run_episode(game, *, seed, limits, policy):
        def watched(system, user):
            episode_threads.add(threading.get_ident())
            return policy(system, user)
        episode_threads.add(threading.get_ident())
        return real_run_episode(game, seed=seed, limits=limits, policy=watched)

    monkeypatch.setattr(rollout_module, "run_episode", run_episode)
    tokenizer, server = FakeTokenizer(), ScriptedServer(six_answers())

    async def main() -> tuple[int, Rollout]:
        return threading.get_ident(), await run_rollout(Task(**ROW), SEED, tokenizer=tokenizer,
                                                        request=server.request)

    loop_thread, rollout = asyncio.run(main())
    assert len(rollout.calls) == 6
    assert tokenizer.threads == server.threads == {loop_thread}
    assert len(episode_threads) == 1 and loop_thread not in episode_threads


def test_a_cancelled_rollout_declines_the_next_call_and_its_episode_thread_ends_without_an_error(monkeypatch,
                                                                                                  caplog):
    real_run_episode = rollout_module.run_episode
    episode_threads: list[threading.Thread] = []

    def run_episode(game, *, seed, limits, policy):
        episode_threads.append(threading.current_thread())
        return real_run_episode(game, seed=seed, limits=limits, policy=policy)

    monkeypatch.setattr(rollout_module, "run_episode", run_episode)
    requests: list[list[int]] = []

    async def main() -> None:
        second_request, release = asyncio.Event(), asyncio.Event()

        async def request(ids: list[int]) -> dict:
            requests.append(ids)
            if len(requests) == 2:
                second_request.set()
                await release.wait()
            return reply(PLANLESS)

        rollout = asyncio.create_task(run_rollout(Task("tw01", 10, 40), SEED, tokenizer=FakeTokenizer(),
                                                  request=request))
        await second_request.wait()
        rollout.cancel()
        with pytest.raises(asyncio.CancelledError):
            await rollout
        release.set()                           # the answer of the call in flight reaches the episode
        (thread,) = episode_threads
        await asyncio.to_thread(thread.join, 10)
        assert not thread.is_alive()

    with caplog.at_level(logging.ERROR):
        asyncio.run(main())
    assert len(requests) == 2
    assert [r.getMessage() for r in caplog.records] == []


def test_concurrent_rollouts_on_one_event_loop_equal_the_same_rollouts_run_alone():
    solutions = STORE.solutions()
    cases = [(game_id, seed) for game_id in ("tw01", "tw05", "ft01", "ft10") for seed in range(4)]

    def script(game_id: str, seed: int) -> list[tuple[str, str]]:
        answers = ["<meta>x</meta><keep/><plan>" + ",".join(map(str, solutions[game_id][0])) + "</plan>"]
        answers += [PLANLESS] * 4
        return [(text, "length" if i == seed else "stop") for i, text in enumerate(answers)]

    async def together() -> list[Rollout]:
        return list(await asyncio.gather(*(
            run_rollout(Task(game_id, 10, 5), seed, tokenizer=FakeTokenizer(),
                        request=ScriptedServer(script(game_id, seed), pause=True).request)
            for game_id, seed in cases)))

    alone = [play(Task(game_id, 10, 5), seed, ScriptedServer(script(game_id, seed))) for game_id, seed in cases]
    assert asyncio.run(together()) == alone
    assert {r.levels_completed for r in alone} == {1}


def test_the_trace_of_an_episode_is_written_under_its_game_and_seed(tmp_path):
    rollout = play(Task(**ROW), SEED, ScriptedServer(six_answers()), trace_dir=tmp_path)
    directory = tmp_path / "tw01" / f"seed{SEED}"
    assert sorted(p.name for p in directory.iterdir()) == ["calls.jsonl", "steps.jsonl", "summary.json"]
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    assert (summary["game_id"], summary["seed"], summary["reflection_calls"]) == ("tw01", SEED, len(rollout.calls))
    assert summary["stop_reason"] == rollout.stop_reason == "policy_stop"


def test_the_trace_of_an_aborted_episode_holds_the_calls_answered_before_the_failure(tmp_path):
    rollout = play(Task(**ROW), SEED, ScriptedServer(six_answers()), tokenizer=FakeTokenizer(fail_at=2),
                   trace_dir=tmp_path)
    directory = tmp_path / "tw01" / f"seed{SEED}"
    assert rollout.abort == "agent_error"
    assert len((directory / "calls.jsonl").read_text(encoding="utf-8").splitlines()) == len(rollout.calls) == 2
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    assert summary["stop_reason"] == rollout.stop_reason == "aborted"


def test_a_trace_that_cannot_be_written_is_logged_and_changes_nothing(tmp_path, caplog):
    (tmp_path / "f").write_text("a file where a directory is needed", encoding="utf-8")
    with caplog.at_level(logging.ERROR, logger="witness_rl"):
        rollout = play(Task(**ROW), SEED, ScriptedServer(six_answers()), trace_dir=tmp_path / "f" / "sub")
    assert rollout.abort is None
    assert rollout == play(Task(**ROW), SEED, ScriptedServer(six_answers()))
    (record,) = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert str(tmp_path / "f" / "sub" / "tw01" / f"seed{SEED}") in record.getMessage()


def test_an_error_of_the_trace_writer_other_than_an_os_error_ends_the_episode_as_an_agent_error(monkeypatch,
                                                                                                tmp_path):
    def broken_writer(result, directory):
        raise ValueError("cannot serialize the trace")

    monkeypatch.setattr(rollout_module, "write_trace", broken_writer)
    rollout = play(Task(**ROW), SEED, ScriptedServer(six_answers()), trace_dir=tmp_path)
    assert (rollout.abort, rollout.stop_reason, len(rollout.calls)) == ("agent_error", "aborted", 6)
    assert rollout.reward == SIX_CALLS_REWARD


# ------------------------------------------------------------------------------------------- per-episode hook
def test_the_episode_hook_returns_one_sample_per_call_sharing_the_episode_rollout_id(monkeypatch):
    samples, inp, server = run_hook(monkeypatch, six_answers())
    assert [s.index for s in samples] == [s.metadata["call_index"] for s in samples] == [0, 1, 2, 3, 4, 5]
    messages = [m for m, _ in inp.state.tokenizer.templates]
    for sample, (_, payload), (system, user) in zip(samples, server.requests, messages, strict=True):
        assert payload["input_ids"] == rendered_ids(system["content"], user["content"])
        assert sample.tokens == payload["input_ids"] + SAMPLED_IDS
        assert (sample.rollout_id, sample.prompt, sample.response, sample.response_length) == (SEED, "", PLANLESS, 3)
        assert sample.loss_mask == [1, 1, 1]
        assert sample.rollout_log_probs == [SAMPLED_LOGPROB] * 3
        assert sample.status == Sample.Status.COMPLETED
        assert sample.reward == SIX_CALLS_REWARD
        assert list(sample.metadata) == METADATA_KEYS
        assert sample.metadata == {"game_id": "tw01", "call_index": sample.index, "episode_reward": SIX_CALLS_REWARD,
                                   "episode_abort": "", "finish_reason": "stop", "levels_completed": 0,
                                   "dropped_calls": 0}


def test_calls_not_ended_by_a_stop_token_are_dropped_but_keep_their_penalty_and_call_numbers(monkeypatch):
    samples, _, server = run_hook(monkeypatch, six_answers("stop", "length", "stop", "other", "abort", "stop"))
    assert len(server.requests) == 6
    assert [s.index for s in samples] == [s.metadata["call_index"] for s in samples] == [0, 2, 5]
    assert {(s.metadata["dropped_calls"], s.metadata["finish_reason"], s.status) for s in samples} == \
           {(3, "stop", Sample.Status.COMPLETED)}
    assert {s.reward for s in samples} == {s.metadata["episode_reward"] for s in samples} == {SIX_CALLS_REWARD}


def test_an_episode_whose_calls_were_all_cut_off_returns_call_zero_with_a_zero_loss_mask(monkeypatch):
    samples, _, server = run_hook(monkeypatch, six_answers(*["length"] * 6))
    (sample,) = samples
    assert (sample.index, sample.rollout_id, sample.response_length) == (0, SEED, 3)
    assert sample.tokens == server.requests[0][1]["input_ids"] + SAMPLED_IDS
    assert sample.loss_mask == [0, 0, 0]
    assert sample.status == Sample.Status.TRUNCATED
    assert (sample.metadata["dropped_calls"], sample.metadata["finish_reason"]) == (6, "length")
    assert sample.reward == SIX_CALLS_REWARD


def marker_of_failed_first_call(monkeypatch) -> tuple[list[Sample], GenerateFnInput, ScriptedServer]:
    """The per-episode hook on the six-call episode with a server that fails every attempt."""
    return run_hook(monkeypatch, [RuntimeError("connection refused")] * 3)


def test_an_episode_without_an_answered_call_returns_the_input_sample_as_a_marker(monkeypatch):
    samples, inp, server = marker_of_failed_first_call(monkeypatch)
    assert len(server.requests) == hooks.REQUEST_ATTEMPTS == 3
    (sample,) = samples
    assert sample is inp.sample
    assert (sample.rollout_id, sample.reward, sample.tokens) == (SEED, 0.0, [])
    assert sample.metadata == {**ROW, "episode_reward": 0.0, "episode_abort": "inference_error",
                               "levels_completed": 0, "dropped_calls": 0}


@pytest.mark.parametrize("metadata, script", [
    ({"game_id": "tw01", "max_reflection_calls": 6}, []),
    (ROW, [{"text": PLANLESS}] * 6),
    (ROW, [reply(PLANLESS), reply(PLANLESS), {"meta_info": reply(PLANLESS)["meta_info"]}]),
], ids=["row_without_max_levels", "response_without_meta_info", "third_response_without_text"])
def test_the_episode_hook_never_raises(monkeypatch, metadata, script):
    samples, inp, server = run_hook(monkeypatch, script, metadata=metadata)
    (sample,) = samples
    assert sample is inp.sample
    assert (sample.rollout_id, sample.reward, sample.tokens) == (SEED, 0.0, [])
    assert sample.metadata["episode_abort"] == "agent_error"
    assert len(server.requests) == len(script)


def test_call_records_that_do_not_match_the_answered_calls_make_the_episode_a_marker(monkeypatch):
    def run_episode(game, *, seed, limits, policy):
        response = policy("S", "U")
        record = CallRecord(index=0, level=0, level_step=0, system="S", user="another user message",
                            response=response, plan=None)
        return EpisodeResult(game_id=game.game_id, seed=seed, max_levels=limits.max_levels, calls=(record,),
                             steps=(), resets=(), levels_completed=0, stop_reason="policy_stop")

    monkeypatch.setattr(rollout_module, "run_episode", run_episode)
    with pytest.raises(RuntimeError, match=f"game tw01 seed {SEED}: the 1 call records of the harness do not match"):
        play(Task(**ROW), SEED, ScriptedServer(six_answers()))
    samples, inp, server = run_hook(monkeypatch, six_answers())
    assert samples == [inp.sample] and len(server.requests) == 1
    assert (samples[0].tokens, samples[0].metadata["episode_abort"]) == ([], "agent_error")


def test_requests_go_to_the_router_with_the_miles_request_body(monkeypatch):
    _, inp, server = run_hook(monkeypatch, six_answers())
    messages = [m for m, _ in inp.state.tokenizer.templates]
    assert len(server.requests) == len(messages) == 6
    for (url, payload), (system, user) in zip(server.requests, messages, strict=True):
        assert url == ROUTER_URL
        assert payload == {
            "input_ids": rendered_ids(system["content"], user["content"]),
            "sampling_params": {"temperature": 1.0, "top_p": 1.0, "top_k": -1, "stop": None, "stop_token_ids": None,
                                "skip_special_tokens": False, "no_stop_trim": True,
                                "spaces_between_special_tokens": False, "max_new_tokens": 8192},
            "return_logprob": True, "return_routed_experts": False, "return_indexer_topk": False}
    assert inp.sampling_params == SAMPLING_PARAMS


def request_once(monkeypatch, script) -> tuple[object, ScriptedServer]:
    """One request through the episode request function of `hooks`; returns its response or its exception."""
    server = ScriptedServer(script)
    monkeypatch.setattr(hooks, "post", server.post)
    try:
        return asyncio.run(hooks._sglang_request(ARGS, SAMPLING_PARAMS)([1, 2, 3])), server
    except InferenceError as exc:
        return exc, server


@pytest.mark.parametrize("errors, succeeds", [
    ([TimeoutError(), TimeoutError(), TimeoutError()], False),
    ([RuntimeError("503 Service Unavailable")], True),
    ([RuntimeError("400 Bad Request")] * 3, False),
], ids=["timeouts", "one_failure", "client_errors"])
def test_requests_are_retried_up_to_three_attempts_whatever_the_error(monkeypatch, caplog, errors, succeeds):
    with caplog.at_level(logging.WARNING, logger="witness_rl"):
        outcome, server = request_once(monkeypatch, [*errors, (PLANLESS, "stop")])
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    if succeeds:
        assert outcome == reply(PLANLESS)
        assert (len(server.requests), len(warnings)) == (2, 1)
    else:
        assert isinstance(outcome, InferenceError)
        assert outcome.__cause__ is errors[-1]
        assert str(outcome).startswith(f"{ROUTER_URL}: ")
        assert (len(server.requests), len(warnings)) == (3, 2)
    assert warnings[0].getMessage() == f"request attempt 1/3 failed: {type(errors[0]).__name__}: {errors[0]}"


def test_a_request_attempt_is_bounded_by_the_request_timeout(monkeypatch):
    attempts = []

    async def slow_server(url, payload):
        attempts.append(url)
        await asyncio.sleep(1.0)
        return reply(PLANLESS)

    monkeypatch.setattr(hooks, "post", slow_server)
    monkeypatch.setattr(hooks, "REQUEST_TIMEOUT_S", 0.01)
    with pytest.raises(InferenceError) as raised:
        asyncio.run(hooks._sglang_request(ARGS, SAMPLING_PARAMS)([1, 2, 3]))
    assert isinstance(raised.value.__cause__, TimeoutError)
    assert len(attempts) == 3


def test_the_episode_hook_writes_traces_only_when_the_trainer_arguments_name_a_trace_directory(monkeypatch,
                                                                                               tmp_path):
    without, _, _ = run_hook(monkeypatch, six_answers())
    assert list(tmp_path.iterdir()) == []
    args = SimpleNamespace(**vars(ARGS), witness_trace_dir=str(tmp_path))
    with_traces, _, _ = run_hook(monkeypatch, six_answers(), args=args)
    directory = tmp_path / "tw01" / f"seed{SEED}"
    assert sorted(p.name for p in directory.iterdir()) == ["calls.jsonl", "steps.jsonl", "summary.json"]
    assert len((directory / "calls.jsonl").read_text(encoding="utf-8").splitlines()) == 6
    assert with_traces == without
    assert [s.index for s in without] == [0, 1, 2, 3, 4, 5]


# ---------------------------------------------------------------------------------------- reward_post_process
class Episode(NamedTuple):
    """One episode of a training step as `reward_post_process` sees it."""

    game_id: str
    rollout_id: int
    reward: float
    abort: str                                  # "" for a valid episode
    samples: int                                # number of training samples


def call_samples(episode: Episode) -> list[Sample]:
    """The training samples of one episode, shaped as the per-episode hook returns them."""
    return [Sample(index=i, rollout_id=episode.rollout_id, tokens=[7, 8, 9, 10], response_length=2,
                   reward=episode.reward,
                   metadata={"game_id": episode.game_id, "call_index": i, "episode_reward": episode.reward,
                             "episode_abort": episode.abort, "finish_reason": "stop", "levels_completed": 0,
                             "dropped_calls": 0})
            for i in range(episode.samples)]


def step_episodes(seed: int = 0) -> list[Episode]:
    """The 22 x 8 episodes of one step: games in a shuffled order, 1-4 samples per episode, one aborted episode in
    each of five games."""
    rng = random.Random(seed)
    games = list(TRAINING_GAMES)
    rng.shuffle(games)
    aborted = {(game_id, rng.randrange(8)) for game_id in games[:5]}
    return [Episode(game_id, 2000 + 8 * g + k, round(rng.uniform(-1.5, 10.5), 3),
                    "agent_error" if (game_id, k) in aborted else "", rng.randint(1, 4))
            for g, game_id in enumerate(games) for k in range(8)]


def flat(episodes: list[Episode]) -> list[Sample]:
    return [s for episode in episodes for s in call_samples(episode)]


def advantages_by_episode(episodes: list[Episode]) -> dict[int, float]:
    scores = [EpisodeScore(e.game_id, e.reward, valid=e.abort == "") for e in episodes]
    return {e.rollout_id: a for e, a in zip(episodes, broadcast_advantages(scores), strict=True)}


def test_reward_post_process_broadcasts_each_episode_advantage_to_all_its_samples():
    episodes = step_episodes()
    samples = flat(episodes)
    raw, advantages = hooks.reward_post_process(ARGS, samples)
    expected = advantages_by_episode(episodes)
    assert raw == [s.reward for s in samples]
    assert advantages == [expected[s.rollout_id] for s in samples]
    aborted = {e.rollout_id for e in episodes if e.abort}
    by_sample = list(zip(samples, advantages, strict=True))
    assert len(aborted) == 5 and all(a == 0.0 for s, a in by_sample if s.rollout_id in aborted)
    assert len({a for s, a in by_sample if s.rollout_id not in aborted}) > 100


def test_reward_post_process_rejects_a_missing_rollout_id_and_a_game_without_eight_episodes():
    episodes = step_episodes()
    short = [e for e in episodes if e.game_id != "tw01"] + [e for e in episodes if e.game_id == "tw01"][:7]
    with pytest.raises(ValueError, match=re.escape("episodes per game != 8: {'tw01': 7}")):
        hooks.reward_post_process(ARGS, flat(short))
    samples = flat(episodes)
    samples[5].rollout_id = None
    with pytest.raises(ValueError, match="sample 5 has no rollout_id"):
        hooks.reward_post_process(ARGS, samples)


def test_a_step_with_an_episode_without_an_answered_call_is_rejected_naming_it(monkeypatch):
    (marker,), _, _ = marker_of_failed_first_call(monkeypatch)
    episodes = step_episodes()
    replaced = next(i for i, e in enumerate(episodes) if e.game_id == "tw01")
    samples = [s for i, e in enumerate(episodes) for s in ([marker] if i == replaced else call_samples(e))]
    assert sum(s.rollout_id == SEED for s in samples) == 1
    with pytest.raises(RuntimeError, match=re.escape("tw01 seed 1176 (inference_error)")):
        hooks.reward_post_process(ARGS, samples)
    # The episode count per game is checked first.
    other = next(e.rollout_id for i, e in enumerate(episodes) if e.game_id == "tw01" and i != replaced)
    with pytest.raises(ValueError, match=re.escape("episodes per game != 8: {'tw01': 7}")):
        hooks.reward_post_process(ARGS, [s for s in samples if s.rollout_id != other])


def test_an_episode_that_fails_after_answered_calls_returns_them_with_the_abort_type_and_no_advantage(monkeypatch):
    samples, _, server = run_hook(monkeypatch, [*six_answers()[:2], *[RuntimeError("connection refused")] * 3])
    assert [s.index for s in samples] == [0, 1] and len(server.requests) == 2 + hooks.REQUEST_ATTEMPTS
    assert {s.metadata["episode_abort"] for s in samples} == {"inference_error"}
    assert {s.reward for s in samples} == {episode_reward(call_rewards([0, 0], 0, 10))}
    # In a step, the episode gets advantage 0.0 and the other episodes are scored as if it were absent.
    episodes = step_episodes()
    replaced = next(i for i, e in enumerate(episodes) if e.game_id == "tw01")
    step = [s for i, e in enumerate(episodes) for s in (samples if i == replaced else call_samples(e))]
    _, advantages = hooks.reward_post_process(ARGS, step)
    others = advantages_by_episode(episodes[:replaced] + episodes[replaced + 1:])
    assert advantages == [0.0 if s.rollout_id == SEED else others[s.rollout_id] for s in step]


def test_a_game_or_a_step_without_valid_episodes_gets_zero_advantages_and_a_warning_per_game(caplog):
    episodes = step_episodes()
    lost = "ft05"
    without_game = [e._replace(abort="agent_error") if e.game_id == lost else e for e in episodes]
    samples = flat(without_game)
    with caplog.at_level(logging.WARNING, logger="witness_rl"):
        raw, advantages = hooks.reward_post_process(ARGS, samples)
    others = advantages_by_episode([e for e in without_game if e.game_id != lost])
    assert raw == [s.reward for s in samples]
    assert advantages == [0.0 if s.metadata["game_id"] == lost else others[s.rollout_id] for s in samples]
    (record,) = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert f"game={lost} " in record.getMessage()

    caplog.clear()
    all_aborted = flat([e._replace(abort="inference_error") for e in episodes])
    with caplog.at_level(logging.WARNING, logger="witness_rl"):
        raw, advantages = hooks.reward_post_process(ARGS, all_aborted)
    assert raw == [s.reward for s in all_aborted]
    assert advantages == [0.0] * len(all_aborted)
    warned = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert [re.search(r"game=(\S+)", m).group(1) for m in warned] == sorted(TRAINING_GAMES)


def test_the_entry_points_have_the_signatures_miles_loads():
    assert list(inspect.signature(hooks.generate).parameters) == ["input"]
    assert inspect.iscoroutinefunction(hooks.generate)
    assert list(inspect.signature(hooks.reward_post_process).parameters) == ["args", "samples"]
    assert not inspect.iscoroutinefunction(hooks.reward_post_process)


# -------------------------------------------------------------------------------------------- import boundary
RL_WITHOUT_MILES = """
import sys
sys.modules["httpx"] = None                     # any import of the HTTP client fails
sys.modules["miles"] = None                     # any import of the trainer fails
import witness_env, witness_agent, witness_rl.rewards, witness_rl.advantages, witness_rl.rollout
try:
    import witness_rl.hooks
except ImportError:
    print("hooks needs miles")
"""


def test_only_the_hooks_module_needs_miles():
    completed = subprocess.run([sys.executable, "-c", RL_WITHOUT_MILES], capture_output=True, text=True,
                               env={**os.environ, "PYTHONPATH": str(REPOSITORY)}, check=False)
    assert (completed.returncode, completed.stdout, completed.stderr) == (0, "hooks needs miles\n", "")
