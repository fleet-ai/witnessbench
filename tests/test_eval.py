"""The validation evaluator: the HTTP policy, the server check, task rows, the levels-beaten report, the evaluator
end to end and the serving scripts (`witness_agent.http_policy`, `witness_agent.evaluate`, `scripts/`).

Requests go to an in-process OpenAI-compatible stand-in server on 127.0.0.1, and a fake clock drives the retry
pauses. The task rows are the shipped `data/val.jsonl`; the evaluation episodes of `agent/episodes/` (training
games) are replayed through the command line with the server answering their stored responses; every other
expected value is written here by hand.
"""
from __future__ import annotations

import json
import logging
import os
import re
import shlex
import shutil
import socket
import subprocess
import sys
import threading
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import httpx
import pytest
from arcengine import ActionInput, GameAction

from golden import STORE, VALIDATION_GAMES, Episode, fallback_moves, first_diff
from witness_agent import EpisodeAborted, evaluate, http_policy
from witness_agent.episode import CallRecord, EpisodeResult, StepRecord, Task
from witness_agent.evaluate import (DEFAULT_SEEDS, DEFAULT_TASKS, EpisodeOutcome, GameSummary, format_report,
                                    load_tasks, main, run_validation, summarize)
from witness_agent.http_policy import (PROBE_PROMPT, HttpPolicy, PolicyError, PreflightError, chat_request_body,
                                       encode_body, probe_thinking_off)
from witness_agent.trace import write_utf8
from witness_env import make_game

REPOSITORY = Path(__file__).resolve().parent.parent

# --------------------------------------------------------------------- the stand-in server and the fake clock
#
# The server answers POST requests only (any other method gets HTTP 501 and is not recorded). An action is
# (status, JSON value or raw bytes), ("redirect", status, location) (an empty reply with that Location header),
# "drop" (close the connection without a reply) or ("sleep", seconds, action). The probe request (its only message
# is PROBE_PROMPT) is answered with `server.probe_answer` (an action, or a function of the request path that gives
# one), every other POST with `server.answer(path, parsed body)`. An answer that raises (e.g. a failed assertion
# about the prompt) is answered with HTTP 418, which the client never retries, and the `server` fixture then fails
# the test with it.


def reply(content: object = "OK", tokens: int = 2, **message: object) -> dict:
    """A chat-completion reply: one choice whose message has `content` and the `message` fields, and `tokens`
    completion tokens."""
    return {"choices": [{"message": {"role": "assistant", "content": content, **message}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 14, "completion_tokens": tokens, "total_tokens": 14 + tokens}}


PROBE_OK = reply()


@dataclass(frozen=True)
class Received:
    """One POST request as the server received it; header names in lower case."""

    path: str
    headers: dict[str, str]
    body: bytes


class FakeServer:
    """An OpenAI-compatible stand-in server on 127.0.0.1 at `url` that records every request (`received`) and
    every exception its answers raised (`errors`); clients are given `base_url` (`url` + "/v1")."""

    def __init__(self) -> None:
        self.received: list[Received] = []
        self.errors: list[Exception] = []
        self.answer: Callable[[str, object], object] = lambda path, body: (418, {"error": "no answer set"})
        self.probe_answer: object = (200, PROBE_OK)
        self._httpd = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        self._httpd.fake = self
        self.url = f"http://127.0.0.1:{self._httpd.server_address[1]}"
        self.base_url = self.url + "/v1"
        threading.Thread(target=self._httpd.serve_forever, args=(0.01,), daemon=True).start()

    def action_for(self, path: str, body: object) -> object:
        if isinstance(body, dict) and body.get("messages") == [{"role": "user", "content": PROBE_PROMPT}]:
            return self.probe_answer(path) if callable(self.probe_answer) else self.probe_answer
        return self.answer(path, body)

    def close(self) -> None:
        self._httpd.shutdown()
        self._httpd.server_close()


class _Handler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:
        fake = self.server.fake
        body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        fake.received.append(Received(self.path, {name.lower(): value for name, value in self.headers.items()}, body))
        try:
            parsed = json.loads(body)
        except ValueError:
            parsed = None
        try:
            action = fake.action_for(self.path, parsed)
        except Exception as exc:
            fake.errors.append(exc)
            action = (418, {"error": repr(exc)})
        self._act(action)

    def _act(self, action: object) -> None:
        if action == "drop":
            self.connection.shutdown(socket.SHUT_RDWR)
            self.close_connection = True
        elif action[0] == "sleep":
            time.sleep(action[1])
            self._act(action[2])
        elif action[0] == "redirect":
            self._reply(action[1], b"", location=action[2])
        else:
            status, payload = action
            self._reply(status, payload if isinstance(payload, bytes) else json.dumps(payload).encode("utf-8"))

    def _reply(self, status: int, data: bytes, location: str | None = None) -> None:
        try:
            self.send_response(status)
            if location is not None:
                self.send_header("Location", location)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        except OSError:
            pass                                 # the client stopped waiting for this reply

    def log_message(self, format: str, *args: object) -> None:
        pass


@pytest.fixture
def server():
    fake = FakeServer()
    yield fake
    fake.close()
    assert fake.errors == []


def replies(*actions: object) -> Callable[[str, object], object]:
    """An answer giving `actions` in turn, then HTTP 418 (never retried, so an extra request fails the test)."""
    queue = list(actions)
    lock = threading.Lock()

    def answer(path: str, body: object) -> object:
        with lock:
            return queue.pop(0) if queue else (418, {"error": "no scripted reply left"})
    return answer


class FakeClock:
    """A clock that starts at an arbitrary time and advances only when `sleep` is called; `sleeps` lists the
    pauses."""

    def __init__(self) -> None:
        self.t = 1000.0
        self.sleeps: list[float] = []

    def now(self) -> float:
        return self.t

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.t += seconds


def policy(base_url: str, clock: FakeClock) -> HttpPolicy:
    return HttpPolicy(base_url, "witness-policy", clock=clock.now, sleep=clock.sleep)


def closed_port_url() -> str:
    """A base URL on a local port nothing listens on."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    return f"http://127.0.0.1:{port}/v1"


MALFORMED_URL = "http://127.0.0.1:not_a_port/v1"
LONG_LABEL_URL = "http://" + "a" * 64 + ".example/v1"      # a host label may have at most 63 characters


# -------------------------------------------------------------- the policy: request, completion text, retries

BODY_SYS_USER = (b'{"messages":[{"role":"system","content":"SYS"},{"role":"user","content":"USER"}],'
                 b'"model":"witness-policy","max_tokens":8192,"temperature":1.0,'
                 b'"chat_template_kwargs":{"enable_thinking":false}}')
OK = (200, reply("fine"))


def test_a_call_posts_both_messages_as_compact_json_without_credentials_and_returns_the_completion(server):
    server.answer = replies(OK)
    assert HttpPolicy(server.base_url, "witness-policy")("SYS", "USER") == "fine"
    [request] = server.received
    assert request.path == "/v1/chat/completions"
    assert request.headers["content-type"] == "application/json"
    assert "authorization" not in request.headers
    assert request.body == BODY_SYS_USER


def test_the_body_keeps_non_ascii_text_raw_and_escapes_control_characters_and_quotes(server):
    system, user = "\u26a0 \u00e9", 'a\nb\t"c"'
    expected = (b'{"messages":[{"role":"system","content":"\xe2\x9a\xa0 \xc3\xa9"},'
                b'{"role":"user","content":"a\\nb\\t\\"c\\""}],"model":"witness-policy","max_tokens":8192,'
                b'"temperature":1.0,"chat_template_kwargs":{"enable_thinking":false}}')
    server.answer = replies(OK)
    HttpPolicy(server.base_url + "/", "witness-policy")(system, user)
    [request] = server.received
    assert request.path == "/v1/chat/completions"
    assert request.body == encode_body(chat_request_body("witness-policy", system, user)) == expected
    assert list(json.loads(request.body)) == ["messages", "model", "max_tokens", "temperature",
                                              "chat_template_kwargs"]


@pytest.mark.parametrize("content", ["  x \n\n", "<plan>1 2</plan><|endoftext|>", ""],
                         ids=["surrounding_whitespace", "stop_token_string", "empty"])
def test_the_completion_text_is_returned_exactly_and_other_reply_fields_are_ignored(server, content):
    payload = {"choices": [{"message": {"role": "assistant", "content": content, "reasoning_content": None},
                            "finish_reason": "length"}]}
    server.answer = replies((200, payload))
    clock = FakeClock()
    assert policy(server.base_url, clock)("s", "u") == content
    assert len(server.received) == 1 and clock.sleeps == []


@pytest.mark.parametrize("failures, pauses", [
    ([(503, {"error": "busy"})], [1.0]),
    ([(429, {"error": "slow down"}), (429, {"error": "slow down"})], [1.0, 5.0]),
    (["drop"], [1.0]),
    ([(200, reply(None))], [1.0]),
    ([(200, {"choices": [{"message": {"role": "assistant"}}]})], [1.0]),
    ([(200, reply(123))], [1.0]),
    ([(200, {"choices": []})], [1.0]),
    ([(200, b"<html>")], [1.0]),
    ([(200, [])], [1.0]),
    ([(200, {"choices": [{"message": "x"}]})], [1.0]),
], ids=["server_busy", "rate_limited_twice", "connection_dropped", "content_null", "content_absent",
        "content_not_a_string", "no_choices", "body_not_json", "body_a_json_array", "message_not_an_object"])
def test_retryable_failures_are_repeated_after_growing_pauses_until_a_completion_arrives(server, failures, pauses):
    server.answer = replies(*failures, OK)
    clock = FakeClock()
    assert policy(server.base_url, clock)("s", "u") == "fine"
    assert len(server.received) == len(failures) + 1
    assert clock.sleeps == pauses


def test_a_request_that_times_out_is_repeated(server, monkeypatch):
    monkeypatch.setattr(http_policy, "REQUEST_TIMEOUT_S", 0.2)
    server.answer = replies(("sleep", 1.0, (200, reply("late"))), (200, reply("second")))
    clock = FakeClock()
    assert policy(server.base_url, clock)("s", "u") == "second"
    assert len(server.received) == 2 and clock.sleeps == [1.0]


def test_a_server_that_always_fails_is_given_up_after_one_hour_of_pauses(server):
    server.answer = lambda path, body: (500, {"error": "internal"})
    clock = FakeClock()
    with pytest.raises(PolicyError) as raised:
        policy(server.base_url, clock)("s", "u")
    assert len(server.received) == 64
    assert clock.sleeps == [1.0, 5.0, 10.0, 20.0, 40.0] + [60.0] * 58
    assert str(raised.value).startswith("no completion after 64 attempts in 3556 s; last error: HTTP 500")
    assert raised.value.__cause__ is None


def test_a_refused_connection_is_repeated_until_the_hour_is_spent():
    clock = FakeClock()
    with pytest.raises(PolicyError) as raised:
        policy(closed_port_url(), clock)("s", "u")
    assert clock.sleeps == [1.0, 5.0, 10.0, 20.0, 40.0] + [60.0] * 58
    assert str(raised.value).startswith("no completion after 64 attempts in 3556 s; last error: ConnectError")
    assert isinstance(raised.value.__cause__, httpx.ConnectError)


@pytest.mark.parametrize("status, payload, fragment", [
    (400, {"error": "context length exceeded"}, "context length exceeded"),
    (401, {"error": "unauthorized"}, "unauthorized"),
    (404, {"error": "not found"}, "not found"),
    (422, {"error": "unprocessable"}, "unprocessable"),
], ids=["bad_request", "unauthorized", "not_found", "unprocessable"])
def test_other_http_statuses_raise_at_once(server, status, payload, fragment):
    server.answer = replies((status, payload))
    clock = FakeClock()
    with pytest.raises(PolicyError) as raised:
        policy(server.base_url, clock)("s", "u")
    assert len(server.received) == 1 and clock.sleeps == []
    assert f"HTTP {status} from {server.base_url}/chat/completions" in str(raised.value)
    assert fragment in str(raised.value)


@pytest.mark.parametrize("first, paths", [
    ((201, reply("fine")), ["/v1/chat/completions"]),
    ((203, reply("fine")), ["/v1/chat/completions"]),
    (("redirect", 307, "/v2/chat/completions"), ["/v1/chat/completions", "/v2/chat/completions"]),
], ids=["created", "non_authoritative", "temporary_redirect"])
def test_any_2xx_reply_is_read_and_redirects_are_followed(server, first, paths):
    server.answer = replies(first, OK)
    clock = FakeClock()
    assert policy(server.base_url, clock)("s", "u") == "fine"
    assert [r.path for r in server.received] == paths and clock.sleeps == []
    assert len({r.body for r in server.received}) == 1


@pytest.mark.parametrize("base_url, cause", [("ftp://127.0.0.1/v1", httpx.UnsupportedProtocol),
                                             (MALFORMED_URL, httpx.InvalidURL), (LONG_LABEL_URL, UnicodeError)],
                         ids=["unsupported_scheme", "malformed_port", "host_label_too_long"])
def test_a_malformed_or_unsupported_url_raises_at_once(base_url, cause):
    clock = FakeClock()
    with pytest.raises(PolicyError) as raised:
        policy(base_url, clock)("s", "u")
    assert clock.sleeps == []
    assert isinstance(raised.value.__cause__, cause)


# ------------------------------------------------------------------------------------------- the server check

PROBE_BODY = (b'{"model":"witness-policy","temperature":0.0,"max_tokens":600,'
              b'"messages":[{"role":"user","content":"Reply with exactly the word OK."}],'
              b'"chat_template_kwargs":{"enable_thinking":false}}')


@pytest.mark.parametrize("tokens", [2, 20])
def test_the_server_check_sends_one_short_prompt_and_returns_the_completion_tokens(server, tokens):
    server.probe_answer = (200, reply(tokens=tokens))
    assert probe_thinking_off(server.base_url, "witness-policy") == tokens
    [request] = server.received
    assert request.path == "/v1/chat/completions"
    assert request.body == PROBE_BODY


@pytest.mark.parametrize("answer, message", [
    ((200, reply(tokens=21)), "thinking-off probe failed: 21 completion tokens"),
    ((200, reply("<think>\nhmm")), "thinking-off probe failed: 2 completion tokens, content '<think>"),
    ((200, reply(reasoning_content="x")), "reasoning 1 chars"),
    ((200, reply(reasoning="x")), "reasoning 1 chars"),
    ((500, {"error": "internal"}), "probe got HTTP 500"),
    ((200, {"choices": reply()["choices"]}), "malformed probe reply"),
    ((200, {**reply(), "usage": {"completion_tokens": "2"}}), "malformed probe reply"),
    ((200, {**reply(), "usage": {"completion_tokens": True}}), "malformed probe reply"),
    ((200, reply(123)), "malformed probe reply"),
], ids=["too_many_tokens", "thinking_text", "reasoning_content", "reasoning", "server_error", "no_usage",
        "tokens_a_string", "tokens_a_boolean", "content_not_a_string"])
def test_the_server_check_fails_once_without_retrying_on_a_thinking_or_failing_server(server, answer, message):
    server.probe_answer = answer
    with pytest.raises(PreflightError, match=re.escape(message)):
        probe_thinking_off(server.base_url, "witness-policy")
    assert len(server.received) == 1


@pytest.mark.parametrize("answer", [
    lambda path: (201, PROBE_OK),
    lambda path: ("redirect", 308, "/v2/chat/completions") if path.startswith("/v1/") else (200, PROBE_OK),
], ids=["created", "permanent_redirect"])
def test_the_server_check_accepts_any_2xx_reply_and_follows_redirects(server, answer):
    server.probe_answer = answer
    assert probe_thinking_off(server.base_url, "witness-policy") == 2
    assert server.received[-1].body == PROBE_BODY


@pytest.mark.parametrize("make_url, cause", [(closed_port_url, httpx.ConnectError),
                                             (lambda: MALFORMED_URL, httpx.InvalidURL),
                                             (lambda: LONG_LABEL_URL, UnicodeError)],
                         ids=["nothing_listens", "malformed_port", "host_label_too_long"])
def test_the_server_check_fails_when_the_server_cannot_be_reached(make_url, cause):
    with pytest.raises(PreflightError) as raised:
        probe_thinking_off(make_url(), "witness-policy")
    assert isinstance(raised.value.__cause__, cause)
    assert str(raised.value).startswith(f"server unreachable: {type(raised.value.__cause__).__name__}: ")


# -------------------------------------------------------------------------------------------------- task rows

VALIDATION_TASKS = tuple(Task(g, 5, 250) for g in VALIDATION_GAMES)
VAL_ROW = '{"prompt": "tw08", "metadata": {"game_id": "tw08", "max_levels": 5, "max_reflection_calls": 250}}'


def test_the_validation_rows_cap_each_game_at_the_levels_of_its_level_file():
    tasks = load_tasks(REPOSITORY / "data" / "val.jsonl")
    assert tasks == VALIDATION_TASKS
    for task in tasks:
        entry = make_game(task.game_id).perform_action(ActionInput(id=GameAction.RESET))
        assert entry.win_levels == task.max_levels, task.game_id


@pytest.mark.parametrize("text, prefix", [
    (VAL_ROW + "\n{not json\n", ":2: "),
    ('{"prompt": "tw08"}\n', ":1: "),
    (VAL_ROW + "\n" + VAL_ROW + "\n", ": game ids on more than one row: tw08"),
    ("\n\n", ": no task rows"),
], ids=["line_not_json", "row_without_metadata", "game_on_two_rows", "blank_lines_only"])
def test_malformed_task_files_raise_value_error_naming_the_file(tmp_path, text, prefix):
    path = tmp_path / "tasks.jsonl"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError) as raised:
        load_tasks(path)
    assert str(raised.value).startswith(f"{path}{prefix}")


# ----------------------------------------------------------------------------------- the levels-beaten report

TASK_TW08, TASK_TW12, TASK_FV03 = Task("tw08", 5, 250), Task("tw12", 4, 250), Task("fv03", 5, 250)
BAD_REQUEST = 'PolicyError: HTTP 400 from http://127.0.0.1:30000/v1/chat/completions: {"error": "bad request"}'


def episode(task: Task, seed: int, levels_completed: int, stop_reason: str, *, calls: int = 0,
            steps: int = 0) -> EpisodeResult:
    """An episode of `task` with `calls` reflection calls and `steps` actions (their contents do not matter)."""
    return EpisodeResult(task.game_id, seed, task.max_levels,
                         calls=tuple(CallRecord(i, 0, 0, "s", "u", "", None) for i in range(calls)),
                         steps=tuple(StepRecord(0, i, 1, "random", False) for i in range(steps)),
                         resets=(), levels_completed=levels_completed, stop_reason=stop_reason)


def outcome(task: Task, seed: int, beaten: int | None, error: str = BAD_REQUEST) -> EpisodeOutcome:
    """The record of an episode that beat `beaten` levels, or of an aborted one (`beaten` None) with `error`."""
    if beaten is None:
        return EpisodeOutcome.from_result(task, episode(task, seed, 0, "aborted"), error)
    stop = "max_levels" if beaten == task.max_levels else "level_budget"
    return EpisodeOutcome.from_result(task, episode(task, seed, beaten, stop), None)


def test_an_episode_record_lists_its_fields_in_order():
    record = EpisodeOutcome.from_result(TASK_TW08, episode(TASK_TW08, 0, 2, "level_budget", calls=40, steps=371), None)
    assert json.dumps(record.to_json(), indent=2) == "\n".join([
        "{",
        '  "game_id": "tw08",',
        '  "seed": 0,',
        '  "max_levels": 5,',
        '  "max_reflection_calls": 250,',
        '  "levels_beaten": 2,',
        '  "levels_completed": 2,',
        '  "reflection_calls": 40,',
        '  "steps": 371,',
        '  "stop_reason": "level_budget",',
        '  "error": null,',
        '  "observation_version": 2',
        "}"])


def test_levels_beaten_are_capped_at_the_row_maximum_and_undefined_for_an_aborted_episode():
    capped = EpisodeOutcome.from_result(TASK_TW12, episode(TASK_TW12, 3, 5, "game_won", calls=7, steps=90), None)
    assert (capped.levels_beaten, capped.levels_completed, capped.max_levels) == (4, 5, 4)
    aborted = EpisodeOutcome.from_result(TASK_TW12, episode(TASK_TW12, 1, 1, "aborted", calls=3, steps=20),
                                         "RuntimeError: x")
    assert aborted == EpisodeOutcome("tw12", 1, 4, 250, None, 1, 3, 20, "aborted", "RuntimeError: x")


def example_report():
    """tw08 beats 2, 5, 4 levels, tw12 1, (aborted), 4 and fv03 0, 0, 3 with seeds 0, 1, 2."""
    levels = {TASK_TW08: (2, 5, 4), TASK_TW12: (1, None, 4), TASK_FV03: (0, 0, 3)}
    episodes = [outcome(task, seed, beaten) for task, row in levels.items()
                for seed, beaten in zip((0, 1, 2), row, strict=True)]
    return summarize("witness-policy", [TASK_TW08, TASK_TW12, TASK_FV03], [0, 1, 2], episodes[::-1])


def test_the_report_gives_per_game_and_per_seed_means_and_sample_deviations_over_completed_episodes():
    report = example_report()
    assert report.games == (GameSummary("tw08", 5, (2, 5, 4), 3.6666666666666665, 1.5275252316519468),
                            GameSummary("tw12", 4, (1, None, 4), 2.5, 2.1213203435596424),
                            GameSummary("fv03", 5, (0, 0, 3), 1.0, 1.7320508075688772))
    assert report.seed_totals == (3, None, 11)
    assert (report.max_total, report.mean_total, report.std_total) == (14, 7.0, 5.656854249492381)
    values = [report.mean_total, report.std_total] + [x for g in report.games for x in (g.mean, g.std)]
    assert all(type(x) is float for x in values)          # whole values too: results.json writes 7.0, not 7
    assert [(e.game_id, e.seed) for e in report.episodes] == [(g, s) for g in ("tw08", "tw12", "fv03")
                                                              for s in (0, 1, 2)]
    assert report.aborted == (outcome(TASK_TW12, 1, None),)
    assert (report.model, report.seeds) == ("witness-policy", (0, 1, 2))
    assert report.tasks == (TASK_TW08, TASK_TW12, TASK_FV03)


REPORT_EXAMPLE = """\
Levels beaten (model witness-policy; temperature 1.0; max_tokens 8192; thinking off)
observation version: 2
game    max  seed 0  seed 1  seed 2    mean     std
tw08      5       2       5       4    3.67    1.53
tw12      4       1   abort       4    2.50    2.12
fv03      5       0       0       3    1.00    1.73
total    14       3   abort      11    7.00    5.66
aborted: tw12 seed 1: PolicyError: HTTP 400 from http://127.0.0.1:30000/v1/chat/completions: {"error": "bad request"}
"""


def test_the_report_text_is_a_fixed_width_table_with_the_aborted_episodes_below():
    assert format_report(example_report()) == REPORT_EXAMPLE


def test_the_report_text_of_one_seed_has_no_deviations():
    report = summarize("witness-policy", [TASK_TW08], [7], [outcome(TASK_TW08, 7, 3)])
    assert format_report(report) == (
        "Levels beaten (model witness-policy; temperature 1.0; max_tokens 8192; thinking off)\n"
        "observation version: 2\n"
        "game    max  seed 7    mean     std\n"
        "tw08      5       3    3.00       -\n"
        "total     5       3    3.00       -\n")


def test_the_report_text_shows_only_the_first_line_of_an_error():
    report = summarize("m", [TASK_TW08], [0], [outcome(TASK_TW08, 0, None, "RuntimeError: first\nsecond")])
    assert format_report(report).endswith("\naborted: tw08 seed 0: RuntimeError: first\n")


def test_with_one_seed_every_deviation_is_undefined():
    report = summarize("witness-policy", [TASK_TW08, TASK_FV03], [0],
                       [outcome(TASK_TW08, 0, 2), outcome(TASK_FV03, 0, 5)])
    assert [(g.mean, g.std) for g in report.games] == [(2.0, None), (5.0, None)]
    assert (report.seed_totals, report.mean_total, report.std_total) == ((7,), 7.0, None)


def test_when_every_episode_aborted_no_mean_or_total_is_defined():
    report = summarize("witness-policy", [TASK_TW08, TASK_FV03], [0, 1],
                       [outcome(task, seed, None) for task in (TASK_TW08, TASK_FV03) for seed in (0, 1)])
    assert [(g.mean, g.std) for g in report.games] == [(None, None), (None, None)]
    assert (report.seed_totals, report.mean_total, report.std_total) == ((None, None), None, None)
    assert len(report.aborted) == 4 and report.max_total == 10


def test_a_row_allowing_more_levels_than_its_game_has_reports_the_game_levels_as_the_maximum():
    """A row allowing 10 levels for tw08, which has 5: its episode ends when the game is won."""
    tw08, tw01 = Task("tw08", 10, 300), Task("tw01", 10, 300)
    report = summarize("witness-policy", [tw08, tw01], [0],
                       [EpisodeOutcome.from_result(tw08, episode(tw08, 0, 5, "game_won"), None), outcome(tw01, 0, 3)])
    assert [g.max_levels for g in report.games] == [5, 10] and report.max_total == 15
    assert [g["max_levels"] for g in report.to_json()["games"]] == [5, 10]
    assert format_report(report).splitlines()[3:] == ["tw08      5       5    5.00       -",
                                                      "tw01     10       3    3.00       -",
                                                      "total    15       8    8.00       -"]


@pytest.mark.parametrize("seeds, episodes, message", [
    ([0, 1], [outcome(TASK_TW08, 0, 1)], "missing ['tw08 seed 1'], unexpected []"),
    ([0], [outcome(TASK_TW08, 0, 1), outcome(TASK_TW08, 0, 2)], "more than one episode of tw08 seed 0"),
    ([0], [outcome(TASK_TW08, 0, 1), outcome(TASK_TW08, 1, 2)], "missing [], unexpected ['tw08 seed 1']"),
    ([0, 0], [outcome(TASK_TW08, 0, 1)], "the seeds repeat a seed"),
], ids=["missing_episode", "episode_twice", "episode_not_asked_for", "seed_twice"])
def test_the_report_needs_exactly_one_episode_per_game_and_seed(seeds, episodes, message):
    with pytest.raises(ValueError, match=re.escape(message)):
        summarize("witness-policy", [TASK_TW08], seeds, episodes)


# ----------------------------------------------------------------------------------- the evaluator end to end
#
# Most runs play tw12 and fv03 with a cap of 5 reflection calls against a server that answers every reflection
# call with an empty text: each call is followed by one random move, so every episode ends after 5 calls and 5
# moves without beating a level, whatever the level contents.

CAPPED = (Task("tw12", 4, 5), Task("fv03", 5, 5))


def planless(path: str, body: object) -> object:
    """The answer to every reflection call: an empty completion text."""
    return 200, reply("")


def tasks_file(directory: Path, tasks: tuple[Task, ...]) -> Path:
    """A task file in `directory` with one row per task."""
    path = directory / "tasks.jsonl"
    path.write_text("".join(json.dumps({"prompt": t.game_id, "metadata": {
        "game_id": t.game_id, "max_levels": t.max_levels, "max_reflection_calls": t.max_reflection_calls}}) + "\n"
        for t in tasks), encoding="utf-8")
    return path


def run_cli(server: FakeServer, directory: Path, *options: str, out: str = "out",
            tasks: tuple[Task, ...] = CAPPED) -> tuple[int, Path]:
    """Run the command line against `server` with `tasks` and `options`; returns the exit status and the output
    directory `directory/out`."""
    out_dir = directory / out
    code = main(["--base-url", server.base_url, "--out", str(out_dir), "--tasks", str(tasks_file(directory, tasks)),
                 *options])
    return code, out_dir


def reflection_prompts(server: FakeServer) -> list[tuple[str, str]]:
    """The (system prompt, user message) of every request except the server check, in arrival order."""
    prompts = []
    for request in server.received:
        messages = json.loads(request.body)["messages"]
        if messages != [{"role": "user", "content": PROBE_PROMPT}]:
            prompts.append((messages[0]["content"], messages[1]["content"]))
    return prompts


def trace_lines(out: Path, game_id: str, seed: int, name: str) -> list[dict]:
    path = out / "traces" / game_id / f"seed{seed}" / name
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def capped_record(task: Task, seed: int) -> dict:
    """The record of a capped episode answered with empty texts only."""
    return {"game_id": task.game_id, "seed": seed, "max_levels": task.max_levels, "max_reflection_calls": 5,
            "levels_beaten": 0, "levels_completed": 0, "reflection_calls": 5, "steps": 5,
            "stop_reason": "call_budget", "error": None, "observation_version": 2}


CAPPED_REPORT = """\
Levels beaten (model witness-policy; temperature 1.0; max_tokens 8192; thinking off)
observation version: 2
game    max  seed 0  seed 1    mean     std
tw12      4       0       0    0.00    0.00
fv03      5       0       0    0.00    0.00
total     9       0       0    0.00    0.00
"""


def test_answers_without_a_plan_take_random_moves_until_the_call_cap_ends_every_episode(server, tmp_path, capsys):
    server.answer = planless
    code, out = run_cli(server, tmp_path, "--seeds", "0", "1")
    assert code == 0
    assert len(server.received) == 1 + 4 * 5
    for task in CAPPED:
        for seed in (0, 1):
            record = (out / "episodes" / f"{task.game_id}_seed{seed}.json").read_text(encoding="utf-8")
            assert record == json.dumps(capped_record(task, seed), indent=2) + "\n"
            moves = [step["action"] for step in trace_lines(out, task.game_id, seed, "steps.jsonl")]
            assert moves == fallback_moves(task.game_id, seed, 5)
            level_total = f"Level: 1/{task.max_levels}"
            assert all(level_total in call["user"] for call in trace_lines(out, task.game_id, seed, "calls.jsonl"))
    assert [step["action"] for step in trace_lines(out, "tw12", 0, "steps.jsonl")] == [1, 2, 2, 4, 1]
    assert [step["action"] for step in trace_lines(out, "fv03", 1, "steps.jsonl")] == [2, 4, 2, 2, 4]
    traced = [(call["system"], call["user"]) for task in CAPPED for seed in (0, 1)
              for call in trace_lines(out, task.game_id, seed, "calls.jsonl")]
    assert Counter(reflection_prompts(server)) == Counter(traced)
    bodies = [r.body for r in server.received[1:]]                 # the server check comes first
    assert Counter(bodies) == Counter(encode_body(chat_request_body("witness-policy", *p)) for p in traced)
    results = json.loads((out / "results.json").read_text(encoding="utf-8"))
    assert (results["max_total"], results["seed_totals"], results["mean_total"], results["std_total"]) == \
        (9, [0, 0], 0.0, 0.0)
    assert (out / "results.txt").read_text(encoding="utf-8") == capsys.readouterr().out == CAPPED_REPORT


def test_the_command_logs_the_server_check_and_the_end_of_each_episode_under_its_thread_name(server, tmp_path):
    server.answer = planless
    completed = subprocess.run(
        [sys.executable, "-m", "witness_agent.evaluate", "--base-url", server.base_url, "--out", str(tmp_path / "out"),
         "--tasks", str(tasks_file(tmp_path, CAPPED)), "--games", "tw12", "--seeds", "0", "--no-traces"],
        capture_output=True, text=True, env={**os.environ, "PYTHONPATH": str(REPOSITORY)}, check=False)
    assert completed.returncode == 0
    assert completed.stderr == (                   # no line per request: the HTTP client logs warnings only
        "MainThread: server check passed: thinking off (2 completion tokens)\n"
        "tw12-seed0: tw12 seed 0: 0/4 levels beaten (5 reflection calls, 5 steps, stop call_budget)\n")
    assert completed.stdout == (tmp_path / "out" / "results.txt").read_text(encoding="utf-8")


def replay_answer(ep: Episode) -> Callable[[str, object], object]:
    """An answer giving the stored responses of `ep` in turn, each after checking that the request carries the
    stored prompt of that call; a request beyond the stored calls fails the test."""
    calls = list(zip(ep.calls, ep.responses[:len(ep.calls)], strict=True))
    lock = threading.Lock()

    def answer(path: str, body: object) -> object:
        with lock:
            assert calls, f"{ep.name}: a request beyond the {len(ep.calls)} stored calls"
            call, response = calls.pop(0)
        system, user = (m["content"] for m in body["messages"])
        assert system == call.system, first_diff(call.system, system)
        assert user == call.user, first_diff(call.user, user)
        return 200, reply(response)
    return answer


EVALUATION_EPISODES = [name for name in STORE.episode_names()
                       if (ep := STORE.episode(name)).wiring == "eval" and ep.stop_reason != "policy_stop"]


def test_there_are_stored_evaluation_episodes_to_replay():
    assert EVALUATION_EPISODES


@pytest.mark.parametrize("name", EVALUATION_EPISODES)
def test_a_stored_evaluation_episode_replays_through_the_command_line_with_identical_prompts_and_outcome(
        server, tmp_path, name):
    ep = STORE.episode(name)
    server.answer = replay_answer(ep)
    code, out = run_cli(server, tmp_path, "--seeds", str(ep.seed),
                        tasks=(Task(ep.game_id, ep.max_levels, ep.max_reflection_calls),))
    assert server.errors == []                  # a prompt that differs from the stored one is reported here
    assert code == 0
    record = json.loads((out / "episodes" / f"{ep.game_id}_seed{ep.seed}.json").read_text(encoding="utf-8"))
    assert [record[k] for k in ("levels_completed", "levels_beaten", "reflection_calls", "steps", "stop_reason")] \
        == [ep.levels_completed, ep.levels_completed, len(ep.calls), len(ep.steps), ep.stop_reason]
    assert record["levels_completed"] <= record["max_levels"]
    calls = trace_lines(out, ep.game_id, ep.seed, "calls.jsonl")
    assert [(c["system"], c["user"], c["response"]) for c in calls] == \
        [(c.system, c.user, r) for c, r in zip(ep.calls, ep.responses[:len(ep.calls)], strict=True)]
    steps = trace_lines(out, ep.game_id, ep.seed, "steps.jsonl")
    assert [(s["level"], s["step"], s["action"], s["actor"], s["noop_reset"]) for s in steps] == list(ep.steps)


# Each case prepares `directory` (the output directory is `directory/out`) and returns the options to add and
# the start of the expected error output.
InputCase = Callable[[Path], tuple[list[str], str]]


def _game_not_in_the_tasks(directory: Path) -> tuple[list[str], str]:
    return ["--games", "no_such_game"], (f"error: --games not in {DEFAULT_TASKS}: no_such_game; the tasks file "
                                         f"offers {' '.join(VALIDATION_GAMES)}\n")


def _game_twice(directory: Path) -> tuple[list[str], str]:
    return ["--games", "tw12", "tw12"], (f"error: --games given twice: tw12; the tasks file offers "
                                         f"{' '.join(VALIDATION_GAMES)}\n")


def _seed_twice(directory: Path) -> tuple[list[str], str]:
    return ["--seeds", "0", "0"], "error: --seeds repeats a seed: 0 0\n"


def _earlier_output(name: str) -> InputCase:
    def case(directory: Path) -> tuple[list[str], str]:
        out = directory / "out"
        out.mkdir()
        if name in ("episodes", "traces"):
            (out / name).mkdir()
        else:
            (out / name).write_text("{}", encoding="utf-8")
        return [], f"error: {out} already holds evaluator outputs ({name}); use a new output directory\n"
    return case


def _tasks_text(text: str, problem: str) -> InputCase:
    def case(directory: Path) -> tuple[list[str], str]:
        path = directory / "rows.jsonl"
        path.write_text(text, encoding="utf-8")
        return ["--tasks", str(path)], f"error: {path}:1: {problem}"
    return case


def _tasks_missing(directory: Path) -> tuple[list[str], str]:
    path = directory / "missing.jsonl"
    return ["--tasks", str(path)], f"error: [Errno 2] No such file or directory: '{path}'\n"


def _output_is_a_file(directory: Path) -> tuple[list[str], str]:
    (directory / "out").write_text("", encoding="utf-8")
    return [], f"error: [Errno 17] File exists: '{directory / 'out'}'\n"


UNKNOWN_ROW = ('{"prompt": "no_such_game", "metadata": {"game_id": "no_such_game", "max_levels": 5, '
               '"max_reflection_calls": 250}}\n')


@pytest.mark.parametrize("case", [
    pytest.param(_game_not_in_the_tasks, id="game_not_in_the_tasks"),
    pytest.param(_game_twice, id="game_twice"),
    pytest.param(_seed_twice, id="seed_twice"),
    pytest.param(_earlier_output("results.json"), id="earlier_report"),
    pytest.param(_earlier_output("results.txt"), id="earlier_report_text"),
    pytest.param(_earlier_output("episodes"), id="earlier_episodes"),
    pytest.param(_earlier_output("traces"), id="earlier_traces"),
    pytest.param(_tasks_text("{not json\n", ""), id="tasks_not_json"),
    pytest.param(_tasks_text(UNKNOWN_ROW, "malformed task row"), id="unknown_game_in_the_tasks"),
    pytest.param(_tasks_missing, id="tasks_missing"),
    pytest.param(_output_is_a_file, id="output_is_a_file"),
])
def test_invalid_inputs_exit_with_status_2_before_any_request(server, tmp_path, capsys, case):
    options, message = case(tmp_path)
    code = main(["--base-url", server.base_url, "--out", str(tmp_path / "out"), *options])
    assert code == 2
    assert capsys.readouterr().err.startswith(message)
    assert server.received == []


def test_without_a_source_checkout_the_tasks_option_is_required(server, tmp_path, capsys, monkeypatch):
    absent = tmp_path / "absent" / "val.jsonl"
    monkeypatch.setattr(evaluate, "DEFAULT_TASKS", absent)
    code = main(["--base-url", server.base_url, "--out", str(tmp_path / "out")])
    assert code == 2
    assert capsys.readouterr().err == (f"error: the default task file {absent} does not exist (it is part of a "
                                       "source checkout); pass --tasks PATH\n")
    assert server.received == []


def test_a_thinking_server_fails_the_server_check_with_status_3_before_any_episode(server, tmp_path, capsys):
    server.probe_answer = (200, reply("<think>\nThe user wants", tokens=150))
    server.answer = planless
    code, out = run_cli(server, tmp_path)
    assert code == 3
    assert capsys.readouterr().err.startswith("error: thinking-off probe failed: 150 completion tokens")
    assert not (out / "episodes").exists()
    assert len(server.received) == 1


def test_an_episode_whose_request_is_refused_is_aborted_and_left_out_of_its_seed_total(server, tmp_path):
    refusal = '{"error": "caf\u00e9 closed"}'.encode()                  # non-ASCII text is kept as is
    server.answer = lambda path, body: (400, refusal) if path.startswith("/refused/") else (200, reply(""))

    def policy_for(game_id: str, seed: int) -> HttpPolicy:
        prefix = "/refused" if (game_id, seed) == ("tw12", 1) else ""
        return HttpPolicy(server.url + prefix + "/v1", "witness-policy")

    outcomes = run_validation(CAPPED, [0, 1], policy_for, tmp_path)
    assert [(o.game_id, o.seed) for o in outcomes] == [("tw12", 0), ("tw12", 1), ("fv03", 0), ("fv03", 1)]
    refused = outcomes[1]
    assert (refused.levels_beaten, refused.stop_reason, refused.reflection_calls) == (None, "aborted", 0)
    assert refused.error.startswith("PolicyError: HTTP 400") and refused.error.endswith('caf\u00e9 closed"}')
    assert (tmp_path / "episodes" / "tw12_seed1.json").read_text(encoding="utf-8") == \
        json.dumps(refused.to_json(), indent=2, ensure_ascii=False) + "\n"
    assert (tmp_path / "traces" / "tw12" / "seed1").is_dir()
    assert [o.to_json() for o in (outcomes[0], outcomes[2], outcomes[3])] == \
        [capped_record(CAPPED[0], 0), capped_record(CAPPED[1], 0), capped_record(CAPPED[1], 1)]
    assert summarize("witness-policy", CAPPED, [0, 1], outcomes).seed_totals == (0, None)


def test_when_every_episode_aborts_the_report_lists_them_and_the_exit_status_is_1(server, tmp_path):
    server.answer = lambda path, body: (400, {"error": "bad"})
    code, out = run_cli(server, tmp_path, "--seeds", "0", "1")
    assert code == 1
    results = json.loads((out / "results.json").read_text(encoding="utf-8"))
    pairs = [(g, s) for g in ("tw12", "fv03") for s in (0, 1)]
    assert [(e["game_id"], e["seed"], e["levels_beaten"]) for e in results["episodes"]] == \
        [(g, s, None) for g, s in pairs]
    assert all(e["error"].startswith("PolicyError: HTTP 400") for e in results["episodes"])
    assert (results["seed_totals"], results["mean_total"], results["std_total"]) == ([None, None], None, None)
    lines = (out / "results.txt").read_text(encoding="utf-8").splitlines()
    assert lines[-5].startswith("total ")
    assert all(line.startswith(f"aborted: {g} seed {s}: PolicyError: HTTP 400")
               for line, (g, s) in zip(lines[-4:], pairs, strict=True))


def test_a_policy_that_raises_aborts_its_episode_and_the_log_holds_the_whole_traceback(tmp_path, caplog):
    caplog.set_level(logging.ERROR)
    boom = RuntimeError("boom")

    def raising_policy(system: str, user: str) -> str:
        raise boom

    [aborted] = run_validation([CAPPED[0]], [0], lambda game_id, seed: raising_policy, tmp_path)
    assert (aborted.levels_beaten, aborted.stop_reason, aborted.reflection_calls, aborted.error) == \
        (None, "aborted", 0, "RuntimeError: boom")
    [record] = [r for r in caplog.records if r.name == "witness_agent.evaluate" and r.levelno == logging.ERROR]
    assert record.getMessage().startswith("tw12 seed 0: aborted after 0 reflection calls: RuntimeError: boom")
    assert isinstance(record.exc_info[1], EpisodeAborted) and record.exc_info[1].__cause__ is boom
    for text in ("Traceback", "RuntimeError: boom", "The above exception was the direct cause"):
        assert text in caplog.text


def test_an_error_outside_the_episodes_is_raised_once_every_episode_has_ended(tmp_path):
    calls: Counter[str] = Counter()

    def policy_for(game_id: str, seed: int) -> Callable[[str, str], str]:
        def answer(system: str, user: str) -> str:
            calls[game_id] += 1
            if game_id == "fv03":                  # a directory in its place: no fv03 record can be written
                (tmp_path / "episodes" / f"fv03_seed{seed}.json").mkdir(exist_ok=True)
            return ""
        return answer

    with pytest.raises(IsADirectoryError, match="fv03_seed0"):
        run_validation(CAPPED, [0, 1], policy_for, tmp_path)
    assert calls == {"tw12": 2 * 5, "fv03": 2 * 5}
    assert sorted(p.name for p in (tmp_path / "episodes").iterdir() if p.is_file()) == \
        ["tw12_seed0.json", "tw12_seed1.json"]


def test_a_trace_that_cannot_be_written_is_logged_and_costs_neither_its_record_nor_the_run(server, tmp_path, caplog):
    caplog.set_level(logging.ERROR)
    server.answer = planless
    (tmp_path / "traces").mkdir()
    (tmp_path / "traces" / "fv03").write_text("", encoding="utf-8")      # no fv03 trace can be written
    outcomes = run_validation(CAPPED, [0, 1], lambda game_id, seed: HttpPolicy(server.base_url, "witness-policy"),
                              tmp_path)
    assert [o.to_json() for o in outcomes] == [capped_record(task, seed) for task in CAPPED for seed in (0, 1)]
    assert sorted(p.name for p in (tmp_path / "episodes").iterdir()) == \
        ["fv03_seed0.json", "fv03_seed1.json", "tw12_seed0.json", "tw12_seed1.json"]
    assert all((tmp_path / "traces" / "tw12" / f"seed{seed}" / "summary.json").is_file() for seed in (0, 1))
    failures = sorted((r.getMessage(), isinstance(r.exc_info[1], OSError)) for r in caplog.records
                      if r.name == "witness_agent.evaluate" and r.levelno == logging.ERROR)
    assert failures == [(f"fv03 seed {seed}: the trace could not be written to {tmp_path / 'traces' / 'fv03'}"
                         f"/seed{seed}", True) for seed in (0, 1)]


def lone_surrogate_reply(content: str) -> Callable[[str, object], object]:
    """An answer whose raw JSON body holds `content` as written, so that an escape `\\ud83d` in it decodes to a lone
    surrogate (any server may send one; SGLang and vLLM never do)."""
    body = '{"choices":[{"message":{"role":"assistant","content":"%s"}}]}' % content
    return lambda path, parsed: (200, body.encode("ascii"))


def test_a_reply_with_a_lone_surrogate_outside_the_rules_completes_its_episode_and_the_report_is_written(
        server, tmp_path, capsys):
    server.answer = lone_surrogate_reply("<meta>\\ud83d</meta>")
    code, out = run_cli(server, tmp_path, "--seeds", "0", "1")
    assert code == 0
    assert len(server.received) == 1 + 4 * 5
    for task in CAPPED:
        for seed in (0, 1):
            record = (out / "episodes" / f"{task.game_id}_seed{seed}.json").read_text(encoding="utf-8")
            assert json.loads(record) == capped_record(task, seed)
    assert (out / "results.txt").read_text(encoding="utf-8") == capsys.readouterr().out == CAPPED_REPORT


def test_a_lone_surrogate_in_a_rule_aborts_each_episode_at_its_next_request_and_the_report_lists_them(
        server, tmp_path, capsys):
    server.answer = lone_surrogate_reply("<add>rule \\ud83d x</add>")
    code, out = run_cli(server, tmp_path, "--seeds", "0", "1")
    assert code == 1
    assert len(server.received) == 1 + 4                 # the second prompt of an episode holds the rule
    results = json.loads((out / "results.json").read_text(encoding="utf-8"))
    assert [(e["levels_beaten"], e["reflection_calls"], e["stop_reason"]) for e in results["episodes"]] == \
        [(None, 1, "aborted")] * 4
    assert all(e["error"].startswith("UnicodeEncodeError: 'utf-8' codec can't encode character '\\ud83d'")
               for e in results["episodes"])
    text = (out / "results.txt").read_text(encoding="utf-8")
    assert text == capsys.readouterr().out
    assert [line.split(": ")[1] for line in text.splitlines() if line.startswith("aborted: ")] == \
        [f"{g} seed {s}" for g in ("tw12", "fv03") for s in (0, 1)]


def test_the_output_files_keep_non_ascii_text_and_write_a_lone_surrogate_as_its_escape(tmp_path):
    write_utf8(tmp_path / "results.txt", "é \ud83d\n")
    assert (tmp_path / "results.txt").read_bytes() == "é \\ud83d\n".encode("utf-8")


def test_an_error_text_with_a_lone_surrogate_is_written_to_the_episode_record_as_a_json_escape(tmp_path):
    def raising_policy(system: str, user: str) -> str:
        raise RuntimeError("bad \ud83d text")

    [aborted] = run_validation([CAPPED[0]], [0], lambda game_id, seed: raising_policy, tmp_path, write_traces=False)
    data = (tmp_path / "episodes" / "tw12_seed0.json").read_bytes()
    assert b'"error": "RuntimeError: bad \\ud83d text"' in data
    assert json.loads(data)["error"] == aborted.error == "RuntimeError: bad \ud83d text"


class Interrupt(BaseException):
    """Stands for a KeyboardInterrupt or SystemExit raised inside an episode: not an `Exception`, so the episode
    loop does not turn it into an aborted episode."""


def test_an_interrupt_inside_an_episode_ends_its_thread_and_the_run_fails_naming_the_episode(tmp_path, monkeypatch):
    reported = []
    monkeypatch.setattr(threading, "excepthook", reported.append)
    interrupt = Interrupt()

    def interrupting_policy(system: str, user: str) -> str:
        raise interrupt

    def policy_for(game_id: str, seed: int) -> Callable[[str, str], str]:
        return interrupting_policy if seed == 1 else (lambda system, user: "")

    with pytest.raises(RuntimeError, match="^the episode thread of tw12 seed 1 ended without a record; "):
        run_validation(CAPPED[:1], [0, 1], policy_for, tmp_path, write_traces=False)
    assert [(r.thread.name, r.exc_value) for r in reported] == [("tw12-seed1", interrupt)]
    assert [p.name for p in (tmp_path / "episodes").iterdir()] == ["tw12_seed0.json"]


def test_a_game_that_cannot_be_made_stops_the_run_after_the_server_check_and_before_any_episode(
        server, tmp_path, monkeypatch):
    monkeypatch.setenv("ONLY_RESET_LEVELS", "true")
    server.answer = planless
    with pytest.raises(RuntimeError, match="ONLY_RESET_LEVELS"):
        run_cli(server, tmp_path)
    assert len(server.received) == 1
    assert not (tmp_path / "out" / "episodes").exists()


def test_without_traces_only_the_episode_records_and_the_report_are_written(server, tmp_path):
    server.answer = planless
    code, out = run_cli(server, tmp_path, "--seeds", "0", "1", "--no-traces")
    assert code == 0
    assert sorted(p.name for p in out.iterdir()) == ["episodes", "results.json", "results.txt"]
    assert len(list((out / "episodes").iterdir())) == 4


def output_files(out: Path) -> dict[Path, bytes]:
    """The episode records and trace files under `out`, by path relative to `out`."""
    return {p.relative_to(out): p.read_bytes() for name in ("episodes", "traces") for p in (out / name).rglob("*")
            if p.is_file()}


def test_episode_records_and_traces_do_not_depend_on_the_seed_order(server, tmp_path):
    server.answer = planless
    first = run_cli(server, tmp_path, "--seeds", "0", "1", out="first")
    second = run_cli(server, tmp_path, "--seeds", "1", "0", out="second")
    assert (first[0], second[0]) == (0, 0)
    assert len(output_files(first[1])) == 4 + 4 * 3
    assert output_files(first[1]) == output_files(second[1])


def test_all_episodes_run_at_once_each_in_a_daemon_thread_named_after_its_game_and_seed(tmp_path):
    barrier = threading.Barrier(4, timeout=10)
    threads: dict[tuple[str, int], tuple[str, bool]] = {}

    def policy_for(game_id: str, seed: int) -> Callable[[str, str], str]:
        def first_call_waits_for_the_other_episodes(system: str, user: str) -> str:
            if (game_id, seed) not in threads:
                thread = threading.current_thread()
                threads[(game_id, seed)] = (thread.name, thread.daemon)
                barrier.wait()          # breaks after 10 s unless all four episodes are in their first call
            return ""
        return first_call_waits_for_the_other_episodes

    outcomes = run_validation(CAPPED, [0, 1], policy_for, tmp_path, write_traces=False)
    assert [o.error for o in outcomes] == [None] * 4
    assert threads == {(g, s): (f"{g}-seed{s}", True) for g in ("tw12", "fv03") for s in (0, 1)}


def test_the_defaults_are_the_repository_validation_rows_and_seeds_0_1_2():
    assert DEFAULT_TASKS == REPOSITORY / "data" / "val.jsonl"
    assert load_tasks(DEFAULT_TASKS) == VALIDATION_TASKS
    assert DEFAULT_SEEDS == (0, 1, 2)


def test_by_default_every_row_plays_seeds_0_1_2_with_the_served_policy_and_the_report_records_the_settings(
        server, tmp_path):
    server.answer = planless
    code, out = run_cli(server, tmp_path)
    assert code == 0
    assert len(server.received) == 1 + 2 * 3 * 5
    assert all(b'"model":"witness-policy"' in r.body for r in server.received)
    settings = json.loads((out / "results.json").read_text(encoding="utf-8"))["settings"]
    assert json.dumps(settings) == json.dumps({
        "model": "witness-policy", "temperature": 1.0, "max_tokens": 8192,
        "chat_template_kwargs": {"enable_thinking": False}, "max_steps_per_level": 300, "observation_version": 2,
        "game_catalog": None, "seeds": [0, 1, 2],
        "tasks": [{"game_id": "tw12", "max_levels": 4, "max_reflection_calls": 5},
                  {"game_id": "fv03", "max_levels": 5, "max_reflection_calls": 5}]})
    for path in (p for p in out.rglob("*") if p.is_file()):
        text = path.read_text(encoding="utf-8")
        assert re.search(r"20\d\d-\d\d-\d\d", text) is None, path
        assert str(tmp_path) not in text and str(tmp_path.resolve()) not in text, path


def test_the_model_name_and_the_seed_order_come_from_the_command_line(server, tmp_path):
    server.answer = planless
    code, out = run_cli(server, tmp_path, "--model", "policy-b", "--seeds", "2", "0", tasks=CAPPED[:1])
    assert code == 0
    assert len(server.received) == 1 + 2 * 5
    assert all(b'"model":"policy-b"' in r.body for r in server.received)
    results = json.loads((out / "results.json").read_text(encoding="utf-8"))
    assert (results["settings"]["model"], results["settings"]["seeds"]) == ("policy-b", [2, 0])
    assert [(e["game_id"], e["seed"]) for e in results["episodes"]] == [("tw12", 2), ("tw12", 0)]
    assert (out / "results.txt").read_text(encoding="utf-8").splitlines()[:3] == [
        "Levels beaten (model policy-b; temperature 1.0; max_tokens 8192; thinking off)", "observation version: 2",
        "game    max  seed 2  seed 0    mean     std"]


def test_the_selected_games_keep_the_order_of_the_tasks_file_and_the_maximum_total_is_theirs(server, tmp_path):
    server.answer = planless
    code, out = run_cli(server, tmp_path, "--games", "tw08", "tw12", "--seeds", "0", "--no-traces",
                        tasks=CAPPED + (Task("tw08", 5, 5),))
    assert code == 0
    assert len(server.received) == 1 + 2 * 5
    results = json.loads((out / "results.json").read_text(encoding="utf-8"))
    assert [t["game_id"] for t in results["settings"]["tasks"]] == ["tw12", "tw08"]
    assert [(e["game_id"], e["seed"]) for e in results["episodes"]] == [("tw12", 0), ("tw08", 0)]
    assert results["max_total"] == 4 + 5


def test_a_missing_output_directory_is_created_with_its_parents(server, tmp_path):
    server.answer = planless
    code, out = run_cli(server, tmp_path, "--games", "tw12", "--seeds", "0", "--no-traces", out="new/nested/out")
    assert code == 0
    assert sorted(p.name for p in out.iterdir()) == ["episodes", "results.json", "results.txt"]


def test_a_seed_column_widens_so_that_a_long_seed_label_keeps_a_space_before_it(server, tmp_path):
    server.answer = planless
    code, out = run_cli(server, tmp_path, "--seeds", "1008", "7", "--no-traces", tasks=CAPPED[:1])
    assert code == 0
    assert (out / "results.txt").read_text(encoding="utf-8") == (
        "Levels beaten (model witness-policy; temperature 1.0; max_tokens 8192; thinking off)\n"
        "observation version: 2\n"
        "game    max seed 1008  seed 7    mean     std\n"
        "tw12      4         0       0    0.00    0.00\n"
        "total     4         0       0    0.00    0.00\n")


# -------------------------------------------------------------------------------------------- serving scripts

SCRIPTS = REPOSITORY / "scripts"


def launch_arguments(script: str) -> list[str]:
    """The arguments `script` passes to `python3 -m sglang.launch_server`."""
    command = script.split("exec python3 -m sglang.launch_server", 1)[1]
    return shlex.split(command.replace("\\\n", " "))


@pytest.mark.parametrize("name", ["serve_policy.sh", "run_validation.sh"])
def test_the_serving_scripts_are_executable(name):
    assert os.access(SCRIPTS / name, os.X_OK)


def test_the_policy_server_runs_with_the_validation_settings():
    assert launch_arguments((SCRIPTS / "serve_policy.sh").read_text(encoding="utf-8")) == [
        "--model-path", "$1", "--served-model-name", "witness-policy", "--tp", "1", "--port", "${2:-30000}",
        "--host", "127.0.0.1", "--mem-fraction-static", "0.85", "--context-length", "32768",
        "--sampling-defaults", "openai", "--default-chat-template-kwargs", '{"enable_thinking": false}',
        "--max-running-requests", "64"]


@pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
@pytest.mark.parametrize("name", ["serve_policy.sh", "run_validation.sh"])
def test_the_serving_scripts_are_valid_bash(name):
    subprocess.run(["bash", "-n", str(SCRIPTS / name)], check=True)


# -------------------------------------------------------------------------------------------- import boundary

HARNESS_WITHOUT_HTTP = """
import pkgutil, sys
sys.modules["httpx"] = None                     # any import of the HTTP client fails
import witness_env, witness_agent
evaluator = {"http_policy", "evaluate", "api_policy", "evaluate_api"}
for module in pkgutil.iter_modules(witness_agent.__path__):
    if module.name not in evaluator:
        __import__("witness_agent." + module.name)
assert not evaluator & {name.rsplit(".", 1)[-1] for name in sys.modules if name.startswith("witness_agent.")}
try:
    import witness_agent.http_policy
except ImportError:
    print("http_policy needs httpx")
"""


def test_the_harness_imports_without_the_http_client_and_the_http_policy_needs_it():
    completed = subprocess.run([sys.executable, "-c", HARNESS_WITHOUT_HTTP], capture_output=True, text=True,
                               env={**os.environ, "PYTHONPATH": str(REPOSITORY)}, check=False)
    assert (completed.returncode, completed.stdout, completed.stderr) == (0, "http_policy needs httpx\n", "")


def test_evaluate_plays_the_games_of_a_catalog_module_and_records_the_catalog_name(server, tmp_path, write_catalog):
    name = write_catalog()
    server.answer = planless
    tasks = (Task("zz01", 1, 2), Task("tw12", 4, 2))
    code, out = run_cli(server, tmp_path, "--seeds", "0", tasks=tasks)
    assert code == 0
    results = json.loads((out / "results.json").read_text(encoding="utf-8"))
    assert results["settings"]["game_catalog"] == name
    assert [e["game_id"] for e in results["episodes"]] == ["zz01", "tw12"]
