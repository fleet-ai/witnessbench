"""Frontier models through OpenRouter (`witness_agent.api_policy`, `witness_agent.journal`,
`witness_agent.evaluate_api`) and the handling of a reply cut off at the output limit (`witness_agent.loop`).

A stand-in OpenRouter on 127.0.0.1 serves the two public endpoint lists and streams chat completions as server-sent
events; a fake clock drives the retry pauses. No real provider is ever called.
"""
from __future__ import annotations

import json
import socket
import threading
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from witness_agent import EpisodeAborted, EpisodeLimits, run_episode
from witness_agent import evaluate_api
from witness_agent.api_policy import (OpenRouterPolicy, ProviderPolicyError, RequestPlan, normalize_provider,
                                      plan_request, read_stream)
from witness_agent.episode import Completion
from witness_agent.http_policy import PolicyError, PreflightError, encode_body
from witness_agent.journal import JournalPolicy, ResumeMismatch, prompt_digest, read_journal
from witness_agent.loop import TRUNCATED_NOTICE
from witness_agent.trace import write_trace
from witness_env import make_game

MODEL = "anthropic/claude-opus-5.5"
KEY = "sk-or-test-key"


def endpoint(tag: str, name: str, limit: str = "max_tokens", cap: int = 128000, reasoning: bool = True) -> dict:
    params = [limit, "stop"] + (["reasoning", "reasoning_effort"] if reasoning else [])
    return {"model_id": MODEL, "tag": tag, "provider_name": name, "max_completion_tokens": cap,
            "supported_parameters": params}


ENDPOINTS = [endpoint("anthropic", "Anthropic"), endpoint("amazon-bedrock", "Amazon Bedrock"),
             endpoint("google-vertex/global", "Google")]
ZDR = [endpoint("amazon-bedrock", "Amazon Bedrock"), endpoint("amazon-bedrock/us-east-1", "Amazon Bedrock"),
       endpoint("google-vertex/global", "Google"), {**endpoint("azure", "Azure"), "model_id": "openai/other"}]
USAGE = {"prompt_tokens": 1200, "completion_tokens": 900, "total_tokens": 2100, "cost": 0.0315,
         "completion_tokens_details": {"reasoning_tokens": 700}, "prompt_tokens_details": {"cached_tokens": 1000}}


def chunks(text: str = "<keep/>", finish: str = "stop", model: str = MODEL, provider: str | None = "Amazon Bedrock",
           usage: dict | None = None, reasoning: str = "thinking it over") -> list[object]:
    """The events of one streamed completion: a keep-alive comment, a reasoning delta, the text in two deltas, the
    finish reason, the usage chunk."""
    base = {"id": "gen-1", "model": model, **({"provider": provider} if provider is not None else {})}
    half = len(text) // 2
    return [": OPENROUTER PROCESSING",
            {**base, "choices": [{"index": 0, "delta": {"role": "assistant", "reasoning": reasoning}}]},
            {**base, "choices": [{"index": 0, "delta": {"content": text[:half]}}]},
            {**base, "choices": [{"index": 0, "delta": {"content": text[half:]}, "finish_reason": finish}]},
            {**base, "choices": [{"index": 0, "delta": {}, "finish_reason": finish}], "usage": usage or USAGE}]


class FakeOpenRouter:
    """Serves GET .../models/<id>/endpoints and GET .../endpoints/zdr, and answers each POST .../chat/completions
    with the next scripted action: ("sse", events[, done]) streams the events (dicts as `data:` lines, strings as
    given), then "data: [DONE]" unless done is False; (status, json) answers plainly. With no action left: HTTP 418,
    which is never retried."""

    def __init__(self) -> None:
        self.posts: list[tuple[dict[str, str], bytes]] = []
        self.script: list[object] = []
        self.endpoints = {"data": {"endpoints": ENDPOINTS}}
        self.zdr = {"data": ZDR}
        self._lock = threading.Lock()
        self._httpd = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        self._httpd.fake = self
        self.base_url = f"http://127.0.0.1:{self._httpd.server_address[1]}/api/v1"
        threading.Thread(target=self._httpd.serve_forever, args=(0.01,), daemon=True).start()

    def next_action(self) -> object:
        with self._lock:
            return self.script.pop(0) if self.script else (418, {"error": "no scripted reply left"})

    def close(self) -> None:
        self._httpd.shutdown()
        self._httpd.server_close()


class _Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        fake = self.server.fake
        if self.path.endswith("/endpoints/zdr"):
            self._json(200, fake.zdr)
        elif "/models/" in self.path and self.path.endswith("/endpoints"):
            self._json(200, fake.endpoints)
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self) -> None:
        fake = self.server.fake
        body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        fake.posts.append(({k.lower(): v for k, v in self.headers.items()}, body))
        action = fake.next_action()
        if action[0] == "sse":
            events, done = action[1], (action[2] if len(action) > 2 else True)
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            for event in events:
                line = event if isinstance(event, str) else "data: " + json.dumps(event)
                self.wfile.write((line + "\n\n").encode("utf-8"))
            if done:
                self.wfile.write(b"data: [DONE]\n\n")
            self.close_connection = True
        else:
            self._json(*action)

    def _json(self, status: int, payload: object) -> None:
        data = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, format: str, *args: object) -> None:
        pass


@pytest.fixture
def fake():
    server = FakeOpenRouter()
    yield server
    server.close()


class FakeClock:
    def __init__(self) -> None:
        self.t = 0.0
        self.sleeps: list[float] = []

    def now(self) -> float:
        return self.t

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.t += seconds


def make_policy(fake: FakeOpenRouter, clock: FakeClock) -> OpenRouterPolicy:
    plan = plan_request(MODEL, base_url=fake.base_url)
    return OpenRouterPolicy(plan, api_key=KEY, base_url=fake.base_url, clock=clock.now, sleep=clock.sleep)


# ------------------------------------------------------------------------------------------- the request plan

def test_the_plan_admits_every_zero_data_retention_provider_of_the_model_and_its_limit_name(fake):
    plan = plan_request(MODEL, base_url=fake.base_url)
    assert plan.providers == ("amazon-bedrock", "google-vertex")
    assert plan.limit_param == "max_tokens" and plan.max_tokens == 128000 and plan.effort == "max"
    assert plan.served_names == {"amazon bedrock", "google", "google vertex"}
    assert plan.body("S", "U") == {
        "model": MODEL,
        "messages": [{"role": "system", "content": "S"}, {"role": "user", "content": "U"}],
        "max_tokens": 128000,
        "reasoning": {"effort": "max"},
        "provider": {"only": ["amazon-bedrock", "google-vertex"], "allow_fallbacks": False,
                     "require_parameters": True, "data_collection": "deny", "zdr": True},
        "stream": True}
    assert "temperature" not in plan.body("S", "U")
    assert plan.describe()["temperature"] == "not sent (provider default)" and "messages" not in plan.describe()


def test_the_plan_uses_max_completion_tokens_where_the_endpoints_take_that_name(fake):
    fake.zdr = {"data": [{**endpoint("azure", "Azure", limit="max_completion_tokens"), "model_id": MODEL}]}
    assert plan_request(MODEL, base_url=fake.base_url).limit_param == "max_completion_tokens"


@pytest.mark.parametrize("zdr, providers, fragment", [
    ([], None, "no zero-data-retention endpoint"),
    (ZDR, ["anthropic"], "providers without a zero-data-retention endpoint"),
    ([endpoint("amazon-bedrock", "Amazon Bedrock"), endpoint("azure", "Azure", limit="max_completion_tokens")],
     None, "narrow the providers"),
    ([endpoint("amazon-bedrock", "Amazon Bedrock", cap=64000)], None, "allows at most 64000 output tokens"),
    ([endpoint("amazon-bedrock", "Amazon Bedrock", reasoning=False)], None, "does not accept reasoning"),
])
def test_the_plan_refuses_a_model_without_a_fitting_zero_data_retention_endpoint(fake, zdr, providers, fragment):
    fake.zdr = {"data": zdr}
    with pytest.raises(PreflightError, match=fragment):
        plan_request(MODEL, providers=providers, base_url=fake.base_url)


def test_an_unlisted_model_or_an_unreachable_list_fails_the_preflight(fake):
    fake.endpoints = {"data": {"endpoints": []}}
    with pytest.raises(PreflightError, match="lists no endpoint"):
        plan_request(MODEL, base_url=fake.base_url)
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    with pytest.raises(PreflightError, match="cannot fetch"):
        plan_request(MODEL, base_url=f"http://127.0.0.1:{port}/api/v1")


def test_provider_names_compare_across_display_names_and_slugs():
    assert normalize_provider("Amazon Bedrock") == normalize_provider("amazon-bedrock") == "amazon bedrock"


# ----------------------------------------------------------------------------------------------- the policy

def test_a_call_streams_the_plan_body_with_the_key_and_returns_the_answer_text_without_reasoning(fake):
    clock = FakeClock()
    fake.script = [("sse", chunks("<add>rule</add><plan>1,2</plan>"))]
    policy = make_policy(fake, clock)
    answer = policy("SYS", "USER")
    headers, body = fake.posts[0]
    assert headers["authorization"] == f"Bearer {KEY}" and body == encode_body(policy.plan.body("SYS", "USER"))
    assert answer.text == "<add>rule</add><plan>1,2</plan>" and "thinking" not in answer.text
    assert (answer.finish_reason, answer.truncated, answer.attempts, answer.seconds) == ("stop", False, 1, 0.0)
    assert answer.served_model == MODEL and answer.served_provider == "Amazon Bedrock"
    assert answer.usage == {"prompt_tokens": 1200, "completion_tokens": 900, "reasoning_tokens": 700,
                            "cached_tokens": 1000, "cost": 0.0315}


def test_a_reply_cut_off_at_the_output_limit_is_returned_as_truncated_without_retrying(fake):
    clock = FakeClock()
    fake.script = [("sse", chunks("", finish="length"))]
    answer = make_policy(fake, clock)("S", "U")
    assert answer.truncated and answer.text == "" and clock.sleeps == [] and len(fake.posts) == 1


def test_a_refusal_is_the_model_answer_and_is_not_repeated(fake):
    clock = FakeClock()
    fake.script = [("sse", chunks("", finish="content_filter"))]
    answer = make_policy(fake, clock)("S", "U")
    assert (answer.finish_reason, answer.text, answer.truncated) == ("content_filter", "", False)
    assert clock.sleeps == [] and len(fake.posts) == 1


@pytest.mark.parametrize("failure", [
    ("sse", chunks("   ")),                                        # finished without answer text
    ("sse", chunks("x", finish="tool_calls")),                     # another finish reason
    ("sse", [{"id": "g", "error": {"code": 502, "message": "upstream"}, "choices": [{"finish_reason": "error"}]}]),
    ("sse", chunks("<keep/>"), False),                              # the stream ends before [DONE]
    ("sse", ["data: {not json"]),
    ("sse", [{"id": "g", "choices": [1]}]),                         # a choice that is not an object
    ("sse", [{"id": "g", "choices": {"0": {"delta": {}}}}]),        # choices that are not a list
    ("sse", [{"id": "g", "choices": [{"delta": "x"}]}]),            # a delta that is not an object
    (429, {"error": "rate limited"}),
    (503, {"error": "unavailable"}),
])
def test_failures_that_a_repeat_can_overcome_are_retried_and_not_counted_as_calls(fake, failure):
    clock = FakeClock()
    fake.script = [failure, ("sse", chunks("<keep/>"))]
    answer = make_policy(fake, clock)("S", "U")
    assert answer.text == "<keep/>" and answer.attempts == 2 and clock.sleeps == [1.0] and answer.seconds == 1.0


@pytest.mark.parametrize("status", [400, 401, 402, 404])
def test_other_statuses_raise_at_once(fake, status):
    clock = FakeClock()
    fake.script = [(status, {"error": {"message": "no"}})]
    with pytest.raises(PolicyError, match=f"HTTP {status}"):
        make_policy(fake, clock)("S", "U")
    assert clock.sleeps == []


def test_a_call_gives_up_after_an_hour_of_pauses(fake):
    clock = FakeClock()
    fake.script = [(503, {"error": "down"})] * 200
    with pytest.raises(PolicyError, match="no usable completion"):
        make_policy(fake, clock)("S", "U")
    assert sum(clock.sleeps) <= 3600 and clock.sleeps[:6] == [1.0, 5.0, 10.0, 20.0, 40.0, 60.0]


@pytest.mark.parametrize("model, ok", [(MODEL, True), (MODEL + "-20260921", True), ("anthropic/claude-opus-5", False),
                                       (MODEL + "-preview", False), ("openai/gpt-6.1-sol", False)])
def test_only_the_requested_model_may_serve_a_reply(fake, model, ok):
    fake.script = [("sse", chunks("<keep/>", model=model))]
    policy = make_policy(fake, FakeClock())
    if ok:
        assert policy("S", "U").served_model == model
    else:
        with pytest.raises(ProviderPolicyError, match="served by model"):
            policy("S", "U")


@pytest.mark.parametrize("provider", ["Anthropic", "Azure", None])
def test_a_provider_outside_the_allowlist_or_an_unnamed_one_raises(fake, provider):
    fake.script = [("sse", chunks("<keep/>", provider=provider))]
    with pytest.raises(ProviderPolicyError, match="served by provider"):
        make_policy(fake, FakeClock())("S", "U")


def test_the_key_must_be_set(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    plan = RequestPlan(MODEL, "max", 128000, "max_tokens", ("amazon-bedrock",), frozenset({"amazon bedrock"}))
    with pytest.raises(PreflightError, match="OPENROUTER_API_KEY is not set"):
        OpenRouterPolicy(plan)


def test_the_stream_reader_ignores_comments_and_reasoning_and_keeps_the_last_usage():
    reply = read_stream([": OPENROUTER PROCESSING", "", "data: " + json.dumps(chunks("ab")[1]),
                         "data: " + json.dumps(chunks("ab")[2]), "data: " + json.dumps(chunks("ab")[3]),
                         "data: " + json.dumps(chunks("ab")[4]), "data: [DONE]"])
    assert (reply.text, reply.finish_reason, reply.usage["cost"]) == ("ab", "stop", 0.0315)


# ------------------------------------------------------------------------------ the loop: a truncated reply

def completion(text: str, finish: str = "stop") -> Completion:
    return Completion(text=text, finish_reason=finish, usage={"prompt_tokens": 10, "completion_tokens": 20,
                                                              "reasoning_tokens": 5, "cached_tokens": 0, "cost": 0.01},
                      served_model=MODEL, served_provider="Amazon Bedrock", attempts=1, seconds=2.0)


class Scripted:
    def __init__(self, answers: list[object]) -> None:
        self.answers = list(answers)
        self.users: list[str] = []

    def __call__(self, system: str, user: str) -> object:
        self.users.append(user)
        return self.answers.pop(0) if self.answers else None


def test_a_truncated_reply_counts_as_a_call_takes_no_action_and_is_announced_next():
    policy = Scripted([completion("<plan>4,4</plan>", finish="length"), completion("<keep/><plan>4</plan>")])
    result = run_episode(make_game("tw01"), seed=0, limits=EpisodeLimits(1), policy=policy)
    first, second = result.calls[0], result.calls[1]
    assert first.plan is None and first.meta["truncated"] and first.level_step == second.level_step == 0
    assert TRUNCATED_NOTICE not in policy.users[0] and TRUNCATED_NOTICE in policy.users[1]
    assert TRUNCATED_NOTICE not in policy.users[2]
    assert result.steps[0].actor == "plan" and result.steps[0].action == 4       # no random move after the cut-off


def test_truncated_replies_count_against_the_call_cap():
    policy = Scripted([completion("", finish="length")] * 5)
    result = run_episode(make_game("tw01"), seed=0, limits=EpisodeLimits(1, max_reflection_calls=3), policy=policy)
    assert result.stop_reason == "call_budget" and len(result.calls) == 3 and result.steps == ()


def test_a_completion_is_read_as_its_text_and_plain_traces_stay_unchanged(tmp_path):
    texts = ["<plan>4</plan>", "<keep/>"]
    limits = EpisodeLimits(1, max_reflection_calls=2)
    plain = run_episode(make_game("tw01"), seed=0, limits=limits, policy=Scripted(list(texts)))
    rich = run_episode(make_game("tw01"), seed=0, limits=limits, policy=Scripted([completion(t) for t in texts]))
    assert [c.user for c in plain.calls] == [c.user for c in rich.calls] and plain.steps == rich.steps
    cut = run_episode(make_game("tw01"), seed=0, limits=limits,
                      policy=Scripted([completion(texts[0], finish="length"), completion(texts[1])]))
    write_trace(plain, tmp_path / "plain")
    write_trace(cut, tmp_path / "rich")
    assert all("meta" not in json.loads(line) for line in (tmp_path / "plain" / "calls.jsonl").read_text().splitlines())
    assert "completions" not in json.loads((tmp_path / "plain" / "summary.json").read_text())
    summary = json.loads((tmp_path / "rich" / "summary.json").read_text())["completions"]
    assert summary["calls"] == 2 and summary["truncated"] == 1 and summary["cost"] == pytest.approx(0.02)
    assert summary["served_models"] == [MODEL] and summary["served_providers"] == ["Amazon Bedrock"]


# ------------------------------------------------------------------------------------- the journal and resume

class Counting:
    """Answers with `answers` in turn and counts the calls; raises RuntimeError once `fail_after` calls were made."""

    def __init__(self, answers: list[Completion], fail_after: int | None = None) -> None:
        self.answers = list(answers)
        self.calls = 0
        self.fail_after = fail_after

    def __call__(self, system: str, user: str) -> Completion:
        if self.fail_after is not None and self.calls >= self.fail_after:
            raise RuntimeError("the connection dropped")
        self.calls += 1
        return self.answers.pop(0)


ANSWERS = [completion("<plan>4</plan>"), completion("<plan>2</plan>"), completion("", finish="length"),
           completion("<keep/><plan>3,4</plan>"), completion("<keep/>")]


def play(policy: Callable[[str, str], object]):
    return run_episode(make_game("tw01"), seed=3, limits=EpisodeLimits(2, max_reflection_calls=5), policy=policy)


def test_an_interrupted_episode_resumes_from_its_journal_without_paying_twice(tmp_path):
    reference = play(Counting(list(ANSWERS)))
    path = tmp_path / "journal" / "tw01_seed3.jsonl"
    with pytest.raises(EpisodeAborted):
        play(JournalPolicy(Counting(list(ANSWERS), fail_after=3), path))
    assert len(read_journal(path)) == 3
    inner = Counting(list(ANSWERS[3:]))
    resumed_policy = JournalPolicy(inner, path)
    resumed = play(resumed_policy)
    assert resumed_policy.replayed == 3 and inner.calls == 2
    assert resumed.steps == reference.steps and [c.user for c in resumed.calls] == [c.user for c in reference.calls]
    assert [e["prompt_sha256"] for e in read_journal(path)] == [prompt_digest(c.system, c.user) for c in resumed.calls]


def test_a_journal_whose_prompts_changed_refuses_to_replay(tmp_path):
    path = tmp_path / "j.jsonl"
    JournalPolicy(Counting([completion("<keep/>")]), path)("S", "U1")
    with pytest.raises(ResumeMismatch):
        JournalPolicy(Counting([]), path)("S", "U2")


def test_an_unfinished_last_journal_line_is_dropped(tmp_path):
    """Also when it was cut inside a character; resuming cuts it off the file, so the next answer is a line of its
    own."""
    path = tmp_path / "j.jsonl"
    JournalPolicy(Counting([completion("<keep/>")]), path)("S", "U")
    path.write_text(path.read_text() + '{"index": 1, "prompt', encoding="utf-8")
    assert len(read_journal(path)) == 1
    complete = path.read_bytes()[:path.read_bytes().rfind(b"\n") + 1]
    path.write_bytes(complete + '{"index": 1, "completion": {"text": "caf'.encode("utf-8") + "\u00e9".encode()[:1])
    assert len(read_journal(path)) == 1
    resumed = JournalPolicy(Counting([completion("<plan>4</plan>")]), path)
    resumed("S", "U")
    resumed("S", "U2")
    assert resumed.replayed == 1 and [e["index"] for e in read_journal(path)] == [0, 1]
    assert path.read_bytes().startswith(complete) and path.read_bytes().count(b"\n") == 2


def test_a_reply_with_a_lone_surrogate_is_journaled_and_replayed_with_the_same_text(tmp_path):
    path = tmp_path / "j.jsonl"
    text = "<keep/> x \ud83d y"
    assert JournalPolicy(Counting([completion(text)]), path)("S", "U").text == text
    assert path.read_bytes().isascii()
    assert JournalPolicy(Counting([]), path)("S", "U").text == text


# ---------------------------------------------------------------------------------- the command end to end

def write_tasks(path: Path, calls: int) -> Path:
    path.write_text(json.dumps({"prompt": "tw01", "metadata": {"game_id": "tw01", "max_levels": 1,
                                                               "max_reflection_calls": calls}}) + "\n")
    return path


def test_the_command_runs_records_the_request_and_resumes_an_aborted_episode(fake, tmp_path, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", KEY)
    monkeypatch.setattr(evaluate_api.OpenRouterPolicy, "__init__", _with_base_url(fake.base_url))
    monkeypatch.setattr(evaluate_api, "plan_request", lambda model, **kw: plan_request(model, **kw,
                                                                                        base_url=fake.base_url))
    tasks = write_tasks(tmp_path / "t.jsonl", calls=3)
    out = tmp_path / "out"
    argv = ["--model", MODEL, "--tasks", str(tasks), "--out", str(out), "--seeds", "0"]
    fake.script = [("sse", chunks("<keep/>")), ("sse", chunks("", finish="length")), (402, {"error": "credits"})]
    assert evaluate_api.main(argv) == 1                                         # aborted on the third call
    record = json.loads((out / "episodes" / "tw01_seed0.json").read_text())
    assert record["error"].startswith("PolicyError: HTTP 402") and len(read_journal(out / "journal" /
                                                                                     "tw01_seed0.jsonl")) == 2
    assert evaluate_api.main(argv) == 2                                         # the run exists: --resume needed
    fake.posts.clear()
    fake.script = [("sse", chunks("<keep/>"))]
    assert evaluate_api.main(argv + ["--resume"]) == 0
    assert len(fake.posts) == 1                                                 # only the unpaid call
    results = json.loads((out / "results.json").read_text())
    assert results["settings"]["request"]["provider"]["zdr"] is True
    assert results["settings"]["observation_version"] == 2 and results["episodes"][0]["stop_reason"] == "call_budget"
    assert results["completions"]["calls"] == 3 and results["completions"]["truncated"] == 1
    assert "cut off at the output limit 1 (33.3%)" in (out / "results.txt").read_text()
    assert "WARNING" in (out / "results.txt").read_text()


def test_a_dry_run_prints_the_request_and_plays_nothing(fake, tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setattr(evaluate_api, "plan_request", lambda model, **kw: plan_request(model, **kw,
                                                                                        base_url=fake.base_url))
    tasks = write_tasks(tmp_path / "t.jsonl", calls=3)
    assert evaluate_api.main(["--model", MODEL, "--tasks", str(tasks), "--out", str(tmp_path / "o"),
                              "--dry-run"]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["episodes"] == 5 and printed["request"]["reasoning"] == {"effort": "max"}
    assert fake.posts == [] and not (tmp_path / "o").exists()


def _with_base_url(base_url: str):
    original = OpenRouterPolicy.__init__

    def init(self, plan, **kwargs):
        original(self, plan, **{**kwargs, "base_url": base_url})
    return init
