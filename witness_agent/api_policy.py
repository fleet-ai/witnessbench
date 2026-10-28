"""Frontier models through OpenRouter: the request plan and the policy.

`plan_request` checks a model and a provider allowlist against OpenRouter's public endpoint lists before a run (no
model is called) and fixes the request: the reasoning effort, the output limit under the parameter name the
endpoints accept, and a `provider` block that admits only the allowlisted zero-data-retention endpoints and never
falls back to others. `OpenRouterPolicy` answers each reflection call with one streamed chat completion of that
request and returns a `Completion`: the answer text (never the reasoning text), the finish reason, the token usage
and cost, and the model and provider that served it. A reply from another model or provider raises
`ProviderPolicyError`; there is no fallback model. No temperature is sent, so each endpoint samples at its provider's
default: the request admits only endpoints that accept every parameter it sends, and the zero-data-retention
endpoints of some models, such as anthropic/claude-opus-5.5, accept no temperature.
"""
from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Final

import httpx

from .episode import Completion
from .http_policy import (REQUEST_ERRORS, RETRY_BUDGET_S, RETRY_DELAYS_S, RETRYABLE_TRANSPORT, Failure, PolicyError,
                          PreflightError, encode_body)

OPENROUTER_URL: Final = "https://openrouter.ai/api/v1"
KEY_ENV: Final = "OPENROUTER_API_KEY"
DEFAULT_EFFORT: Final = "max"
DEFAULT_MAX_TOKENS: Final = 128_000
CONNECT_TIMEOUT_S: Final = 30.0
READ_TIMEOUT_S: Final = 600.0               # silence allowed between two received lines (keep-alive comments count)
PREFLIGHT_TIMEOUT_S: Final = 60.0
LIMIT_PARAMS: Final = ("max_tokens", "max_completion_tokens")
# Finish reasons that are the model's answer whatever their text: cut off at the output limit, or a refusal (counted
# as one call and never repeated).
ANSWERED: Final = ("length", "content_filter")

logger = logging.getLogger(__name__)


class ProviderPolicyError(PolicyError):
    """A reply came from a model other than the requested one, or from a provider outside the allowlist, or did not
    name them."""


def normalize_provider(name: str) -> str:
    """A provider name or slug in one comparable form: "Amazon Bedrock" and "amazon-bedrock" both give
    "amazon bedrock"."""
    return re.sub(r"[-_\s]+", " ", name.strip().lower())


@dataclass(frozen=True)
class RequestPlan:
    """The request of every reflection call of one run."""

    model: str                              # OpenRouter model id, e.g. "anthropic/claude-opus-5.5"
    effort: str                             # reasoning effort, e.g. "max"
    max_tokens: int                         # output limit (reasoning and answer together)
    limit_param: str                        # the name the pinned endpoints accept for it (`LIMIT_PARAMS`)
    providers: tuple[str, ...]              # provider slugs admitted (`provider.only`), sorted
    served_names: frozenset[str]            # normalized names a reply may give as its provider

    def body(self, system: str, user: str) -> dict[str, object]:
        """The request body of one call, keys in this order: model, the system prompt and the user message as the
        only two messages, the output limit, the reasoning effort, the provider block, and stream."""
        return {"model": self.model,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                self.limit_param: self.max_tokens,
                "reasoning": {"effort": self.effort},
                "provider": {"only": list(self.providers), "allow_fallbacks": False, "require_parameters": True,
                             "data_collection": "deny", "zdr": True},
                "stream": True}

    def describe(self) -> dict[str, object]:
        """The request as sent, without the messages (for the run's settings); temperature is not sent."""
        body = self.body("", "")
        del body["messages"]
        return {**body, "temperature": "not sent (provider default)"}


def plan_request(model: str, *, effort: str = DEFAULT_EFFORT, max_tokens: int = DEFAULT_MAX_TOKENS,
                 providers: Iterable[str] | None = None, base_url: str = OPENROUTER_URL,
                 get: Callable[[str], object] | None = None) -> RequestPlan:
    """The request plan of `model`, checked against OpenRouter's public endpoint lists (two GET requests, no model
    call): `<base_url>/models/<model>/endpoints` must list the model, and `<base_url>/endpoints/zdr` must list at
    least one zero-data-retention endpoint of it from each provider in `providers` (default: every provider that has
    one). Every admitted endpoint must accept reasoning and allow `max_tokens` output tokens, and all of them must
    take the output limit under the same parameter name. Raises PreflightError otherwise, or when a list cannot be
    fetched. `get` fetches a URL and returns the parsed JSON (default: an httpx GET).
    """
    fetch = get or _get_json
    listed = _data(fetch(f"{base_url.rstrip('/')}/models/{model}/endpoints"), "endpoints")
    if not listed:
        raise PreflightError(f"OpenRouter lists no endpoint of {model}")
    zdr = [e for e in _data(fetch(f"{base_url.rstrip('/')}/endpoints/zdr"), None) if e.get("model_id") == model]
    slugs = sorted({str(e.get("tag", "")).split("/")[0] for e in zdr if e.get("tag")})
    if not slugs:
        raise PreflightError(f"{model} has no zero-data-retention endpoint on OpenRouter")
    chosen = sorted(set(providers)) if providers is not None else slugs
    outside = [p for p in chosen if p not in slugs]
    if not chosen or outside:
        raise PreflightError(f"providers without a zero-data-retention endpoint of {model}: {outside or chosen}; "
                             f"it has: {' '.join(slugs)}")
    admitted = [e for e in zdr if str(e.get("tag", "")).split("/")[0] in chosen]
    problems = []
    names = set()
    params = set()
    for e in admitted:
        supported = set(e.get("supported_parameters") or [])
        names.update({normalize_provider(str(e.get("provider_name", ""))), normalize_provider(e["tag"].split("/")[0])})
        if "reasoning" not in supported:
            problems.append(f"{e['tag']} does not accept reasoning")
        limit = [p for p in LIMIT_PARAMS if p in supported]
        if len(limit) != 1:
            problems.append(f"{e['tag']} accepts output-limit parameters {limit}")
        params.update(limit)
        cap = e.get("max_completion_tokens")
        if isinstance(cap, int) and cap < max_tokens:
            problems.append(f"{e['tag']} allows at most {cap} output tokens")
    if len(params) > 1:
        problems.append(f"the admitted endpoints take the output limit as {sorted(params)}; narrow the providers")
    if problems:
        raise PreflightError("; ".join(problems))
    return RequestPlan(model=model, effort=effort, max_tokens=max_tokens, limit_param=params.pop(),
                       providers=tuple(chosen), served_names=frozenset(n for n in names if n))


@dataclass(frozen=True)
class _Reply:
    """One complete streamed reply."""

    text: str
    finish_reason: str | None
    model: str | None
    provider: str | None
    usage: Mapping[str, int | float | None]


class OpenRouterPolicy:
    """The policy of one run: every call sends `plan.body(system, user)` and returns a `Completion`.

    A reply is used when it finished ("stop") with answer text that is not only white space, hit the output limit
    ("length", returned as a truncated Completion whatever its text), or was refused ("content_filter", returned as
    it is: an empty refusal is an answer without a plan). Repeated, after the pauses `RETRY_DELAYS_S` (the last one
    repeating) while the time since the first attempt plus the next pause stays within `RETRY_BUDGET_S`: timeouts,
    network errors, dropped connections, HTTP 408, 429 and 5xx, an error event in the stream, a stream that ends
    before "[DONE]", a malformed chunk, and a finished reply whose answer text is empty or only white space or that
    has another finish reason. Any other HTTP status (e.g. 400, 401, 402 out of credits, 404 no endpoint) raises
    PolicyError at once. A reply that does not name its model and provider, or names a model other than the
    requested one (the id itself, or the id with a "-YYYYMMDD" suffix) or a provider outside `plan.served_names`,
    raises ProviderPolicyError at once. `gate`, when given, bounds the requests in flight across the policies that
    share it. The API key is read from `KEY_ENV` when `api_key` is None; it is sent only in the Authorization header
    and never logged.
    """

    def __init__(self, plan: RequestPlan, *, api_key: str | None = None, base_url: str = OPENROUTER_URL,
                 gate: threading.Semaphore | None = None, clock: Callable[[], float] = time.monotonic,
                 sleep: Callable[[float], None] = time.sleep) -> None:
        key = api_key if api_key is not None else os.environ.get(KEY_ENV, "")
        if not key:
            raise PreflightError(f"{KEY_ENV} is not set")
        self.plan = plan
        self._url = base_url.rstrip("/") + "/chat/completions"
        self._headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
        self._gate = gate
        self._clock = clock
        self._sleep = sleep
        self._model_pattern = re.compile(re.escape(plan.model) + r"(-\d{8})?")

    def __call__(self, system: str, user: str) -> Completion:
        body = encode_body(self.plan.body(system, user))
        start = self._clock()
        attempt = 0
        while True:
            attempt += 1
            if self._gate is not None:
                with self._gate:
                    outcome = self._attempt(body)
            else:
                outcome = self._attempt(body)
            if isinstance(outcome, _Reply):
                self._check_served(outcome)
                if outcome.finish_reason in ANSWERED or (outcome.finish_reason == "stop" and outcome.text.strip()):
                    return Completion(text=outcome.text, finish_reason=outcome.finish_reason, usage=outcome.usage,
                                      served_model=outcome.model, served_provider=outcome.provider,
                                      attempts=attempt, seconds=self._clock() - start)
                outcome = Failure(f"unusable reply: finish reason {outcome.finish_reason!r}, "
                                   f"{len(outcome.text)} characters of answer text")
            delay = RETRY_DELAYS_S[min(attempt - 1, len(RETRY_DELAYS_S) - 1)]
            elapsed = self._clock() - start
            if elapsed + delay > RETRY_BUDGET_S:
                raise PolicyError(f"no usable completion after {attempt} attempts in {elapsed:.0f} s; "
                                  f"last error: {outcome.message}") from outcome.cause
            logger.warning("completion failed (%s); retrying in %.0f s", outcome.message, delay)
            self._sleep(delay)

    def _check_served(self, reply: _Reply) -> None:
        if not reply.model or not self._model_pattern.fullmatch(reply.model):
            raise ProviderPolicyError(f"requested {self.plan.model}, the reply was served by model {reply.model!r}")
        if not reply.provider or normalize_provider(reply.provider) not in self.plan.served_names:
            raise ProviderPolicyError(f"the reply was served by provider {reply.provider!r}, outside the allowlist "
                                      f"{' '.join(self.plan.providers)}")

    def _attempt(self, body: bytes) -> _Reply | Failure:
        """One request: the complete reply, or the failure to retry; raises PolicyError for a failure that
        repeating the request cannot overcome."""
        try:
            with httpx.stream("POST", self._url, content=body, headers=self._headers, follow_redirects=True,
                              timeout=httpx.Timeout(READ_TIMEOUT_S, connect=CONNECT_TIMEOUT_S)) as response:
                status = response.status_code
                if not 200 <= status <= 299:
                    detail = response.read()[:500].decode("utf-8", errors="replace")
                    if status in (408, 429) or 500 <= status <= 599:
                        return Failure(f"HTTP {status}: {detail}")
                    raise PolicyError(f"HTTP {status} from {self._url}: {detail}")
                return read_stream(response.iter_lines())
        except RETRYABLE_TRANSPORT as exc:
            return Failure(f"{type(exc).__name__}: {exc}", exc)
        except REQUEST_ERRORS as exc:
            raise PolicyError(f"{type(exc).__name__}: {exc}") from exc


def read_stream(lines: Iterable[str]) -> _Reply | Failure:
    """The reply carried by the server-sent event `lines` of one streamed chat completion: the answer text is the
    concatenated `delta.content` (reasoning deltas are ignored), the finish reason the last one given, the model and
    provider the first ones given, the usage the last usage object. Every line other than a "data:" line (comments,
    blank lines, other event fields) is skipped; an error event, a chunk that is not a JSON object (or whose choices
    or deltas are not), or the end of the lines before "data: [DONE]" is a failure."""
    text: list[str] = []
    finish: str | None = None
    model = provider = None
    usage: Mapping[str, object] | None = None
    for line in lines:
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            return _Reply("".join(text), finish, model, provider, _usage(usage))
        try:
            chunk = json.loads(data)
        except ValueError:
            return Failure(f"malformed stream chunk: {data[:200]}")
        if not isinstance(chunk, dict):
            return Failure(f"malformed stream chunk: {data[:200]}")
        if chunk.get("error"):
            error = chunk["error"]
            code = error.get("code") if isinstance(error, dict) else None
            message = error.get("message") if isinstance(error, dict) else error
            return Failure(f"error event in the stream (code {code}): {str(message)[:200]}")
        model = model or chunk.get("model")
        provider = provider or chunk.get("provider")
        if isinstance(chunk.get("usage"), dict):
            usage = chunk["usage"]
        choices = chunk.get("choices") or []
        if not isinstance(choices, list) or not all(isinstance(choice, dict) for choice in choices):
            return Failure(f"malformed stream chunk: {data[:200]}")
        for choice in choices:
            delta = choice.get("delta") or {}
            if not isinstance(delta, dict):
                return Failure(f"malformed stream chunk: {data[:200]}")
            if isinstance(delta.get("content"), str):
                text.append(delta["content"])
            if choice.get("finish_reason"):
                finish = choice["finish_reason"]
    return Failure("the stream ended before [DONE]")


def _usage(usage: Mapping[str, object] | None) -> dict[str, int | float | None]:
    """prompt_tokens, completion_tokens, reasoning_tokens, cached_tokens and cost (US dollars) of a usage object;
    None for each value not reported."""
    u = usage or {}
    completion_details = u.get("completion_tokens_details") or {}
    prompt_details = u.get("prompt_tokens_details") or {}
    return {"prompt_tokens": u.get("prompt_tokens"), "completion_tokens": u.get("completion_tokens"),
            "reasoning_tokens": completion_details.get("reasoning_tokens"),
            "cached_tokens": prompt_details.get("cached_tokens"), "cost": u.get("cost")}


def _get_json(url: str) -> object:
    """GET `url` and return its JSON; PreflightError on any failure."""
    try:
        response = httpx.get(url, timeout=PREFLIGHT_TIMEOUT_S, follow_redirects=True)
    except REQUEST_ERRORS as exc:
        raise PreflightError(f"cannot fetch {url}: {type(exc).__name__}: {exc}") from exc
    if not 200 <= response.status_code <= 299:
        raise PreflightError(f"HTTP {response.status_code} from {url}: {response.text[:300]}")
    try:
        return response.json()
    except ValueError as exc:
        raise PreflightError(f"{url} did not return JSON") from exc


def _data(payload: object, key: str | None) -> list[dict]:
    """The endpoint list of an OpenRouter list response: `data[key]` (or `data` itself when `key` is None)."""
    data = payload.get("data") if isinstance(payload, dict) else None
    items = data.get(key) if key is not None and isinstance(data, dict) else data
    if not isinstance(items, list):
        raise PreflightError(f"unexpected endpoint list: {str(payload)[:300]}")
    return [e for e in items if isinstance(e, dict)]
