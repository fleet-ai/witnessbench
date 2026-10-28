"""The policy of an OpenAI-compatible chat-completions server, and the check that the server answers without thinking.

`HttpPolicy` answers each reflection call with a chat-completion request, repeating it while a later attempt can still
succeed; `probe_thinking_off` checks before an evaluation that the served chat template renders prompts with thinking
off.
"""
from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final

import httpx

REQUEST_TIMEOUT_S: Final = 1200.0          # connect/read/write/pool timeout of every chat-completion request
MAX_TOKENS: Final = 8192
TEMPERATURE: Final = 1.0
CHAT_TEMPLATE_KWARGS: Final = MappingProxyType({"enable_thinking": False})
RETRY_DELAYS_S: Final = (1.0, 5.0, 10.0, 20.0, 40.0, 60.0)   # the last value repeats
RETRY_BUDGET_S: Final = 3600.0             # wall clock per policy call, measured from its first attempt
PROBE_PROMPT: Final = "Reply with exactly the word OK."
PROBE_MAX_TOKENS: Final = 600
PROBE_MAX_COMPLETION_TOKENS: Final = 20
PROBE_TIMEOUT_S: Final = 600.0

# Transport errors after which the same request can succeed: timeouts, network failures, a dropped connection.
RETRYABLE_TRANSPORT: Final = (httpx.TimeoutException, httpx.NetworkError, httpx.RemoteProtocolError)
# Everything httpx.post raises instead of returning a reply: a malformed URL (UnicodeError: a host name the resolver
# cannot encode, e.g. a label longer than 63 characters) and every request error.
REQUEST_ERRORS: Final = (httpx.InvalidURL, httpx.HTTPError, UnicodeError)
logger = logging.getLogger(__name__)


class PolicyError(RuntimeError):
    """No completion could be obtained: a non-retryable HTTP status, a non-retryable request error, or
    retries exhausted within RETRY_BUDGET_S."""


class PreflightError(RuntimeError):
    """A check made before any episode failed or could not be made: the thinking-off probe of a served model, the
    OpenRouter endpoint checks of `api_policy.plan_request`, or a missing OpenRouter API key."""


def chat_request_body(model: str, system: str, user: str) -> dict[str, object]:
    """The request body of one reflection call, keys in this order: the system prompt and the user message as the
    only two messages, the model name, `MAX_TOKENS`, `TEMPERATURE` and `CHAT_TEMPLATE_KWARGS`."""
    return {"messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "model": model, "max_tokens": MAX_TOKENS, "temperature": TEMPERATURE,
            "chat_template_kwargs": dict(CHAT_TEMPLATE_KWARGS)}


def encode_body(body: Mapping[str, object]) -> bytes:
    """`body` as compact JSON (no spaces after separators) in UTF-8, non-ASCII characters written unescaped."""
    return json.dumps(body, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


@dataclass(frozen=True)
class Failure:
    """A failed attempt that the same request may overcome: its description and the transport error, if any."""

    message: str
    cause: httpx.TransportError | None = None


class HttpPolicy:
    """The policy served as `model` at `base_url` (e.g. "http://127.0.0.1:30000/v1").

    Each call POSTs `chat_request_body(model, system, user)` to `<base_url>/chat/completions`, following
    redirects, and returns `choices[0].message.content` of the reply exactly as received ("" included); nothing
    else of the reply is read. Any 2xx status is a reply. A failed attempt is repeated when it can succeed later:
    a timeout, a network error or a dropped connection, HTTP 429 or 5xx, and a 2xx reply without completion text
    (a body that is not JSON, or no string at `choices[0].message.content`). The pauses before the repeats are
    `RETRY_DELAYS_S`, the last one repeating; when the time since the first attempt plus the next pause would
    exceed `RETRY_BUDGET_S`, the call raises `PolicyError`. Any other HTTP status and any other request error (a
    malformed URL, a URL scheme other than http/https, a body that cannot be decoded) raise `PolicyError` at once.
    A call keeps no state, so one object may be called from several threads at a time; `clock` and `sleep` measure
    and wait between attempts.
    """

    def __init__(self, base_url: str, model: str, *, clock: Callable[[], float] = time.monotonic,
                 sleep: Callable[[float], None] = time.sleep) -> None:
        self.base_url = base_url
        self.model = model
        self._clock = clock
        self._sleep = sleep

    def __call__(self, system: str, user: str) -> str:
        url = _completions_url(self.base_url)
        body = encode_body(chat_request_body(self.model, system, user))
        start = self._clock()
        attempt = 0
        while True:
            attempt += 1
            outcome = _attempt(url, body)
            if isinstance(outcome, str):
                return outcome
            delay = RETRY_DELAYS_S[min(attempt - 1, len(RETRY_DELAYS_S) - 1)]
            elapsed = self._clock() - start
            if elapsed + delay > RETRY_BUDGET_S:
                raise PolicyError(f"no completion after {attempt} attempts in {elapsed:.0f} s; "
                                  f"last error: {outcome.message}") from outcome.cause
            logger.warning("chat completion failed (%s); retrying in %.0f s", outcome.message, delay)
            self._sleep(delay)


def probe_thinking_off(base_url: str, model: str) -> int:
    """Check that `model` at `base_url` answers without thinking; returns the completion tokens of the answer.

    Sends one request, never repeated: the user message `PROBE_PROMPT` at temperature 0 with at most
    `PROBE_MAX_TOKENS` tokens and `CHAT_TEMPLATE_KWARGS`, following redirects. Raises PreflightError when the
    request fails without a reply (a malformed URL, an unreachable server), answers with a status outside 2xx, or
    sends a reply without `choices[0].message`, without an integer `usage.completion_tokens` or with a non-string
    content or reasoning text; and when the answer takes more than `PROBE_MAX_COMPLETION_TOKENS` tokens, does not
    contain "OK", or carries a reasoning text (`reasoning_content` or `reasoning`). A model that thinks first needs
    far more tokens for this prompt, and a server that separates reasoning moves the text there.
    """
    url = _completions_url(base_url)
    body = encode_body({"model": model, "temperature": 0.0, "max_tokens": PROBE_MAX_TOKENS,
                        "messages": [{"role": "user", "content": PROBE_PROMPT}],
                        "chat_template_kwargs": dict(CHAT_TEMPLATE_KWARGS)})
    try:
        reply = _post(url, body, PROBE_TIMEOUT_S)
    except REQUEST_ERRORS as exc:
        raise PreflightError(f"server unreachable: {type(exc).__name__}: {exc} ({url})") from exc
    if not 200 <= reply.status_code <= 299:
        raise PreflightError(f"probe got HTTP {reply.status_code} from {url}: {reply.text[:500]}")
    try:
        data = reply.json()
        message = data["choices"][0]["message"]
        tokens = data["usage"]["completion_tokens"]
    except (ValueError, KeyError, IndexError, TypeError):
        message = tokens = None
    if not isinstance(message, dict) or isinstance(tokens, bool) or not isinstance(tokens, int):
        raise PreflightError(f"malformed probe reply: {reply.text[:500]}")
    content = message.get("content") or ""
    reasoning = message.get("reasoning_content") or message.get("reasoning") or ""
    if not isinstance(content, str) or not isinstance(reasoning, str):
        raise PreflightError(f"malformed probe reply: {reply.text[:500]}")
    if not (tokens <= PROBE_MAX_COMPLETION_TOKENS and "OK" in content and not reasoning):
        raise PreflightError(f"thinking-off probe failed: {tokens} completion tokens, content {content[:120]!r}, "
                             f"reasoning {len(reasoning)} chars")
    return tokens


def _completions_url(base_url: str) -> str:
    return base_url.rstrip("/") + "/chat/completions"


def _post(url: str, body: bytes, timeout: float) -> httpx.Response:
    """One POST of the JSON `body`, following redirects; the only header set here is the content type."""
    return httpx.post(url, content=body, headers={"Content-Type": "application/json"}, timeout=timeout,
                      follow_redirects=True)


def _attempt(url: str, body: bytes) -> str | Failure:
    """One request of a policy call: the completion text, or the failure to retry; raises PolicyError for a
    failure that repeating the request cannot overcome."""
    try:
        reply = _post(url, body, REQUEST_TIMEOUT_S)
    except RETRYABLE_TRANSPORT as exc:
        return Failure(f"{type(exc).__name__}: {exc}", exc)
    except REQUEST_ERRORS as exc:
        raise PolicyError(f"{type(exc).__name__}: {exc}") from exc
    status = reply.status_code
    if status == 429 or 500 <= status <= 599:
        return Failure(f"HTTP {status}: {reply.text[:200]}")
    if not 200 <= status <= 299:
        raise PolicyError(f"HTTP {status} from {url}: {reply.text[:500]}")
    try:
        content = reply.json()["choices"][0]["message"]["content"]
    except (ValueError, KeyError, IndexError, TypeError):
        content = None
    if not isinstance(content, str):
        return Failure(f"no completion text in the reply: {reply.text[:200]}")
    return content
