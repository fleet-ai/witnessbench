"""Trace files of one episode, for inspection: every reflection call, every action and a summary."""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from .episode import CallRecord, EpisodeResult


def write_trace(result: EpisodeResult, directory: Path) -> None:
    """Write `calls.jsonl`, `steps.jsonl` and `summary.json` of `result` into `directory` (created if needed;
    existing files are overwritten). Aborted partial results are written the same way.

    `calls.jsonl` has one JSON object per reflection call with the keys index, level, level_step, system, user,
    response and plan (a list of action ids, or null for an answer without a plan or cut off at the output limit),
    and meta (`Completion.meta()`) for a call answered with a `Completion`; `steps.jsonl` one per action with the
    keys level, step, action, actor and noop_reset. `summary.json` holds game_id, seed, max_levels,
    levels_completed, reflection_calls, steps, random_fallback_steps (the actions of the random fallback),
    stop_reason, resets (a {"level", "step"} object per executed RESET) and observation_version; when some call has
    meta, also `completions` (`completion_totals`). All files are written by `write_utf8`.
    """
    directory.mkdir(parents=True, exist_ok=True)
    summary: dict[str, object] = {
        "game_id": result.game_id,
        "seed": result.seed,
        "max_levels": result.max_levels,
        "levels_completed": result.levels_completed,
        "reflection_calls": len(result.calls),
        "steps": len(result.steps),
        "random_fallback_steps": sum(1 for s in result.steps if s.actor == "random"),
        "stop_reason": result.stop_reason,
        "resets": [asdict(r) for r in result.resets],
        "observation_version": result.observation_version,
    }
    if any(c.meta is not None for c in result.calls):
        summary["completions"] = completion_totals(result)
    write_utf8(directory / "calls.jsonl", "".join(_line(_call(c)) for c in result.calls))
    write_utf8(directory / "steps.jsonl", "".join(_line(asdict(s)) for s in result.steps))
    write_utf8(directory / "summary.json", json.dumps(summary, ensure_ascii=False, indent=2) + "\n")


def completion_totals(result: EpisodeResult) -> dict[str, object]:
    """Totals over the calls of `result` answered with a `Completion`: calls, truncated (cut off at the output
    limit), retried (answers that took more than one request), extra_requests (requests beyond the first), the token
    counts and cost summed over the calls that report them, and the distinct served models and providers."""
    metas = [c.meta for c in result.calls if c.meta is not None]

    def total(key: str) -> int | float | None:
        values = [m["usage"].get(key) for m in metas if m["usage"].get(key) is not None]
        return sum(values) if values else None

    return {"calls": len(metas), "truncated": sum(1 for m in metas if m["truncated"]),
            "retried": sum(1 for m in metas if m["attempts"] > 1),
            "extra_requests": sum(m["attempts"] - 1 for m in metas),
            "prompt_tokens": total("prompt_tokens"), "completion_tokens": total("completion_tokens"),
            "reasoning_tokens": total("reasoning_tokens"), "cached_tokens": total("cached_tokens"),
            "cost": total("cost"),
            "served_models": sorted({m["served_model"] for m in metas if m["served_model"]}),
            "served_providers": sorted({m["served_provider"] for m in metas if m["served_provider"]})}


def write_utf8(path: Path, text: str) -> None:
    """Write `text` to `path` as UTF-8, non-ASCII text kept as is. A lone surrogate, which UTF-8 cannot encode (a
    reply can carry one as an unpaired JSON escape), is written as its escape (e.g. `\\ud83d`), so JSON text
    written this way reads back as the same text."""
    path.write_bytes(text.encode("utf-8", errors="backslashreplace"))


def _call(call: CallRecord) -> dict[str, object]:
    """One line of `calls.jsonl`: the record, without the meta key when the call has none (the calls of a plain-text
    policy)."""
    obj = asdict(call)
    if call.meta is None:
        del obj["meta"]
    return obj


def _line(obj: dict[str, object]) -> str:
    return json.dumps(obj, ensure_ascii=False) + "\n"
