"""The harness's prompts (observation version 2) must stay byte for byte what they are: the prompts of the
evaluation harness.

`agent/obs_v2_prompt_hashes.json.gz` was recorded by

    python tests/test_agent_obs_v2_golden.py --record

and holds, for every episode of `obs_support.episode_specs()`, a digest of every reflection call (system prompt,
user message, answer, parsed plan), of the steps, the stop reason, the levels completed and counters of the step
texts (`obs_support.digest`). The episodes: every public game played whole with seeds 0, 1 and 2; every level of
every public game played alone; and the three test-local games of `obs_support` that ring cells, light lamps and
carry a legend. Record again only when a prompt changes on purpose, and say why in the commit.

Two later changes of input, not of the harness: the entries multi_tw01_s0, _s1 and _s2 were recorded again when the
stored solutions of tw01 levels 6 and 9, which their scripted answers replay, were replaced by the shortest ones (13
and 21 actions instead of 15 and 23; see docs/training.md), and their first 35 calls are unchanged. When tw09 and
tw10 got levels 6-10 and tw12 got level 5, the entries multi_tw09_*, multi_tw10_* and multi_tw12_* were recorded again
(their prompts show the new level counts) and the entries of the eleven new levels were added. Every other entry is
byte-identical to the first recording.
"""
from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

if __name__ == "__main__":                                     # recording: make the repository importable
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

import obs_support
from golden import GOLDEN_DIR, TRAINING_GAMES
from witness_env import GAME_IDS

GOLDEN = GOLDEN_DIR / "agent" / "obs_v2_prompt_hashes.json.gz"
RECORDED_AT = "obs-v2"
SPECS = {spec.name: spec for spec in obs_support.episode_specs()}


def load() -> dict:
    data = json.loads(gzip.decompress(GOLDEN.read_bytes()))
    if set(data) != {"recorded_at", "episodes"} or data["recorded_at"] != RECORDED_AT:
        raise ValueError(f"{GOLDEN}: expected the keys recorded_at ({RECORDED_AT}) and episodes")
    for name, entry in data["episodes"].items():
        if set(entry) != {"calls", "steps", "stop", "levels", "counters"}:
            raise ValueError(f"{GOLDEN}: episode {name}: unexpected keys {sorted(entry)}")
    return data


def record() -> None:
    episodes = {}
    for name, spec in SPECS.items():
        result, _, snapshots, _ = obs_support.play_spec(spec)
        episodes[name] = obs_support.digest(result, snapshots)
    text = json.dumps({"recorded_at": RECORDED_AT, "episodes": episodes}, indent=0, sort_keys=True) + "\n"
    GOLDEN.write_bytes(gzip.compress(text.encode("utf-8"), mtime=0))


RECORDING = load()["episodes"] if GOLDEN.exists() else {}


@pytest.mark.parametrize("name", sorted(SPECS))
def test_the_prompts_match_the_recording(name):
    want = RECORDING[name]
    result, policy, snapshots, _ = obs_support.played(name)
    got = obs_support.digest(result, snapshots)
    first = next((i for i, (a, b) in enumerate(zip(want["calls"], got["calls"])) if a != b),
                 None if len(want["calls"]) == len(got["calls"]) else min(len(want["calls"]), len(got["calls"])))
    if first is not None:
        where = (f"level {result.calls[first].level} step {result.calls[first].level_step}"
                 if first < len(result.calls) else "after the last call")
        excerpt = policy.prompts[first][1][:3000] if first < len(policy.prompts) else ""
        pytest.fail(f"{name}: call {first} ({where}) differs from the recording; its user message now:\n{excerpt}")
    assert got == want


def test_the_recording_covers_every_public_game_and_level():
    assert set(RECORDING) == set(SPECS)
    for game_id in GAME_IDS:
        levels = obs_support.level_count(obs_support.game_class(game_id))
        assert {f"level_{game_id}_l{i + 1}" for i in range(levels)} <= set(RECORDING)
        assert {f"multi_{game_id}_s{s}" for s in (0, 1, 2)} <= set(RECORDING)


def _per_game(counter: str, prefix: str = "") -> dict[str, int]:
    """The recorded `counter` summed per game over the episodes whose name starts with `prefix`."""
    totals: dict[str, int] = {}
    for name, entry in RECORDING.items():
        if name.startswith(prefix):
            game_id = SPECS[name].game_id
            totals[game_id] = totals.get(game_id, 0) + entry["counters"][counter]
    return totals


def test_the_recording_exercises_every_branch_of_the_step_texts():
    """Every public game shows the step texts every game can show (a rejected submit at the start, cleared and
    unchanged RESETs, a blocked move, the plan cap, the random fallback, the attempts section); every game with a
    level of several starts shows a reverted and a switched start; every training game completes a level in its
    whole-game episodes; and every episode of a test-local game rings cells, lights and puts out lamps or carries a
    legend, as its game does. The counters rejected_path, action7 and no_visible_change are not checked: the harness
    no longer writes the first two texts, and no recorded step reaches the third."""
    for counter in ("rejected_at_start", "reset_cleared", "reset_at_start", "blocked", "plan_cap", "random_fallback",
                    "attempts"):
        totals = _per_game(counter)
        assert [g for g in GAME_IDS if totals.get(g, 0) < 1] == [], counter
    for counter in ("start_reverted", "start_switched"):
        totals = _per_game(counter)
        assert [g for g in sorted(obs_support.multi_start_games()) if totals.get(g, 0) < 1] == [], counter
    completed = _per_game("level_complete", "multi_")
    assert [g for g in TRAINING_GAMES if completed.get(g, 0) < 1] == []
    shown = {"rings": ("ringing_confirms",), "lamps": ("lamps_lit", "lamps_put_out", "ringing_confirms"),
             "strip": ("legend_snapshots",)}
    for name, entry in RECORDING.items():
        if name.startswith("made_"):
            assert all(entry["counters"][c] >= 1 for c in shown[SPECS[name].game_id]), name


if __name__ == "__main__":
    if sys.argv[1:] != ["--record"]:
        raise SystemExit("usage: python tests/test_agent_obs_v2_golden.py --record")
    record()
    print(f"recorded {len(SPECS)} episodes to {GOLDEN}")
