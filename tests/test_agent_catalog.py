"""The game catalog (`witness_agent.catalog`): the public games, plus the games of a module named by the environment
variable WITNESS_GAME_CATALOG, for task rows, evaluation and training episodes; the training launcher refuses them.

The catalog module of these tests (the `write_catalog` fixture of conftest.py) is written to a temporary directory: it
lists the game id "zz01", a copy of tw01 under that id.
"""
from __future__ import annotations

import asyncio
import importlib.util
import json
import re
import sys
from pathlib import Path

import pytest

import witness_env.games.tw01  # noqa: F401  (imported by the catalog module of these tests)
from golden import STORE
from witness_agent import Task, catalog
from witness_agent.episode import load_tasks
from witness_agent.evaluate import run_validation
from witness_env import GAME_IDS, TRAIN_R1_IDS, TRAIN_R2_IDS
from witness_env import make_game as make_public

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("train_launcher_catalog", ROOT / "scripts" / "train.py")
train = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(train)


def plan(actions) -> str:
    return "<meta>x</meta><keep/><plan>" + ",".join(map(str, actions)) + "</plan>"


def test_without_the_catalog_variable_the_harness_knows_the_public_games_only(monkeypatch):
    for value in (None, ""):
        if value is None:
            monkeypatch.delenv(catalog.CATALOG_ENV, raising=False)
        else:
            monkeypatch.setenv(catalog.CATALOG_ENV, value)
        assert catalog.catalog_name() is None
        assert catalog.game_ids() == GAME_IDS + TRAIN_R1_IDS + TRAIN_R2_IDS
        with pytest.raises(ValueError, match=r"^game_id must be a known game id, got 'zz01' \(WITNESS_GAME_CATALOG is "
                                             r"unset: public games only\)$"):
            Task("zz01", 1, 1)
        with pytest.raises(KeyError) as ours:
            catalog.make_game("zz01")
        with pytest.raises(KeyError) as theirs:
            make_public("zz01")
        assert str(ours.value) == str(theirs.value)
        assert type(catalog.make_game("tw01")) is type(make_public("tw01"))


def test_a_catalog_module_adds_its_games_to_tasks_evaluation_and_rollouts(write_catalog, tmp_path):
    from witness_rl.rollout import run_rollout

    write_catalog()
    assert catalog.game_ids() == GAME_IDS + TRAIN_R1_IDS + TRAIN_R2_IDS + ("zz01",)
    task = Task("zz01", 1, 3)
    with pytest.raises(ValueError, match=r"^game_id must be a known game id, got 'zz02'$"):
        Task("zz02", 1, 3)
    rows = tmp_path / "rows.jsonl"
    rows.write_text(json.dumps({"prompt": "zz01", "metadata": {"game_id": "zz01", "max_levels": 1,
                                                               "max_reflection_calls": 3}}) + "\n")
    assert load_tasks(rows) == (task,)
    game = catalog.make_game("zz01")
    assert game.game_id == "zz01" and game.truth() == make_public("tw01").truth()

    first_level = STORE.solutions()["tw01"][0]                         # zz01's levels are tw01's
    out = tmp_path / "eval"
    out.mkdir()
    (outcome,) = run_validation([task], [0], lambda game_id, seed: lambda s, u: plan(first_level), out,
                                write_traces=False)
    assert (outcome.game_id, outcome.levels_beaten, outcome.stop_reason) == ("zz01", 1, "max_levels")

    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            return messages[0]["content"] + messages[1]["content"]

        def __call__(self, text, *, add_special_tokens):
            return {"input_ids": [1, 2, 3]}

    async def request(ids):
        return {"text": plan(first_level), "meta_info": {"finish_reason": {"type": "stop"}}}

    rollout = asyncio.run(run_rollout(task, 0, tokenizer=Tokenizer(), request=request))
    assert (rollout.abort, rollout.levels_completed, len(rollout.calls)) == (None, 1, 1)


def test_training_refuses_a_task_row_of_a_catalog_game(write_catalog, tmp_path, monkeypatch, capsys):
    write_catalog()
    shipped = (ROOT / "data" / "train.jsonl").read_text(encoding="utf-8").splitlines()
    rows = [json.dumps({"prompt": "zz01", "metadata": {"game_id": "zz01", "max_levels": 10,
                                                       "max_reflection_calls": 300}})] + shipped[1:]
    path = tmp_path / "train.jsonl"
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    assert len(load_tasks(path)) == 22                                   # the row is a valid task row
    message = ("training plays the games of witness_env only (games added by a catalog module are never trained "
               "on); zz01 is not one of them")
    with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
        train.check_task_rows(path, 22)
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setitem(sys.modules, "miles", None)
    assert train.main(["--model", "9b", "--run-name", "RUN"]) == 2
    captured = capsys.readouterr()
    assert (captured.out, captured.err) == ("", f"error: {message}\n")


def test_a_catalog_game_id_that_is_also_public_is_rejected(write_catalog):
    name = write_catalog(write_catalog.default.replace('PRIVATE_GAME_IDS = ("zz01",)',
                                                       'PRIVATE_GAME_IDS = ("zz01", "tw02")'))
    with pytest.raises(ValueError, match=f"^the game catalog {name} lists public game ids: tw02$"):
        catalog.game_ids()
    with pytest.raises(ValueError, match="lists public game ids"):
        Task("zz01", 1, 1)


def test_an_error_inside_the_catalog_module_is_raised_not_hidden(write_catalog, monkeypatch):
    write_catalog("raise RuntimeError('the catalog is broken')\n")
    with pytest.raises(RuntimeError, match="the catalog is broken"):
        catalog.game_ids()
    with pytest.raises(RuntimeError, match="the catalog is broken"):
        Task("tw01", 1, 1)
    monkeypatch.setenv(catalog.CATALOG_ENV, "zz_no_such_catalog_module")
    with pytest.raises(ModuleNotFoundError):
        catalog.game_ids()


def test_only_the_named_catalog_module_is_imported(write_catalog):
    name = write_catalog()
    before = set(sys.modules)
    catalog.game_ids()
    run = catalog.make_game("zz01")
    assert set(sys.modules) - before == {name}
    assert run.win_score == make_public("tw01").win_score
