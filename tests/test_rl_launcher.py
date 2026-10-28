"""The training launcher `scripts/train.py` and the checkpoint converter `scripts/convert_checkpoint.sh` (with its
Python part `scripts/convert_checkpoint.py`).

The expected dry-run output of both model configurations is the golden fixture `rl/dryrun.json`. The launch
itself is replaced by a recording function, and the converter runs against a small stand-in of the miles
checkout (and of PyTorch's save and load), so nothing here needs a GPU or starts a training or server process.
"""
from __future__ import annotations

import base64
import hashlib
import importlib.util
import json
import os
import pickle
import re
import shlex
import shutil
import subprocess
import sys
import types
from collections.abc import Callable
from pathlib import Path

import pytest
import yaml

from golden import STORE, TRAINING_GAMES
from witness_env import TRAIN_R2_IDS

ROOT = Path(__file__).resolve().parent.parent
CONFIGS = ROOT / "configs"
TRAIN_ROWS = ROOT / "data" / "train.jsonl"
TRAIN_R1_ROWS = ROOT / "data" / "train_r1.jsonl"
TRAIN_R2_ROWS = ROOT / "data" / "train_r2.jsonl"
CONVERTER = ROOT / "scripts" / "convert_checkpoint.sh"
WANDB_FLAGS = ["--use-wandb", "--wandb-project P", "--wandb-group RUN", "--disable-wandb-random-suffix"]

_spec = importlib.util.spec_from_file_location("train_launcher", ROOT / "scripts" / "train.py")
train = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(train)

needs_bash = pytest.mark.skipif(shutil.which("bash") is None, reason="the converter is a bash script")


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch):
    """No launcher-relevant variable leaks in from the calling shell."""
    for name in ("MODEL_DIR", "CKPT_DIR", "DATA_DIR", "WANDB_API_KEY", "ONLY_RESET_LEVELS"):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def fixture_paths(monkeypatch):
    """The directories the golden dry-run output was written for."""
    monkeypatch.setenv("MODEL_DIR", "/MODELS")
    monkeypatch.setenv("CKPT_DIR", "/CKPT")
    monkeypatch.setenv("DATA_DIR", "/DATA")


@pytest.fixture
def no_miles(monkeypatch):
    """Any import of miles fails, so a launcher that returns normally stopped before launching."""
    monkeypatch.setitem(sys.modules, "miles", None)


def run(capsys, *argv: str) -> tuple[int, str, str]:
    code = train.main(list(argv))
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def config_digests() -> dict[str, str]:
    return {p.name: hashlib.md5(p.read_bytes()).hexdigest() for p in sorted(CONFIGS.glob("*.yaml"))}


# ---------------------------------------------------------------------------------------------------- dry run
@pytest.mark.parametrize("model", ["27b", "9b"])
def test_dry_run_prints_the_golden_miles_arguments_of_each_model(model, capsys, fixture_paths):
    assert run(capsys, "--model", model, "--run-name", "RUN", "--dry-run") == (0, STORE.rl_dryrun(model), "")


@pytest.mark.parametrize("model", ["27b", "9b"])
def test_num_rollout_changes_only_the_step_count_line_and_no_config_file(model, capsys, fixture_paths):
    before = config_digests()
    expected = STORE.rl_dryrun(model).splitlines()
    assert expected.count("--num-rollout 30") == 1
    code, out, err = run(capsys, "--model", model, "--run-name", "RUN", "--num-rollout", "2", "--dry-run")
    assert (code, err) == (0, "") and out.endswith("\n")
    assert out.splitlines() == ["--num-rollout 2" if line == "--num-rollout 30" else line for line in expected]
    assert run(capsys, "--model", model, "--run-name", "RUN", "--dry-run")[1] == STORE.rl_dryrun(model)
    assert config_digests() == before


def test_seed_changes_only_the_seed_line(capsys, fixture_paths):
    expected = STORE.rl_dryrun("27b").splitlines()
    assert expected.count("--seed 42") == 1
    code, out, err = run(capsys, "--model", "27b", "--run-name", "RUN", "--seed", "43", "--dry-run")
    assert (code, err) == (0, "")
    assert out.splitlines() == ["--seed 43" if line == "--seed 42" else line for line in expected]


@pytest.mark.parametrize("dry_run", [True, False], ids=["dry_run", "launch"])
@pytest.mark.parametrize("value", ["0", "-1", "x"])
def test_num_rollout_below_one_or_not_an_integer_is_a_usage_error(value, dry_run, capsys, no_miles):
    argv = ["--model", "27b", "--run-name", "RUN", "--num-rollout", value] + (["--dry-run"] if dry_run else [])
    with pytest.raises(SystemExit) as excinfo:
        train.main(argv)
    captured = capsys.readouterr()
    assert excinfo.value.code == 2
    assert captured.out == "" and "--num-rollout" in captured.err


# ----------------------------------------------------------------------------------------------------- recipe
def test_the_shipped_recipes_load_with_their_model_type():
    for model, model_type in (("27b", "qwen3.8-27B"), ("9b", "qwen3.5-9B")):
        loaded_type, recipe = train.load_recipe(model, CONFIGS)
        assert loaded_type == model_type
        assert set(recipe) <= set(train.RECIPE_KEYS) and not set(recipe) & set(train.LAUNCHER_KEYS)


def test_a_true_value_emits_the_bare_flag_and_false_omits_the_flag(tmp_path):
    common = (CONFIGS / "rl_common.yaml").read_text(encoding="utf-8").replace("colocate: true", "colocate: false")
    (tmp_path / "rl_common.yaml").write_text(common, encoding="utf-8")
    shutil.copyfile(CONFIGS / "rl_9b.yaml", tmp_path / "rl_9b.yaml")
    _, recipe = train.load_recipe("9b", tmp_path)
    groups = train.build_argv(recipe, paths={"MODEL_DIR": "/m", "CKPT_DIR": "/c", "DATA_DIR": "/d"}, run_name="RUN",
                              trace_dir=None, wandb_project=None)
    assert recipe["colocate"] == "false" and ["--colocate"] not in groups
    assert not any(group[0] == "--colocate" or "false" in group for group in groups)
    assert ["--balance-data"] in groups and ["--seed", "42"] in groups


@pytest.mark.parametrize("edited, edit, problem", [
    ("rl_common.yaml", lambda text: text + "lr_decy: x\n", "rl_common.yaml: unknown keys ['lr_decy']"),
    ("rl_27b.yaml", lambda text: text + "lr: 2e-6\n", "keys defined in both rl_common.yaml and rl_27b.yaml: ['lr']"),
    ("rl_common.yaml", lambda text: text + "lr: 2e-6\n", "rl_common.yaml: keys defined more than once: ['lr']"),
    ("rl_27b.yaml", lambda text: text.replace("max_tokens_per_gpu: 8192\n", "max_tokens_per_gpu:\n"),
     "rl_27b.yaml: keys without a value: ['max_tokens_per_gpu']"),
    ("rl_common.yaml", lambda text: text + "use_wandb: true\n",
     "rl_common.yaml: keys set only by launcher flags: ['use_wandb']"),
    ("rl_27b.yaml", lambda text: text + "custom_config_path: x.yaml\n",
     "rl_27b.yaml: keys set only by launcher flags: ['custom_config_path']"),
    ("rl_common.yaml", lambda text: text + "megatron_model_type: qwen3.8-27B\n",
     "rl_common.yaml: unknown keys ['megatron_model_type']"),
    ("rl_27b.yaml", lambda text: text.replace("megatron_model_type: qwen3.8-27B\n", ""),
     "rl_27b.yaml: missing key megatron_model_type"),
    ("rl_common.yaml", lambda text: text + "lr: [1e-6\n", "rl_common.yaml: not valid YAML: "),
    ("rl_27b.yaml", lambda text: text + "\tseed: 1\n", "rl_27b.yaml: not valid YAML: "),
], ids=["unknown_key", "key_in_both_files", "key_twice_in_one_file", "key_without_value", "launcher_key",
        "launcher_key_in_model_file", "model_type_in_common", "no_model_type", "unclosed_bracket", "tab_indent"])
def test_load_recipe_rejects_unknown_duplicated_empty_and_launcher_only_keys(tmp_path, edited, edit, problem):
    for name in ("rl_common.yaml", "rl_27b.yaml"):
        text = (CONFIGS / name).read_text(encoding="utf-8")
        (tmp_path / name).write_text(edit(text) if name == edited else text, encoding="utf-8")
    with pytest.raises(ValueError, match=re.escape(problem)) as excinfo:
        train.load_recipe("27b", tmp_path)
    assert "\n" not in str(excinfo.value)


def test_an_unknown_placeholder_is_an_error_naming_its_key():
    recipe = {"hf_checkpoint": "${MODEL_DRI}/Qwen3.5-9B"}
    with pytest.raises(ValueError, match=re.escape("hf_checkpoint: unknown placeholder 'MODEL_DRI'")):
        train.build_argv(recipe, paths={"MODEL_DIR": "/m", "CKPT_DIR": "/c", "DATA_DIR": "/d"}, run_name="RUN",
                         trace_dir=None, wandb_project=None)


@pytest.mark.parametrize("dry_run", [True, False], ids=["dry_run", "launch"])
@pytest.mark.parametrize("old, new, message", [
    ("hf_checkpoint: ${MODEL_DIR}/", "hf_checkpoint: ${MODEL_DRI}/", "hf_checkpoint: unknown placeholder 'MODEL_DRI'"),
    ("\nnum_gpus_per_node: 4\n", "\n", "missing recipe key 'num_gpus_per_node'"),
], ids=["unknown_placeholder", "missing_key"])
def test_a_bad_placeholder_or_a_missing_recipe_key_stops_the_launcher_with_one_line(old, new, message, dry_run,
                                                                                    capsys, no_miles, tmp_path,
                                                                                    monkeypatch):
    shutil.copytree(CONFIGS, tmp_path / "configs")
    model_file = tmp_path / "configs" / "rl_9b.yaml"
    text = model_file.read_text(encoding="utf-8")
    assert text.count(old) == 1
    model_file.write_text(text.replace(old, new), encoding="utf-8")
    monkeypatch.setattr(train, "REPO", tmp_path)
    argv = ["--model", "9b", "--run-name", "RUN"] + (["--dry-run"] if dry_run else [])
    assert run(capsys, *argv) == (2, "", f"error: {message}\n")


# --------------------------------------------------------------------------------------------- W&B and traces
def test_wandb_project_adds_the_wandb_flags_after_the_clip_flags_and_never_a_key_flag(capsys, fixture_paths):
    base = STORE.rl_dryrun("27b").splitlines()
    at = base.index("--eps-clip-high 0.28") + 1
    code, out, _ = run(capsys, "--model", "27b", "--run-name", "RUN", "--dry-run", "--wandb-project", "P")
    assert code == 0
    assert out.splitlines() == base[:at] + WANDB_FLAGS + base[at:]
    assert "--wandb-key" not in out
    assert "wandb" not in run(capsys, "--model", "27b", "--run-name", "RUN", "--dry-run")[1]


def test_wandb_project_without_the_api_key_in_the_environment_stops_before_launching(capsys, no_miles):
    code, out, err = run(capsys, "--model", "27b", "--run-name", "RUN", "--wandb-project", "P")
    assert (code, out) == (2, "")
    assert err.count("\n") == 1 and "WANDB_API_KEY" in err


def test_trace_dir_reaches_miles_as_an_inline_config_holding_the_absolute_directory(capsys, fixture_paths,
                                                                                     tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    code, out, _ = run(capsys, "--model", "9b", "--run-name", "RUN", "--dry-run", "--trace-dir", "rel/dir")
    lines = out.splitlines()
    config = [line for line in lines if line.startswith("--custom-config-path ")]
    assert code == 0 and len(config) == 1
    assert lines.index(config[0]) + 1 == lines.index("--num-steps-per-rollout 1")
    assert [line for line in lines if line not in config] == STORE.rl_dryrun("9b").splitlines()
    value = config[0].split(" ", 1)[1]
    assert value.startswith("base64:")
    document = base64.b64decode(value[len("base64:"):], validate=True).decode("utf-8")
    assert yaml.safe_load(document) == {"witness_trace_dir": os.path.abspath("rel/dir")}
    assert not (tmp_path / "rel").exists()                      # a dry run creates nothing


def test_an_unusable_trace_directory_stops_the_launch_with_one_line_naming_it(capsys, no_miles, tmp_path):
    (tmp_path / "f").write_text("", encoding="utf-8")
    target = tmp_path / "f" / "sub"
    code, out, err = run(capsys, "--model", "27b", "--run-name", "RUN", "--trace-dir", str(target))
    assert (code, out) == (2, "")
    assert err.count("\n") == 1 and str(target) in err


def test_the_trace_directory_is_created_empty_before_the_wandb_key_is_checked(capsys, no_miles, tmp_path):
    target = tmp_path / "new" / "dir"
    code, out, err = run(capsys, "--model", "27b", "--run-name", "RUN", "--trace-dir", str(target),
                         "--wandb-project", "P")
    assert (code, out) == (2, "")
    assert target.is_dir() and list(target.iterdir()) == []
    assert "WANDB_API_KEY" in err


def test_an_existing_trace_directory_is_accepted_so_a_resumed_run_can_write_to_it_again(tmp_path):
    target = tmp_path / "traces"
    train.check_trace_dir(target)
    (target / "tw01").mkdir()
    train.check_trace_dir(target)
    assert [p.name for p in target.iterdir()] == ["tw01"]


# -------------------------------------------------------------------------------------------------- task rows
def test_row_check_accepts_the_shipped_rows_whose_order_fixes_the_episode_seeds():
    train.check_task_rows(TRAIN_ROWS, 22)
    rows = [json.loads(line) for line in TRAIN_ROWS.read_text(encoding="utf-8").splitlines()]
    assert tuple(row["metadata"]["game_id"] for row in rows) == TRAINING_GAMES


def test_the_r1_rows_are_the_shipped_rows_with_the_revised_games_and_without_ft13():
    train.check_task_rows(TRAIN_R1_ROWS, 21)
    rows = [json.loads(line) for line in TRAIN_R1_ROWS.read_text(encoding="utf-8").splitlines()]
    shipped = [json.loads(line) for line in TRAIN_ROWS.read_text(encoding="utf-8").splitlines()]
    revised = {"tw02": "tw02r1", "tw03": "tw03r1", "tw10": "tw10r1", "ft03": "ft03r1", "ft06": "ft06r1"}
    expected = [{"prompt": revised.get(r["prompt"], r["prompt"]),
                 "metadata": {**r["metadata"], "game_id": revised.get(r["prompt"], r["prompt"])}}
                for r in shipped if r["prompt"] != "ft13"]
    assert rows == expected


def test_tasks_flag_trains_on_its_rows_with_one_group_per_row(capsys, fixture_paths):
    """--tasks changes three miles arguments of the golden dry run and no other."""
    code, out, err = run(capsys, "--model", "27b", "--run-name", "RUN", "--tasks", str(TRAIN_R1_ROWS), "--dry-run")
    assert (code, err) == (0, "")
    golden_lines, lines = STORE.rl_dryrun("27b").splitlines(), out.splitlines()
    assert [(a, b) for a, b in zip(golden_lines, lines) if a != b] == [
        ("--prompt-data /DATA/train.jsonl", f"--prompt-data {TRAIN_R1_ROWS}"),
        ("--rollout-batch-size 22", "--rollout-batch-size 21"),
        ("--global-batch-size 176", "--global-batch-size 168")]
    assert len(lines) == len(golden_lines)


def test_the_r2_rows_are_the_r1_rows_and_one_row_per_region_game(capsys, fixture_paths):
    train.check_task_rows(TRAIN_R2_ROWS, 21 + len(TRAIN_R2_IDS))
    rows = TRAIN_R2_ROWS.read_text(encoding="utf-8").splitlines()
    assert rows[:21] == TRAIN_R1_ROWS.read_text(encoding="utf-8").splitlines()
    assert tuple(json.loads(line)["metadata"]["game_id"] for line in rows[21:]) == TRAIN_R2_IDS
    code, out, err = run(capsys, "--model", "27b", "--run-name", "RUN", "--tasks", str(TRAIN_R2_ROWS), "--dry-run")
    assert (code, err) == (0, "")
    golden_lines, lines = STORE.rl_dryrun("27b").splitlines(), out.splitlines()
    assert [(a, b) for a, b in zip(golden_lines, lines) if a != b] == [
        ("--prompt-data /DATA/train.jsonl", f"--prompt-data {TRAIN_R2_ROWS}"),
        ("--rollout-batch-size 22", "--rollout-batch-size 26"),
        ("--global-batch-size 176", "--global-batch-size 208")]
    assert len(lines) == len(golden_lines)


def test_tasks_flag_reports_a_missing_file_in_one_line(capsys, tmp_path):
    code, out, err = run(capsys, "--model", "9b", "--run-name", "RUN", "--tasks", str(tmp_path / "none.jsonl"))
    assert (code, out) == (2, "") and err.count("\n") == 1 and "none.jsonl" in err


def write_rows(path: Path, edit: Callable[[list[str]], list[str]]) -> Path:
    """The lines of the shipped rows, changed by `edit`, written to `path`."""
    path.write_text("\n".join(edit(TRAIN_ROWS.read_text(encoding="utf-8").splitlines())) + "\n", encoding="utf-8")
    return path


@pytest.mark.parametrize("edit, problem", [
    (lambda rows: rows[:4] + ['{"prompt": "tw06", "metadata": {"game_id": "tw06"'] + rows[5:], "train.jsonl:5: "),
    (lambda rows: ["", *rows[:4], "{", *rows[5:]], "train.jsonl:6: "),
    (lambda rows: rows[:21], "21 rows, but a step takes rollout_batch_size = 22"),
    (lambda rows: rows + [rows[0].replace("tw01", "tw08")], "23 rows, but a step takes rollout_batch_size = 22"),
    (lambda rows: rows[:21] + rows[:1], "game ids on more than one row: tw01"),
], ids=["broken_line", "broken_line_after_a_blank_line", "missing_row", "extra_row", "repeated_game"])
def test_row_check_rejects_a_broken_line_a_wrong_row_count_and_a_repeated_game(tmp_path, edit, problem):
    path = write_rows(tmp_path / "train.jsonl", edit)
    with pytest.raises(ValueError, match=re.escape(problem)):
        train.check_task_rows(path, 22)


def test_row_check_skips_blank_lines_as_the_trainer_and_the_evaluator_do(tmp_path):
    path = write_rows(tmp_path / "train.jsonl", lambda rows: ["", *rows[:11], " \t", "", *rows[11:], ""])
    train.check_task_rows(path, 22)


@pytest.mark.parametrize("edit, problem", [
    (lambda rows: rows[:4] + ["{"] + rows[5:], "train.jsonl:5: "),
    (lambda rows: rows[:21], ": 21 rows, but a step takes rollout_batch_size = 22"),
    (None, "No such file or directory"),
], ids=["broken_line", "missing_row", "no_rows_file"])
def test_the_launch_checks_the_rows_under_data_dir_and_stops_with_one_line_naming_the_file(edit, problem, capsys,
                                                                                            no_miles, tmp_path,
                                                                                            monkeypatch):
    path = tmp_path / "train.jsonl"
    if edit is not None:
        write_rows(path, edit)
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    code, out, err = run(capsys, "--model", "9b", "--run-name", "RUN")
    assert (code, out) == (2, "")
    assert err.count("\n") == 1 and str(path) in err and problem in err


def test_row_check_constructs_every_game_and_so_refuses_reset_only_levels(monkeypatch, capsys, no_miles):
    monkeypatch.setenv("ONLY_RESET_LEVELS", "true")
    with pytest.raises(RuntimeError, match="ONLY_RESET_LEVELS"):
        train.check_task_rows(TRAIN_ROWS, 22)
    code, out, err = run(capsys, "--model", "9b", "--run-name", "RUN")
    assert (code, out) == (2, "")
    assert err.count("\n") == 1 and "ONLY_RESET_LEVELS" in err


# ------------------------------------------------------------------------------------------------- the launch
def test_paths_with_spaces_round_trip_through_shell_quoting():
    _, recipe = train.load_recipe("27b", CONFIGS)
    groups = train.build_argv(recipe, paths={"MODEL_DIR": "/m", "CKPT_DIR": "/a b", "DATA_DIR": "/d"},
                              run_name="RUN", trace_dir=None, wandb_project=None)
    assert shlex.split(" ".join(shlex.join(group) for group in groups)) == [t for group in groups for t in group]
    assert {group[0]: group[1:] for group in groups}["--load"] == ["/a b/RUN"]


def test_the_launch_hands_execute_train_one_shell_quoted_argument_string(capsys, monkeypatch):
    calls = []
    command_utils = types.ModuleType("miles.utils.external_utils.command_utils")
    command_utils.execute_train = lambda **kwargs: calls.append(kwargs)
    monkeypatch.setitem(sys.modules, "miles.utils.external_utils.command_utils", command_utils)
    monkeypatch.setenv("CKPT_DIR", "/a b")
    assert run(capsys, "--model", "27b", "--run-name", "RUN") == (0, "", "")
    _, recipe = train.load_recipe("27b", CONFIGS)
    expected = train.build_argv(recipe, paths={"MODEL_DIR": str(ROOT / "models"), "CKPT_DIR": "/a b",
                                               "DATA_DIR": str(ROOT / "data")},
                                run_name="RUN", trace_dir=None, wandb_project=None)
    assert len(calls) == 1 and set(calls[0]) == {"train_args", "num_gpus_per_node", "megatron_model_type"}
    assert shlex.split(calls[0]["train_args"]) == [t for group in expected for t in group]
    assert (calls[0]["num_gpus_per_node"], calls[0]["megatron_model_type"]) == (8, "qwen3.8-27B")


@pytest.mark.parametrize("name", ["", ".", "..", "run/one"], ids=["empty", "dot", "dot_dot", "two_components"])
def test_run_name_must_be_a_single_path_component(name, capsys):
    with pytest.raises(SystemExit) as excinfo:
        train.main(["--model", "27b", "--run-name", name, "--dry-run"])
    assert excinfo.value.code == 2 and capsys.readouterr().out == ""


# -------------------------------------------------------------------------------------------------- converter
# What the stand-in model-argument tool prints: the structure flags the conversion to Hugging Face format reads,
# among others.
MODEL_FLAGS = ("--disable-bias-linear --num-attention-heads 4 --num-query-groups 2 --kv-channels 16 --num-layers 2 "
               "--hidden-size 8 --normalization RMSNorm --vocab-size 32 --swiglu")
STRUCTURE = {"num_layers": 2, "hidden_size": 8, "num_attention_heads": 4, "num_query_groups": 2, "kv_channels": 16,
             "vocab_size": 32, "num_experts": None, "sglang_enable_ep_moe": False}


@needs_bash
@pytest.mark.parametrize("args", [[], ["to-torch-dist", "13b", "a", "b"], ["to-hf", "a", "b"],
                                  ["to-safetensors", "a", "b", "c"], ["to-hf", "a", "b", "c"],
                                  ["to-hf", "13b", "a", "b", "c"]],
                         ids=["no_arguments", "unknown_model", "missing_argument", "unknown_direction",
                              "to_hf_without_model", "to_hf_unknown_model"])
def test_converter_usage_errors_exit_with_status_2(args, tmp_path):
    result = subprocess.run(["bash", str(CONVERTER), *args], capture_output=True, text=True,
                            env={**os.environ, "MILES_DIR": str(tmp_path)})
    assert result.returncode == 2 and result.stderr.startswith("usage: ")


@pytest.fixture
def miles_checkout(tmp_path):
    """A stand-in miles checkout whose model-argument tool and converter tools log their arguments (the converter
    to Hugging Face format also logs the Megatron arguments of its input's common.pt and what its input directory
    holds), a Megatron-LM directory, a stand-in for the two PyTorch functions the conversion uses, and fake
    `torchrun` / `python3` commands first on PATH. Returns the environment for the converter and the log path."""
    miles, megatron, site = tmp_path / "miles", tmp_path / "megatron", tmp_path / "site"
    bin_dir, log = tmp_path / "bin", tmp_path / "log"
    for d in (miles / "miles" / "utils" / "external_utils", miles / "tools", megatron, site, bin_dir):
        d.mkdir(parents=True)
    record = ('import json, os, sys\n'
              'with open(os.environ["CONVERTER_LOG"], "a") as fh:\n'
              '    fh.write(json.dumps({"tool": os.path.basename(sys.argv[0]), "argv": sys.argv[1:],\n'
              '                         "pythonpath": os.environ["PYTHONPATH"]}) + "\\n")\n')
    (miles / "miles" / "utils" / "external_utils" / "model_args_utils.py").write_text(
        record + f'print(os.environ.get("STUB_MODEL_FLAGS", {MODEL_FLAGS!r}))\n', encoding="utf-8")
    (miles / "tools" / "convert_torch_dist_to_hf.py").write_text(
        record + 'import torch\n'
                 'input_dir = sys.argv[sys.argv.index("--input-dir") + 1]\n'
                 'args = torch.load(os.path.join(input_dir, "common.pt"), weights_only=False)["args"]\n'
                 'paths = {name: os.path.join(input_dir, name) for name in os.listdir(input_dir)}\n'
                 'held = {name: os.readlink(p) if os.path.islink(p) else "file" for name, p in paths.items()}\n'
                 'with open(os.environ["CONVERTER_LOG"], "a") as fh:\n'
                 '    fh.write(json.dumps({"common.pt": vars(args), "input": held}) + "\\n")\n', encoding="utf-8")
    (miles / "tools" / "convert_hf_to_torch_dist.py").write_text(
        record + 'save = sys.argv[sys.argv.index("--save") + 1]\n'
                 'os.makedirs(save, exist_ok=True)\n'
                 'with open(os.path.join(save, "latest_checkpointed_iteration.txt"), "w") as fh:\n'
                 '    fh.write(os.environ.get("STUB_ITERATION", "release"))\n', encoding="utf-8")
    (site / "torch.py").write_text(
        'import pickle\n\n\n'
        'def save(obj, path):\n'
        '    with open(path, "wb") as fh:\n'
        '        pickle.dump(obj, fh)\n\n\n'
        'def load(path, weights_only=True):\n'
        '    assert weights_only is False\n'
        '    with open(path, "rb") as fh:\n'
        '        return pickle.load(fh)\n', encoding="utf-8")
    (bin_dir / "torchrun").write_text(
        '#!/usr/bin/env bash\n'
        '[ "$1" = --nproc-per-node ] || exit 64\n'
        'printf \'{"tool": "torchrun", "nproc": "%s"}\\n\' "$2" >> "$CONVERTER_LOG"\n'
        'shift 2\n'
        'exec python3 "$@"\n', encoding="utf-8")
    (bin_dir / "python3").write_text(f'#!/bin/sh\nexec {shlex.quote(sys.executable)} "$@"\n', encoding="utf-8")
    for tool in ("torchrun", "python3"):
        (bin_dir / tool).chmod(0o755)
    env = {**os.environ, "MILES_DIR": str(miles), "MEGATRON_DIR": str(megatron),
           "PYTHONPATH": os.pathsep.join([str(site), str(ROOT)]),
           "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}", "CONVERTER_LOG": str(log)}
    for name in ("STUB_ITERATION", "STUB_MODEL_FLAGS"):
        env.pop(name, None)
    return env, log


def convert(env: dict, *args: str, converter: Path = CONVERTER) -> tuple[int, list[dict], str]:
    """Exit status, logged tool calls and stderr of one converter run."""
    result = subprocess.run(["bash", str(converter), *args], env=env, capture_output=True, text=True)
    log = Path(env["CONVERTER_LOG"])
    entries = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()] if log.exists() else []
    log.unlink(missing_ok=True)
    return result.returncode, entries, result.stderr


def iteration_dir(path: Path, *files: str) -> Path:
    """A stand-in iteration directory holding `files`."""
    path.mkdir(parents=True)
    for name in files:
        (path / name).write_bytes(b"x")
    return path


@needs_bash
def test_converter_calls_the_miles_tools_with_the_expected_arguments(miles_checkout, tmp_path):
    env, _ = miles_checkout
    pythonpath = f"{env['MILES_DIR']}:{env['MEGATRON_DIR']}:{env['PYTHONPATH']}"
    hf, out = str(tmp_path / "hf model"), str(tmp_path / "out dir")
    for model, model_type, nproc in (("9b", "qwen3.5-9B", "4"), ("27b", "qwen3.8-27B", "8")):
        assert convert(env, "to-torch-dist", model, hf, out) == (0, [
            {"tool": "model_args_utils.py", "argv": [model_type], "pythonpath": pythonpath},
            {"tool": "torchrun", "nproc": nproc},
            {"tool": "convert_hf_to_torch_dist.py", "argv": [*MODEL_FLAGS.split(), "--hf-checkpoint", hf,
                                                             "--save", out], "pythonpath": pythonpath}], "")
    code, entries, err = convert({**env, "STUB_ITERATION": "iter_0000001"}, "to-torch-dist", "9b", hf, out)
    assert code != 0 and err == f"error: {out} does not hold a release checkpoint\n"
    assert [e["tool"] for e in entries] == ["model_args_utils.py", "torchrun", "convert_hf_to_torch_dist.py"]
    # A saved iteration holds no common.pt: the converter reads a temporary directory that links the iteration's
    # files and adds a common.pt with the structure arguments of the model flags; the iteration is left unchanged.
    shutil.rmtree(out)
    iteration = iteration_dir(tmp_path / "run dir" / "iter_0000001", ".metadata", "__0_0.distcp", "metadata.json")
    for model, model_type in (("9b", "qwen3.5-9B"), ("27b", "qwen3.8-27B")):
        code, entries, err = convert(env, "to-hf", model, str(iteration), hf, out)
        assert (code, err) == (0, "")
        view = entries[1]["argv"][-1]
        assert entries == [
            {"tool": "model_args_utils.py", "argv": [model_type], "pythonpath": pythonpath},
            {"tool": "convert_torch_dist_to_hf.py", "argv": ["--origin-hf-dir", hf, "--output-dir", out,
                                                             "--input-dir", view], "pythonpath": pythonpath},
            {"common.pt": {**STRUCTURE, "hf_checkpoint": hf},
             "input": {name: str(iteration / name) for name in (".metadata", "__0_0.distcp", "metadata.json")}
             | {"common.pt": "file"}}]
        assert view != str(iteration) and not os.path.exists(view)
        assert sorted(p.name for p in iteration.iterdir()) == [".metadata", "__0_0.distcp", "metadata.json"]


@needs_bash
def test_converter_to_hf_reads_the_common_pt_of_an_iteration_that_has_one(miles_checkout, tmp_path):
    env, _ = miles_checkout
    iteration = iteration_dir(tmp_path / "iter_0000001", ".metadata")
    with open(iteration / "common.pt", "wb") as fh:
        pickle.dump({"args": types.SimpleNamespace(saved_by="training")}, fh)
    hf, out = str(tmp_path / "hf"), str(tmp_path / "out")
    code, entries, err = convert({**env, "STUB_MODEL_FLAGS": "--swiglu"}, "to-hf", "27b", str(iteration), hf, out)
    assert (code, err) == (0, "")
    assert entries[1]["argv"] == ["--origin-hf-dir", hf, "--output-dir", out, "--input-dir", str(iteration)]
    assert entries[2] == {"common.pt": {"saved_by": "training"}, "input": {".metadata": "file", "common.pt": "file"}}


@needs_bash
@pytest.mark.parametrize("existing_output, model_flags, message", [
    (True, MODEL_FLAGS, "error: {out} already exists\n"),
    (False, MODEL_FLAGS.replace("--kv-channels 16 ", ""),
     "error: the model flags need --kv-channels once, followed by an integer: {flags!r}\n"),
], ids=["output_exists", "structure_flag_missing"])
def test_converter_to_hf_stops_before_the_miles_converter_on_an_existing_output_or_a_missing_structure_flag(
        existing_output, model_flags, message, miles_checkout, tmp_path):
    env, _ = miles_checkout
    iteration = iteration_dir(tmp_path / "iter_0000001", ".metadata")
    out = tmp_path / "out"
    if existing_output:
        out.mkdir()
    code, entries, err = convert({**env, "STUB_MODEL_FLAGS": model_flags}, "to-hf", "9b", str(iteration),
                                 str(tmp_path / "hf"), str(out))
    assert (code, err) == (2, message.format(out=out, flags=model_flags))
    assert [e["tool"] for e in entries] == ["model_args_utils.py"]


@needs_bash
def test_converter_takes_the_model_type_and_the_gpu_count_from_the_model_config(miles_checkout, tmp_path):
    env, _ = miles_checkout
    repo = tmp_path / "repo"
    shutil.copytree(ROOT / "scripts", repo / "scripts")
    shutil.copytree(CONFIGS, repo / "configs")
    model_file = repo / "configs" / "rl_9b.yaml"
    text = model_file.read_text(encoding="utf-8")
    model_file.write_text(text.replace("megatron_model_type: qwen3.5-9B\n", "megatron_model_type: other-9B\n")
                          .replace("num_gpus_per_node: 4\n", "num_gpus_per_node: 2\n"), encoding="utf-8")
    code, entries, _ = convert(env, "to-torch-dist", "9b", str(tmp_path / "hf"), str(tmp_path / "out"),
                               converter=repo / "scripts" / "convert_checkpoint.sh")
    assert code == 0 and [e.get("argv", e.get("nproc")) for e in entries[:2]] == [["other-9B"], "2"]
    code, entries, _ = convert(env, "to-hf", "9b", str(iteration_dir(tmp_path / "iter", ".metadata")),
                               str(tmp_path / "hf"), str(tmp_path / "out2"),
                               converter=repo / "scripts" / "convert_checkpoint.sh")
    assert code == 0 and entries[0]["argv"] == ["other-9B"]


@needs_bash
@pytest.mark.parametrize("direction", ["to-torch-dist", "to-hf"])
def test_converter_stops_before_torchrun_when_the_model_argument_tool_fails(direction, miles_checkout, tmp_path):
    env, _ = miles_checkout
    tool = Path(env["MILES_DIR"]) / "miles" / "utils" / "external_utils" / "model_args_utils.py"
    tool.write_text('import sys\nprint("--num-layers 2")\nsys.exit(1)\n', encoding="utf-8")
    paths = [str(tmp_path / "hf"), str(tmp_path / "out")]
    if direction == "to-hf":
        paths.insert(0, str(iteration_dir(tmp_path / "iter", ".metadata")))
    code, entries, _ = convert(env, direction, "27b", *paths)
    assert code != 0 and entries == []
