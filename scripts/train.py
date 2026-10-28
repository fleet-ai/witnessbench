"""Launch broadcast-GRPO training on miles for one model configuration.

    python scripts/train.py --model {27b,9b} --run-name NAME [--num-rollout N] [--seed N] [--tasks PATH]
                            [--trace-dir DIR] [--wandb-project NAME] [--dry-run]

The recipe is configs/rl_common.yaml plus configs/rl_<model>.yaml. Their keys are miles arguments with
underscores for dashes, and every value is passed on as written: `true` emits the bare flag, `false` omits the
flag. The placeholders ${MODEL_DIR}, ${CKPT_DIR} and ${DATA_DIR} are taken from the environment variables of the
same names (defaults: models/, checkpoints/ and data/ of this repository) and ${RUN_NAME} from --run-name.
Checkpoints are saved to $CKPT_DIR/NAME after every step; running a command again with the same run name resumes
from the last saved step.

--num-rollout N trains N steps instead of the recipe's 30 (give such a run its own run name). --seed N sets miles'
random seed instead of the recipe's 42 (a second training seed of the same setup). --tasks PATH trains on the
rows of PATH instead of $DATA_DIR/train.jsonl (for example data/train_r1.jsonl or data/train_r2.jsonl, see
docs/training.md): a step takes one group per row, so rollout_batch_size becomes the row count
and global_batch_size the row count times n_samples_per_prompt. --trace-dir DIR writes the trace of every training
episode to DIR/<game>/seed<seed>/. --wandb-project NAME logs to the Weights & Biases project NAME; the API key is read
from WANDB_API_KEY only. --dry-run prints the model type, the GPU count and the miles arguments, one flag per line,
and launches nothing; it does not need miles.

Before launching, the recipe keys and placeholders, the task rows (every game must be a public game or a training
game of `witness_env.TRAIN_R1_IDS` or `witness_env.TRAIN_R2_IDS`, and constructible), the trace directory and the
W&B key are checked; a problem found by these checks is reported in one line and the command exits with status 2,
as it does when miles cannot be imported. The launch itself is miles' execute_train, which stops every Ray and
SGLang process on the node, starts a Ray head and submits the training job.
"""
from __future__ import annotations

import argparse
import base64
import os
import re
import shlex
import string
import sys
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path

import yaml

from witness_agent.episode import load_tasks
from witness_env import GAME_IDS, TRAIN_R1_IDS, TRAIN_R2_IDS, make_game

REPO = Path(__file__).resolve().parents[1]
MODELS: tuple[str, ...] = ("27b", "9b")

# Every miles argument the launcher can emit, in emission order: the recipe keys plus LAUNCHER_KEYS.
RECIPE_KEYS: tuple[str, ...] = (
    "hf_checkpoint", "ref_load", "load", "save", "save_interval", "prompt_data", "input_key",
    "apply_chat_template", "rollout_shuffle", "num_rollout", "rollout_batch_size", "n_samples_per_prompt",
    "rollout_max_response_len", "rollout_temperature", "global_batch_size", "balance_data", "optimizer", "lr",
    "lr_decay_style", "weight_decay", "adam_beta1", "adam_beta2", "optimizer_cpu_offload",
    "overlap_cpu_optimizer_d2h_h2d", "use_precision_aware_optimizer", "advantage_estimator", "kl_loss_coef",
    "entropy_coef", "eps_clip", "eps_clip_high", "use_wandb", "wandb_project", "wandb_group",
    "disable_wandb_random_suffix", "tensor_model_parallel_size", "sequence_parallel",
    "pipeline_model_parallel_size", "context_parallel_size", "expert_model_parallel_size",
    "expert_tensor_parallel_size", "recompute_granularity", "recompute_method", "recompute_num_layers",
    "use_dynamic_batch_size", "max_tokens_per_gpu", "rollout_num_gpus_per_engine", "sglang_mem_fraction_static",
    "attention_dropout", "hidden_dropout", "accumulate_allreduce_grads_in_fp32", "attention_softmax_in_fp32",
    "attention_backend", "colocate", "actor_num_nodes", "actor_num_gpus_per_node", "num_gpus_per_node",
    "custom_generate_function_path", "custom_reward_post_process_path", "custom_config_path",
    "num_steps_per_rollout", "seed",
)
# Set only by launcher flags, never by the YAML files.
LAUNCHER_KEYS: tuple[str, ...] = (
    "use_wandb", "wandb_project", "wandb_group", "disable_wandb_random_suffix", "custom_config_path")
# The one key of a model file that is not a miles argument: miles' model-architecture entry.
MODEL_TYPE_KEY = "megatron_model_type"
# Path placeholders and their defaults relative to the repository.
PATH_DEFAULTS: tuple[tuple[str, str], ...] = (("MODEL_DIR", "models"), ("CKPT_DIR", "checkpoints"),
                                              ("DATA_DIR", "data"))


def load_recipe(model: str, config_dir: Path) -> tuple[str, dict[str, str]]:
    """(megatron model type, merged recipe) of rl_common.yaml and rl_<model>.yaml in `config_dir`.

    Raises ValueError for a file that is not valid YAML, a key outside RECIPE_KEYS (except `megatron_model_type` in
    the model file, which must set it), a key that only launcher flags may set, a key written twice or without a
    value in one file, or a key defined in both files.
    """
    common = _read_yaml(config_dir / "rl_common.yaml")
    specific = _read_yaml(config_dir / f"rl_{model}.yaml")
    for name, keys in (("rl_common.yaml", common), (f"rl_{model}.yaml", specific.keys() - {MODEL_TYPE_KEY})):
        reserved = sorted(k for k in keys if k in LAUNCHER_KEYS)
        if reserved:
            raise ValueError(f"{name}: keys set only by launcher flags: {reserved}")
        unknown = sorted(k for k in keys if k not in RECIPE_KEYS)
        if unknown:
            raise ValueError(f"{name}: unknown keys {unknown}")
    both = sorted(common.keys() & specific.keys())
    if both:
        raise ValueError(f"keys defined in both rl_common.yaml and rl_{model}.yaml: {both}")
    if MODEL_TYPE_KEY not in specific:
        raise ValueError(f"rl_{model}.yaml: missing key {MODEL_TYPE_KEY}")
    model_type = specific.pop(MODEL_TYPE_KEY)
    return model_type, {**common, **specific}


def build_argv(recipe: Mapping[str, str], *, paths: Mapping[str, str], run_name: str, trace_dir: str | None,
               wandb_project: str | None) -> list[list[str]]:
    """The miles arguments as one [flag, *values] group per emitted key, in RECIPE_KEYS order.

    Recipe values have their ${...} placeholders filled from `paths` and RUN_NAME (ValueError naming the key for
    an unknown one); "true" becomes the bare flag and "false" omits it. `wandb_project` adds the W&B flags (the
    run name is the group) and `trace_dir` adds `--custom-config-path` with an inline YAML document setting
    `witness_trace_dir`; these two values are passed as given. Raises ValueError for a recipe key outside
    RECIPE_KEYS.
    """
    unknown = sorted(k for k in recipe if k not in RECIPE_KEYS)
    if unknown:
        raise ValueError(f"unknown recipe keys {unknown}")
    values = _resolve(recipe, paths, run_name)
    if wandb_project is not None:
        values.update(use_wandb="true", wandb_project=wandb_project, wandb_group=run_name,
                      disable_wandb_random_suffix="true")
    if trace_dir is not None:
        document = yaml.safe_dump({"witness_trace_dir": trace_dir})
        values["custom_config_path"] = "base64:" + base64.b64encode(document.encode("utf-8")).decode("ascii")
    groups = []
    for key in RECIPE_KEYS:
        if values.get(key, "false") == "false":
            continue
        flag = "--" + key.replace("_", "-")
        groups.append([flag] if values[key] == "true" else [flag, values[key]])
    return groups


def check_task_rows(path: Path, rollout_batch_size: int) -> None:
    """Check the training rows before a launch: `load_tasks` reads them (the evaluator's loader: ValueError naming
    the line of a bad row, or a game on more than one row; blank lines are skipped, as miles' reader skips them),
    there is one row per group of a step (`rollout_batch_size` rows), every game is a public game or a training
    game of TRAIN_R1_IDS or TRAIN_R2_IDS (games added by a catalog module are never trained on: ValueError) and
    every game can be constructed (its level file loads; ONLY_RESET_LEVELS is not set). Raises what `make_game`
    raises."""
    tasks = load_tasks(path)
    if len(tasks) != rollout_batch_size:
        raise ValueError(f"{path}: {len(tasks)} rows, but a step takes rollout_batch_size = {rollout_batch_size}")
    for task in tasks:
        if task.game_id not in GAME_IDS + TRAIN_R1_IDS + TRAIN_R2_IDS:
            raise ValueError("training plays the games of witness_env only (games added by a catalog module are "
                             f"never trained on); {task.game_id} is not one of them")
        make_game(task.game_id)


def check_trace_dir(path: Path) -> None:
    """Create the trace directory if needed and prove it writable by creating and deleting one file in it;
    raises OSError otherwise."""
    os.makedirs(path, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path):
        pass


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command line above; returns the exit status (argparse exits with 2 on a usage error)."""
    args = _parser().parse_args(argv)
    try:
        model_type, recipe = load_recipe(args.model, REPO / "configs")
        if args.num_rollout is not None:
            recipe["num_rollout"] = str(args.num_rollout)
        if args.seed is not None:
            recipe["seed"] = str(args.seed)
        if args.tasks is not None:
            tasks_path = os.path.abspath(args.tasks)
            rows = len(load_tasks(Path(tasks_path)))
            recipe["prompt_data"] = tasks_path.replace("$", "$$")         # taken as written, not as a placeholder
            recipe["rollout_batch_size"] = str(rows)
            recipe["global_batch_size"] = str(rows * int(recipe["n_samples_per_prompt"]))
        num_gpus = int(recipe["num_gpus_per_node"])
        paths = {name: os.path.abspath(os.environ.get(name) or REPO / default) for name, default in PATH_DEFAULTS}
        trace_dir = None if args.trace_dir is None else os.path.abspath(args.trace_dir)
        groups = build_argv(recipe, paths=paths, run_name=args.run_name, trace_dir=trace_dir,
                            wandb_project=args.wandb_project)
        if args.dry_run:
            print(f"{MODEL_TYPE_KEY}={model_type}")
            print(f"num_gpus_per_node={num_gpus}")
            for group in groups:
                print(shlex.join(group))
            return 0
        prompt_data = Path(_resolve(recipe, paths, args.run_name)["prompt_data"])
        check_task_rows(prompt_data, int(recipe["rollout_batch_size"]))
        if trace_dir is not None:
            check_trace_dir(Path(trace_dir))
        if args.wandb_project is not None and not os.environ.get("WANDB_API_KEY"):
            raise ValueError("--wandb-project needs the W&B API key in the environment variable WANDB_API_KEY")
    except KeyError as exc:
        print(f"error: missing recipe key {exc}", file=sys.stderr)
        return 2
    except (ValueError, OSError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    try:
        from miles.utils.external_utils.command_utils import execute_train
    except ImportError as exc:
        print(f"error: cannot import miles ({exc}); run training inside the miles container (docs/training.md), "
              "or add --dry-run to print the miles arguments", file=sys.stderr)
        return 2
    execute_train(train_args=shlex.join(token for group in groups for token in group), num_gpus_per_node=num_gpus,
                  megatron_model_type=model_type)
    return 0


def _resolve(recipe: Mapping[str, str], paths: Mapping[str, str], run_name: str) -> dict[str, str]:
    """The recipe with the ${...} placeholders of every value filled from `paths` and RUN_NAME; raises
    ValueError naming the key of a value with an unknown placeholder."""
    substitutions = {**paths, "RUN_NAME": run_name}
    values = {}
    for key, raw in recipe.items():
        try:
            values[key] = string.Template(raw).substitute(substitutions)
        except KeyError as exc:
            raise ValueError(f"{key}: unknown placeholder {exc}") from None
    return values


def _read_yaml(path: Path) -> dict[str, str]:
    """A config file as a mapping of strings (YAML base loader: every scalar stays the text written). Raises
    ValueError unless the file is one YAML document that maps each key, written once, to one non-empty value."""
    with open(path, encoding="utf-8") as fh:
        try:
            node = yaml.compose(fh, Loader=yaml.BaseLoader)
        except yaml.YAMLError as exc:
            raise ValueError(f"{path.name}: not valid YAML: {' '.join(str(exc).split())}") from None
    if not isinstance(node, yaml.MappingNode) or not all(
            isinstance(k, yaml.ScalarNode) and isinstance(v, yaml.ScalarNode) for k, v in node.value):
        raise ValueError(f"{path.name}: expected a mapping of keys to single values")
    repeated = _repeated([k.value for k, _ in node.value])
    if repeated:
        raise ValueError(f"{path.name}: keys defined more than once: {repeated}")
    empty = sorted(k.value for k, v in node.value if v.value == "")
    if empty:
        raise ValueError(f"{path.name}: keys without a value: {empty}")
    return {k.value: v.value for k, v in node.value}


def _repeated(items: Sequence[str]) -> list[str]:
    """The items that occur more than once, sorted."""
    return sorted(item for item, count in Counter(items).items() if count > 1)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Launch broadcast-GRPO training on miles. Run it inside the miles "
                                                 "container (docs/training.md); --dry-run needs neither miles nor a "
                                                 "GPU.")
    parser.add_argument("--model", required=True, choices=MODELS,
                        help="the recipe to use: configs/rl_common.yaml plus configs/rl_<model>.yaml")
    parser.add_argument("--run-name", required=True, type=_run_name, metavar="NAME",
                        help="checkpoint directory name under $CKPT_DIR; an existing run name is resumed")
    parser.add_argument("--num-rollout", type=_positive_int, metavar="N",
                        help="number of training steps (default: 30, from configs/rl_common.yaml)")
    parser.add_argument("--seed", type=_positive_int, metavar="N",
                        help="miles' random seed (default: 42, from configs/rl_common.yaml)")
    parser.add_argument("--tasks", metavar="PATH", help="task file to train on (default: $DATA_DIR/train.jsonl)")
    parser.add_argument("--trace-dir", metavar="DIR", help="write the trace of every training episode under DIR")
    parser.add_argument("--wandb-project", metavar="NAME",
                        help="log to this Weights & Biases project (reads the key from WANDB_API_KEY)")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the model type, the GPU count and the miles arguments, and launch nothing")
    return parser


def _positive_int(text: str) -> int:
    if not re.fullmatch(r"[0-9]+", text) or int(text) < 1:
        raise argparse.ArgumentTypeError(f"expected an integer >= 1, got {text!r}")
    return int(text)


def _run_name(text: str) -> str:
    if text in ("", ".", "..") or "/" in text:
        raise argparse.ArgumentTypeError(f"expected a single path component, got {text!r}")
    return text


if __name__ == "__main__":
    sys.exit(main())
