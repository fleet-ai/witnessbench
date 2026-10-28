"""The Python part of scripts/convert_checkpoint.sh, which runs it with miles and Megatron-LM on PYTHONPATH.

    convert_checkpoint.py model {27b,9b}
    convert_checkpoint.py to-hf --model-flags=MODEL_FLAGS CONVERTER ITER_DIR HF_DIR OUT_DIR

`model` prints the miles model type and the GPU count of the model's recipe (`megatron_model_type` and
`num_gpus_per_node` of configs/rl_<model>.yaml, read by scripts/train.py's `load_recipe`), separated by a space.

`to-hf` converts the torch_dist checkpoint ITER_DIR to Hugging Face format in OUT_DIR, which must not exist, by
running CONVERTER, miles' unmodified `tools/convert_torch_dist_to_hf.py`, with `--origin-hf-dir HF_DIR` and its
defaults otherwise. That converter reads the Megatron arguments from `common.pt` in its input directory, which the
iteration directories that training saves in the miles container do not hold. For such a directory the converter
reads a temporary directory instead that links every file of ITER_DIR and adds a `common.pt` holding the arguments
built from MODEL_FLAGS, the flag line that miles' model-argument tool prints for the model type (the model flags
training ran with). The converter reads only the model structure from these arguments: the sizes in
STRUCTURE_FLAGS, no experts, and HF_DIR as the origin checkpoint.
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import subprocess
import sys
import tempfile
import types
from collections.abc import Sequence
from pathlib import Path

HERE = Path(__file__).resolve().parent
# The Megatron arguments that miles' converter reads for a dense model, as flags of miles' model-argument tool
# (underscores for dashes); each takes an integer.
STRUCTURE_FLAGS: tuple[str, ...] = ("num_layers", "hidden_size", "num_attention_heads", "num_query_groups",
                                    "kv_channels", "vocab_size")


def model_settings(model: str) -> tuple[str, int]:
    """(miles model type, GPU count) of the recipe of `model`, loaded by scripts/train.py's `load_recipe`."""
    module_spec = importlib.util.spec_from_file_location("witness_train_launcher", HERE / "train.py")
    train = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(train)
    model_type, recipe = train.load_recipe(model, train.REPO / "configs")
    return model_type, int(recipe["num_gpus_per_node"])


def structure_args(model_flags: str, origin_hf_dir: str) -> types.SimpleNamespace:
    """The Megatron arguments miles' converter reads, built from the flag line `model_flags`: STRUCTURE_FLAGS (each
    must occur once, followed by an integer; ValueError otherwise), no experts, and `origin_hf_dir` as the origin
    checkpoint."""
    tokens = model_flags.split()
    values = {}
    for name in STRUCTURE_FLAGS:
        flag = "--" + name.replace("_", "-")
        at = [i for i, token in enumerate(tokens) if token == flag]
        value = tokens[at[0] + 1] if len(at) == 1 and at[0] + 1 < len(tokens) else ""
        if not value.isdigit():
            raise ValueError(f"the model flags need {flag} once, followed by an integer: {model_flags!r}")
        values[name] = int(value)
    return types.SimpleNamespace(**values, num_experts=None, sglang_enable_ep_moe=False,
                                 hf_checkpoint=origin_hf_dir)


def to_hf(converter: str, model_flags: str, input_dir: str, origin_hf_dir: str, output_dir: str) -> int:
    """Run the miles converter `converter` on `input_dir` (see the module docstring); returns its exit status.
    Raises ValueError, before running it, if `output_dir` exists or `input_dir` has no `common.pt` and the model
    flags lack a structure flag."""
    if os.path.exists(output_dir):
        raise ValueError(f"{output_dir} already exists")
    command = [sys.executable, converter, "--origin-hf-dir", origin_hf_dir, "--output-dir", output_dir]
    if os.path.exists(os.path.join(input_dir, "common.pt")):
        return subprocess.run([*command, "--input-dir", input_dir], check=False).returncode
    megatron_args = structure_args(model_flags, origin_hf_dir)
    import torch  # noqa: PLC0415 (the container's PyTorch; only this conversion needs it)
    with tempfile.TemporaryDirectory() as view:
        for name in os.listdir(input_dir):
            os.symlink(os.path.abspath(os.path.join(input_dir, name)), os.path.join(view, name))
        torch.save({"args": megatron_args}, os.path.join(view, "common.pt"))
        return subprocess.run([*command, "--input-dir", view], check=False).returncode


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command line above; returns the exit status (2 with a one-line error for a ValueError)."""
    args = _parser().parse_args(argv)
    try:
        if args.command == "model":
            print(*model_settings(args.model))
            return 0
        return to_hf(args.converter, args.model_flags, args.iter_dir, args.hf_dir, args.out_dir)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="The Python part of scripts/convert_checkpoint.sh, which sets "
                                                 "PYTHONPATH and the model flags; run that script instead.")
    commands = parser.add_subparsers(dest="command", required=True)
    model = commands.add_parser("model", help="print the miles model type and the GPU count of a recipe")
    model.add_argument("model", choices=("27b", "9b"), help="the model configuration: configs/rl_<model>.yaml")
    hf = commands.add_parser("to-hf", help="convert a torch_dist iteration directory to Hugging Face format")
    hf.add_argument("--model-flags", required=True, metavar="FLAGS",
                    help="the flag line that miles' model-argument tool prints for the model type; write it as "
                         "--model-flags='--num-layers ...', because it starts with --")
    for name, text in (("converter", "miles' tools/convert_torch_dist_to_hf.py"),
                       ("iter_dir", "a saved iteration, e.g. checkpoints/NAME/iter_0000029"),
                       ("hf_dir", "the Hugging Face directory of the base model"),
                       ("out_dir", "the output directory; it must not exist")):
        hf.add_argument(name, help=text)
    return parser


if __name__ == "__main__":
    sys.exit(main())
