#!/usr/bin/env bash
# Convert checkpoints between Hugging Face and Megatron torch_dist formats with the converter tools of
# the miles checkout in the training container. The miles model type and the GPU count are those of
# configs/rl_<model>.yaml; the model flags are what miles' model-argument tool prints for that type.
#   convert_checkpoint.sh to-torch-dist {27b|9b} HF_DIR OUT_DIR      base model, before training
#   convert_checkpoint.sh to-hf {27b|9b} ITER_DIR HF_DIR OUT_DIR     a saved iteration, e.g. <run>/iter_0000029
# to-hf runs miles' converter through scripts/convert_checkpoint.py, which supplies the Megatron arguments
# when ITER_DIR has no common.pt.
set -euo pipefail
MILES_DIR="${MILES_DIR:-/root/miles}"
MEGATRON_DIR="${MEGATRON_DIR:-/root/Megatron-LM}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="$MILES_DIR:$MEGATRON_DIR${PYTHONPATH:+:$PYTHONPATH}"
usage_text="usage: $0 to-torch-dist {27b|9b} HF_DIR OUT_DIR
       $0 to-hf {27b|9b} ITER_DIR HF_DIR OUT_DIR"
case "${1:-}" in
  -h | --help)
    echo "$usage_text"
    echo "to-torch-dist converts a base model to Megatron torch_dist format before training. to-hf converts a"
    echo "saved iteration (e.g. checkpoints/NAME/iter_0000029) back to Hugging Face format in OUT_DIR, which"
    echo "must not exist. Run it in the miles container (docs/training.md)."
    exit 0 ;;
esac
usage() { echo "$usage_text" >&2; exit 2; }
case "${1:-} $#" in "to-torch-dist 4" | "to-hf 5") ;; *) usage ;; esac
case "$2" in 27b | 9b) ;; *) usage ;; esac
settings="$(python3 "$SCRIPT_DIR/convert_checkpoint.py" model "$2")"
read -r model_type nproc <<< "$settings"
model_args_line="$(python3 "$MILES_DIR/miles/utils/external_utils/model_args_utils.py" "$model_type")"
read -r -a model_args <<< "$model_args_line"
case "$1" in
  to-torch-dist)
    torchrun --nproc-per-node "$nproc" "$MILES_DIR/tools/convert_hf_to_torch_dist.py" \
      "${model_args[@]}" --hf-checkpoint "$3" --save "$4"
    [ "$(cat "$4/latest_checkpointed_iteration.txt")" = release ] ||
      { echo "error: $4 does not hold a release checkpoint" >&2; exit 1; } ;;
  to-hf)
    python3 "$SCRIPT_DIR/convert_checkpoint.py" to-hf --model-flags="$model_args_line" \
      "$MILES_DIR/tools/convert_torch_dist_to_hf.py" "$3" "$4" "$5" ;;
esac
