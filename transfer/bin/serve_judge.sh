#!/usr/bin/env bash
# serve_judge.sh JUDGE GPU PORT  — SGLang server for one of the two Failing-to-Falsify judges on a SPARE GPU
#   JUDGE in llama|coder:  llama = meta-llama/Llama-3.3-70B-Instruct  (rule-equivalence judge, greedy, 12 tokens)
#                          coder = Qwen/Qwen2.5-Coder-32B-Instruct     (Announce -> Python compatibility judge, T=0.2)
#   Both judged sequentially on ONE spare GPU is fine (judge load ~3.6k short calls per arm x rep x dialect):
#     serve llama -> launch_f2f.sh judge-guess ... ; touch STOP ; serve coder -> launch_f2f.sh judge-compat ...
#   Env: JUDGE_ROOT (STOP.$PORT + logs live there), MODELS_DIR (default transfer/work/models)
#   Weights: Llama-3.3-70B-Instruct is GATED on the HF hub (needs an accepted licence + HF token):
#     huggingface-cli download meta-llama/Llama-3.3-70B-Instruct --local-dir $MODELS_DIR/Llama-3.3-70B-Instruct
#     huggingface-cli download Qwen/Qwen2.5-Coder-32B-Instruct   --local-dir $MODELS_DIR/Qwen2.5-Coder-32B-Instruct
#   The judge MUST NOT share a server/process with any evaluated arm (separate GPU, separate port).
set -u
JUDGE=$1; GPU=$2; PORT=$3; ROOT=${JUDGE_ROOT:?set JUDGE_ROOT}; MODELS_DIR=${MODELS_DIR:-${TRANSFER_WORK:-$(cd "$(dirname "$0")/.." && pwd)/work}/models}; mkdir -p "$ROOT/logs"
case $JUDGE in
  llama) MP=$MODELS_DIR/Llama-3.3-70B-Instruct;    NAME=llama-3.3-70b-instruct;      CTX=8192  ;;   # paper wrapper: max_model_len 8192, bf16
  coder) MP=$MODELS_DIR/Qwen2.5-Coder-32B-Instruct; NAME=qwen2.5-coder-32b-instruct;  CTX=16384 ;;   # paper wrapper: max_model_len 16384, bf16
  *) echo "JUDGE must be llama|coder"; exit 2 ;;
esac
[ -f "$MP/config.json" ] || { echo "missing weights: $MP (see header for download commands)"; exit 3; }
while [ ! -e "$ROOT/STOP.$PORT" ]; do
  echo "$(date -u +%FT%TZ) LAUNCH judge $NAME gpu=$GPU port=$PORT ctx=$CTX"
  CUDA_VISIBLE_DEVICES=$GPU env SGL_FORCE_SHUTDOWN=1 python3 -m sglang.launch_server \
    --model-path "$MP" --served-model-name "$NAME" --host 127.0.0.1 --port "$PORT" --tp 1 --dtype bfloat16 \
    --context-length "$CTX" --sampling-defaults openai --max-running-requests 64 --mem-fraction-static 0.90 --log-level info
  echo "$(date -u +%FT%TZ) EXITED rc=$? $NAME port=$PORT; relaunch in 10 s (touch $ROOT/STOP.$PORT to stop)"
  sleep 10
done
