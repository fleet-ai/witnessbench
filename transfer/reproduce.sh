#!/usr/bin/env bash
# reproduce.sh — one command for the paper's out-of-domain transfer table (CipherBank, PhysGym, WILT, Failing-to-Falsify):
#   set up the four benchmarks at their pinned commits + patches, download both models, evaluate the base model and the
#   RL checkpoint with the paper's settings, and print the four summaries (base vs RL, paired 95% bootstrap intervals).
#
#   GPUS=0,1,2,3,4 OPENROUTER_API_KEY=sk-or-... bash transfer/reproduce.sh [OUT]      (OUT default: transfer/work/results)
#
# Env: GPUS (default 0; 5 GPUs run all lanes of a model in parallel), BASE_MODEL, RL_MODEL (HF ids or local dirs).
# OPENROUTER_API_KEY pays for the two API judges (PhysGym: google/gemini-2.5-flash; F2F: llama-3.3-70b-instruct).
set -euo pipefail
T=$(cd "$(dirname "$0")" && pwd)
OUT=${1:-${TRANSFER_WORK:-$T/work}/results}
BASE_MODEL=${BASE_MODEL:-Qwen/Qwen3.8-27B}
RL_MODEL=${RL_MODEL:-Guanghan-Ning/WitnessGym-Qwen3.8-27B-RL-broadcast-step20}
bash "$T/setup.sh"
rc=0
bash "$T/run_all.sh" run base   "$BASE_MODEL" "$OUT" || rc=1
bash "$T/run_all.sh" run step20 "$RL_MODEL"   "$OUT" || rc=1
bash "$T/run_all.sh" summarize "$OUT"
[ $rc = 0 ] || echo "WARNING: some parts failed (see $OUT/logs/status-*.txt); the summaries above may be incomplete"
exit $rc
