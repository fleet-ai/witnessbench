#!/usr/bin/env bash
# run_all.sh — run the four out-of-domain transfer evaluations of the paper for ONE model, with the settings of the paper:
#   CipherBank          thinking ON,  max_tokens 32768, 2 repetitions
#   PhysGym (L4)        thinking ON,  max_tokens 32000, context 65536, 3 repetitions, in-loop judge google/gemini-2.5-flash
#   WILT                thinking OFF, 5 repetitions
#   Failing-to-Falsify  thinking OFF, 5 repetitions; judges meta-llama/llama-3.3-70b-instruct (OpenRouter, CoreWeave fp16)
#                       and a local Qwen2.5-Coder-32B-Instruct
# Run it once for the base model and once for the RL checkpoint with the same OUT, then `run_all.sh summarize OUT`.
#
#   GPUS=0,1,2,3,4 OPENROUTER_API_KEY=... bash transfer/run_all.sh run  ARM MODEL OUT
#       ARM   = base | step20 | <any name>  (base -> served as qwen3.8-27b-base, step20 -> witnessgym-qwen3.8-27b-rl-broadcast-step20)
#       MODEL = a Hugging Face repo id (downloaded to transfer/work/models/) or a local HF checkpoint dir
#   bash transfer/run_all.sh summarize OUT
#
# Work is split into five lanes: (1) WILT + F2F + F2F judges, (2) CipherBank, (3-5) PhysGym repetition 1/2/3.
# Lane i runs on GPU GPUS[i mod n]; lanes that share a GPU run one after another. 5 GPUs = everything in parallel
# (wall clock ~ the slowest lane, PhysGym; see README for times). Without OPENROUTER_API_KEY, PhysGym and the F2F
# guess judge are skipped (their numbers then cannot be compared with the paper's).
# Run as root (or set SANDBOX_USER=) : WILT and F2F execute model-written Python as user `nobody`.
set -uo pipefail
T=$(cd "$(dirname "$0")" && pwd); B=$T/bin
W=${TRANSFER_WORK:-$T/work}; PY=$W/venv/bin/python
CMD=${1:-}; shift || true
log(){ echo "=== $(date -u +%FT%TZ) $*"; }
served_name(){ case $1 in base) echo qwen3.8-27b-base;; step20) echo witnessgym-qwen3.8-27b-rl-broadcast-step20;; *) echo "$1";; esac; }

if [ "$CMD" = summarize ]; then
  O=${1:?usage: run_all.sh summarize OUT}
  bash "$B/launch_cipherbank.sh" summarize --root "$O/cipherbank"
  bash "$B/launch_wilt.sh" summarize --root "$O/wilt"
  bash "$B/launch_f2f.sh" summarize --root "$O/f2f"
  tags=$(cd "$O/physgym" 2>/dev/null && ls -d */ 2>/dev/null | tr -d / | grep -- '-r[0-9]$' | tr '\n' ' ')
  [ -n "$tags" ] && ROOT="$O/physgym" STEP0=qwen3.8-27b-base STEP10=witnessgym-qwen3.8-27b-rl-broadcast-step20 \
    bash "$B/physgym-merge-summary.sh" merged $tags
  exit 0
fi
[ "$CMD" = run ] || { sed -n 2,22p "$0"; exit 2; }
ARM=${1:?ARM}; MODEL=${2:?MODEL}; O=${3:?OUT}; mkdir -p "$O/logs" "$O/servers"; O=$(cd "$O" && pwd)
NAME=$(served_name "$ARM"); SANDBOX_USER=${SANDBOX_USER-nobody}
IFS=, read -r -a G <<< "${GPUS:-0}"
if [ -d "$MODEL" ]; then HF=$(cd "$MODEL" && pwd)
else HF=$W/models/${MODEL##*/}; [ -f "$HF/model.safetensors.index.json" ] || { log "download $MODEL -> $HF"; hf download "$MODEL" --local-dir "$HF" || exit 3; }; fi
JUDGE=$W/models/Qwen2.5-Coder-32B-Instruct
[ -f "$JUDGE/config.json" ] || { log "download judge -> $JUDGE"; hf download Qwen/Qwen2.5-Coder-32B-Instruct --local-dir "$JUDGE" || exit 3; }
STATUS=$O/logs/status-$ARM.txt; : > "$STATUS"
if [ -n "${OPENROUTER_API_KEY:-}" ]; then ( umask 077; printf %s "$OPENROUTER_API_KEY" > "$O/servers/.openrouter.key" ); KEY=1; else KEY=0; log "no OPENROUTER_API_KEY: PhysGym and the F2F guess judge are skipped"; fi

# policy server: one TP=1 SGLang server, identical flags for every arm (thinking is the only switch)
serve(){ # GPU PORT on|off
  local gpu=$1 port=$2 ctk='{"enable_thinking": false}'; [ "$3" = on ] && ctk='{"enable_thinking": true}'
  CUDA_VISIBLE_DEVICES=$gpu setsid python3 -m sglang.launch_server --model-path "$HF" --served-model-name "$NAME" \
    --host 127.0.0.1 --port "$port" --tp 1 --context-length 262144 --sampling-defaults openai --default-chat-template-kwargs "$ctk" \
    --reasoning-parser qwen3 --tool-call-parser qwen3_coder --max-running-requests 32 --enable-cache-report --mem-fraction-static 0.85 \
    --attention-backend trtllm_mha --page-size 64 --mamba-radix-cache-strategy extra_buffer --max-mamba-cache-size 160 \
    > "$O/servers/server-$port.log" 2>&1 < /dev/null &
  echo $! > "$O/servers/pid-$port"
}
serve_judge(){ # GPU PORT -> F2F compatibility judge
  CUDA_VISIBLE_DEVICES=$1 setsid python3 -m sglang.launch_server --model-path "$JUDGE" --served-model-name qwen2.5-coder-32b-instruct \
    --host 127.0.0.1 --port "$2" --tp 1 --dtype bfloat16 --context-length 16384 --sampling-defaults openai \
    --max-running-requests 64 --mem-fraction-static 0.90 > "$O/servers/server-$2.log" 2>&1 < /dev/null &
  echo $! > "$O/servers/pid-$2"
}
healthy(){ for _ in $(seq 1 120); do [ "$(curl -s -m 5 -o /dev/null -w '%{http_code}' "http://127.0.0.1:$1/health")" = 200 ] && return 0; sleep 10; done; log "server :$1 not healthy"; return 1; }
stop(){ local p; p=$(cat "$O/servers/pid-$1" 2>/dev/null) || return 0; kill -TERM -- "-$p" 2>/dev/null; sleep 20; kill -KILL -- "-$p" 2>/dev/null; }
step(){ local label=$1; shift; "$@" > "$O/logs/$label-$ARM.log" 2>&1; local rc=$?; echo "$label exit $rc" >> "$STATUS"; log "$label exit $rc"; }

lane_wilt_f2f(){ # GPU PORTBASE
  local g=$1 p=$2; serve "$g" "$p" off; healthy "$p" || { echo "wilt_f2f server exit 1" >> "$STATUS"; return; }
  local F2F="REPO=$W/repos/llm-confirmation-bias/wason_experiments PY=$PY"
  step wilt_play env -u OPENROUTER_API_KEY REPO="$W/repos/wilt" PY="$PY" bash "$B/launch_wilt.sh" play --root "$O/wilt" --think off \
    --ports "$ARM=$p" --arms "$ARM" --reps 1-5 --multi 25 --split full ${SANDBOX_USER:+--sandbox-user "$SANDBOX_USER"} &
  local a=$!
  step f2f_play env -u OPENROUTER_API_KEY $F2F bash "$B/launch_f2f.sh" play --root "$O/f2f" --think off --ports "$ARM=$p" --arms "$ARM" --reps 1-5 --par 24 &
  wait $! $a; stop "$p"
  if [ $KEY = 1 ]; then
    step f2f_judge_guess env -u OPENROUTER_API_KEY $F2F F2F_JUDGE_API_KEY_FILE="$O/servers/.openrouter.key" \
      F2F_JUDGE_PROVIDER='{"order":["CoreWeave"],"allow_fallbacks":false}' bash "$B/launch_f2f.sh" judge-guess --root "$O/f2f" --think off \
      --judge 'https://openrouter.ai/api/v1#meta-llama/llama-3.3-70b-instruct' --arms "$ARM" --reps 1-5 --par 6 &
    local jg=$!
  fi
  serve_judge "$g" $((p+1))
  if healthy $((p+1)); then
    step f2f_judge_compat env -u OPENROUTER_API_KEY $F2F bash "$B/launch_f2f.sh" judge-compat --root "$O/f2f" --think off \
      --judge "http://127.0.0.1:$((p+1))/v1#qwen2.5-coder-32b-instruct" --arms "$ARM" --reps 1-5 --par 6 ${SANDBOX_USER:+--sandbox-user "$SANDBOX_USER"}
  else echo "f2f_judge_compat exit 1" >> "$STATUS"; fi
  stop $((p+1)); [ -n "${jg:-}" ] && wait "$jg"
  step f2f_analyze env -u OPENROUTER_API_KEY $F2F bash "$B/launch_f2f.sh" analyze --root "$O/f2f" --think off --arms "$ARM" --reps 1-5
}
lane_cipherbank(){ # GPU PORT
  serve "$1" "$2" on; healthy "$2" || { echo "cipherbank server exit 1" >> "$STATUS"; return; }
  step cipherbank_play env -u OPENROUTER_API_KEY REPO="$W/repos/CipherBank" PY="$PY" bash "$B/launch_cipherbank.sh" play \
    --root "$O/cipherbank" --think on --ports "$ARM=$2" --arms "$ARM" --reps 1-2 --par 32
  stop "$2"
}
lane_physgym(){ # GPU PORT REP   (physgym-eval.sh starts its own server; the paper's PhysGym servers used SGLang's default attention backend)
  [ $KEY = 1 ] || { echo "physgym_rep$3 skipped (no key)" >> "$STATUS"; return; }
  cp "$B/physgym-eval.sh" "$O/servers/physgym-eval.r$3.sh"
  step "physgym_rep$3" env TAG="$NAME-r$3" ROOT="$O/physgym" REPO="$W/repos/PhysGym" VENV="$W/venv-physgym" \
    SUMMARY_PY="$B/physgym_summary.py" CKPT_STEP0="$HF" ARM_STEP0="$NAME" GPU_STEP0="$1" PORT_STEP0="$2" SINGLE_ARM=1 \
    THINKING=on CTX=65536 MAXTOK=32000 TIMEOUT=1800 MODES=no_description_anonymous REPS="$3" PAR_PER_SERVER=24 \
    PROBLEM_TIMEOUT=24h PHYSGYM_KEY_IN_FILE=0 bash "$O/servers/physgym-eval.r$3.sh"
}

LANES=("lane_wilt_f2f 30200" "lane_cipherbank 30300" "lane_physgym 31400 1" "lane_physgym 31401 2" "lane_physgym 31402 3")
PIDS=""
for gi in "${!G[@]}"; do
  ( for li in "${!LANES[@]}"; do
      [ $((li % ${#G[@]})) = "$gi" ] || continue
      set -- ${LANES[$li]}; f=$1; shift; "$f" "${G[$gi]}" "$@"
    done ) &
  PIDS="$PIDS $!"
done
wait $PIDS
rm -f "$O/servers/.openrouter.key"
log "finished $ARM ($NAME):"; cat "$STATUS"
grep -q " exit [1-9]" "$STATUS" && { log "SOME PARTS FAILED"; exit 1; }
exit 0
