#!/usr/bin/env bash
# physgym-eval.sh — run PhysGym (arXiv 2507.15550, repo commit fe68079 + transfer/patches/physgym.diff) on one or two
# served checkpoints (ARM_STEP0 / ARM_STEP10) as an out-of-domain transfer test. Runs on a GPU machine.
# Always execute a private copy:  cp scripts/physgym-eval.sh /tmp/physgym-eval.$TAG.sh && bash /tmp/physgym-eval.$TAG.sh
#
#   TAG=physgym-l4-r5 MODES="no_description_anonymous" REPS="1 2 3 4 5" bash /tmp/physgym-eval.$TAG.sh
#   (judge: export OPENROUTER_API_KEY in the calling shell; with PHYSGYM_KEY_IN_FILE=1 (default) it is also written into
#    the per-run api_keys.env files (mode 600), never logged; PHYSGYM_KEY_IN_FILE=0 keeps it in the environment only)
#
# Verified mechanics (2026-09-03 workflow, code = ground truth):
#   * run_baseline.py's --base-url is NOT passed to the researcher (run_baseline.py:258-263); the server address comes from
#     VLLM_BASE_URL in ./api_keys.env, re-read with override=True on every call (llm_providers.py:548-549, 571). We still
#     pass --base-url so it lands in the experiment_setup log record.
#   * temperature/max_tokens were hardcoded 0.2/4000 (baseline_researcher.py:105; LLM_TEMPERATURE/LLM_MAX_TOKENS are dead
#     on this path) -> the patch reads PHYSGYM_AGENT_TEMPERATURE / PHYSGYM_AGENT_MAX_TOKENS. LLM_TIMEOUT (default 60 s) IS
#     read from api_keys.env (:561). The patch also clamps per-iteration experiments to the remaining quota (an over-budget
#     proposal raised an uncaught ValueError and killed the process), retries the POST 3x, and tolerates non-dict JSON.
#   * PhysGym sends only {model, messages, temperature, max_tokens}: thinking-off / top_p 1 / top_k -1 rely ENTIRELY on the
#     server defaults -> --sampling-defaults openai + --default-chat-template-kwargs, read back from the server log.
#   * The history folder is ../histories/baseline/<mode>/vllm_<tag>/ relative to the CWD (run_baseline.py:71-83); the
#     --llm-model value is that tag (SGLang does not validate the model field; a ':' in it is parsed as a LoRA name -> 400).
#     One frozen working dir per (arm, rep); tag = <arm>-rep<k>. Only the --idx-start/--idx-end form writes
#     experiment_<id>_results.json and skips finished problems (:341-345); --env-id form does not.
#   * Resume: a problem with a history file but no results file would silently CONTINUE from its old observations with the
#     iteration counter reset -> such half-finished problems are deleted before every (re)run.
#   * The LLM judge (google/gemini-2.5-flash via OpenRouter, hardcoded metrics.py:419-421) runs IN-LOOP; without a key it
#     records method=llm_failed and scores 0. The judge reads OPENROUTER_API_KEY from ./api_keys.env (never leave the
#     example file's placeholder line in it: load_dotenv(override=True) would clobber a real exported key).
#   * Model formulas are exec'd in-process without a sandbox timeout -> per-problem `timeout`.
set -uo pipefail
TAG="${TAG:-physgym-$(date -u +%Y%m%dT%H%MZ)}"
TW="${TRANSFER_WORK:-$(cd "$(dirname "$0")/.." && pwd)/work}"
ROOT="${ROOT:-$TW/results/physgym}"
REPO="${REPO:-$TW/repos/PhysGym}"
VENV="${VENV:-$TW/venv-physgym}"
CKPT_STEP0="${CKPT_STEP0:?set CKPT_STEP0 (HF checkpoint dir of the first arm)}"
CKPT_STEP10="${CKPT_STEP10:-}"
ARM_STEP0="${ARM_STEP0:-step00-base}"; ARM_STEP10="${ARM_STEP10:-step10-rl}"      # served-model-names AND tag prefixes (no ':')
GPU_STEP0="${GPU_STEP0:-0}"; GPU_STEP10="${GPU_STEP10:-1}"
PORT_STEP0="${PORT_STEP0:-31000}"; PORT_STEP10="${PORT_STEP10:-31001}"
MODES="${MODES:-no_description_anonymous}"     # L4 = no_description_anonymous, L3 = no_description, L2 = no_context, L1 = default
REPS="${REPS:-1 2 3 4 5}"                      # independent repetitions (PhysGym has no seed control)
IDX_START="${IDX_START:-0}"; IDX_END="${IDX_END:-97}"
THINKING="${THINKING:-off}"                     # off = RL dialect (default); on = Qwen thinking mode (reasoning parser splits thinking out of content)
case "$THINKING" in
  off) CTK='{"enable_thinking": false}'; RP="";                          MAXTOK="${MAXTOK:-8192}";  CTX="${CTX:-32768}"; TIMEOUT="${TIMEOUT:-900}" ;;
  on)  CTK='{"enable_thinking": true}';  RP="--reasoning-parser qwen3"; MAXTOK="${MAXTOK:-32000}"; CTX="${CTX:-65536}"; TIMEOUT="${TIMEOUT:-1800}" ;;   # 32k reply + prompt must fit (32768 -> HTTP 400 on every call, 09-03)
  *) echo "FAILED: THINKING must be off|on" >&2; exit 1 ;;
esac
TEMP="${TEMP:-1.0}"
MAX_ITER="${MAX_ITER:-20}"; QUOTA="${QUOTA:-100}"; EXP_PER_ITER="${EXP_PER_ITER:-20}"
PAR_PER_SERVER="${PAR_PER_SERVER:-48}"          # concurrent problem processes per server (server --max-running-requests 64)
PROBLEM_TIMEOUT="${PROBLEM_TIMEOUT:-24h}"   # per-problem wall-clock cap (a 3 h cap killed ~40 % of thinking-on problems under 24-way sharing)
MEMFRAC="${MEMFRAC:-0.85}"; MAXREQ="${MAXREQ:-64}"
KEEP_SERVERS="${KEEP_SERVERS:-0}"; SKIP_SERVERS="${SKIP_SERVERS:-0}"   # SKIP_SERVERS=1: servers already up
SINGLE_ARM="${SINGLE_ARM:-0}"                                           # 1 = run only ARM_STEP0/CKPT_STEP0 (e.g. one extra checkpoint vs an already-evaluated base)
OUT="$ROOT/$TAG"; mkdir -p "$OUT"
LOG="$OUT/physgym-eval.log"; log(){ echo "[$(date -u +%FT%TZ)] $*" | tee -a "$LOG"; }
SUMMARY_PY="${SUMMARY_PY:-$(cd "$(dirname "$0")" && pwd)/physgym_summary.py}"

case "$ARM_STEP0$ARM_STEP10$TAG" in *:*) log "FAILED: ':' not allowed in arm names / TAG (SGLang parses it as a LoRA name)"; exit 1;; esac
[ -f "$REPO/experiments/run_baseline.py" ] || { log "FAILED: $REPO missing"; exit 1; }
grep -q PHYSGYM_AGENT_TEMPERATURE "$REPO/methods/baseline_researcher.py" || { log "FAILED: $REPO is not patched (PHYSGYM_AGENT_TEMPERATURE missing)"; exit 1; }
[ -x "$VENV/bin/python" ] || { log "FAILED: venv $VENV missing"; exit 1; }
if [ "$SINGLE_ARM" = 1 ]; then ARMS="$ARM_STEP0"; PAIRS="$PORT_STEP0:$ARM_STEP0:$CKPT_STEP0"; PORTS="$PORT_STEP0"; else ARMS="$ARM_STEP0 $ARM_STEP10"; PAIRS="$PORT_STEP0:$ARM_STEP0:$CKPT_STEP0 $PORT_STEP10:$ARM_STEP10:$CKPT_STEP10"; PORTS="$PORT_STEP0 $PORT_STEP10"; fi
for pr in $PAIRS; do c="${pr##*:}"; [ -f "$c/config.json" ] || { log "FAILED: $c has no config.json"; exit 1; }; [ "$c" = "$CKPT_STEP0" ] && [ "$SINGLE_ARM" != 1 ] && [ "$CKPT_STEP10" = "$CKPT_STEP0" ] && continue; [ -f "$c/CONVERT_OK" ] || [ -f "$c/model.safetensors.index.json" ] || { log "FAILED: $c not a finished HF dir"; exit 1; }; done

start_server(){ # gpu port name ckpt
  local gpu="$1" port="$2" name="$3" ckpt="$4"
  CUDA_VISIBLE_DEVICES="$gpu" setsid nohup python3 -m sglang.launch_server --model-path "$ckpt" --served-model-name "$name" \
      --tp 1 --port "$port" --host 127.0.0.1 --mem-fraction-static "$MEMFRAC" --context-length "$CTX" \
      --sampling-defaults openai --default-chat-template-kwargs "$CTK" $RP --max-running-requests "$MAXREQ" ${EXTRA_SERVER_FLAGS:-} \
      > "$OUT/server-$name.log" 2>&1 < /dev/null &
  echo $! > "$OUT/server-$name.pid"
  echo "$ckpt" > "$OUT/served-$name.path"
}
log "TAG=$TAG thinking=$THINKING modes=[$MODES] reps=[$REPS] idx=[$IDX_START,$IDX_END) temp=$TEMP ctx=$CTX max_tokens=$MAXTOK timeout=$TIMEOUT max_iter=$MAX_ITER quota=$QUOTA exp/iter=$EXP_PER_ITER par/server=$PAR_PER_SERVER judge_key=$([ -n "${OPENROUTER_API_KEY:-}" ] && echo present || echo ABSENT) repo_md5=$(md5sum "$REPO/methods/baseline_researcher.py" | cut -c1-8) problem_timeout=$PROBLEM_TIMEOUT"
if [ "$SKIP_SERVERS" != 1 ]; then
  if pgrep -f "sglang.launch_serve[r].*--port (${PORT_STEP0}|${PORT_STEP10})( |$)" >/dev/null; then log "FAILED: a server already listens on $PORT_STEP0/$PORT_STEP10 (use SKIP_SERVERS=1 to reuse)"; exit 1; fi
  start_server "$GPU_STEP0" "$PORT_STEP0" "$ARM_STEP0" "$CKPT_STEP0"
  [ "$SINGLE_ARM" = 1 ] || start_server "$GPU_STEP10" "$PORT_STEP10" "$ARM_STEP10" "$CKPT_STEP10"
  log "launched servers: $ARM_STEP0@$PORT_STEP0 (GPU $GPU_STEP0, $CKPT_STEP0)$([ "$SINGLE_ARM" = 1 ] || echo " $ARM_STEP10@$PORT_STEP10 (GPU $GPU_STEP10, $CKPT_STEP10)") — waiting for /health"
fi
for port in $PORTS; do
  ok=0; for t in $(seq 1 60); do [ "$(curl -s -m 5 -o /dev/null -w '%{http_code}' http://127.0.0.1:$port/health)" = "200" ] && { ok=1; break; }; sleep 15; done
  [ "$ok" = 1 ] || { log "FAILED: server on port $port not healthy after 15 min"; exit 1; }
done
# two-sided readbacks: served ckpt path + sampling defaults + thinking-off probe (and thinking-ON must be switchable = the default is live)
for pair in $PAIRS; do
  IFS=: read -r port name ckpt <<<"$pair"
  curl -s -m 20 "http://127.0.0.1:$port/server_info" > "$OUT/server_info-$name-$(date -u +%H%M%S).json" || true
  python3 - "$OUT/server_info-$name-"*.json "$ckpt" <<'PY' || { log "FAILED: server_info readback for $name"; exit 1; }
import json,sys,glob
cands=[x for x in sys.argv[1:-1] if not x.endswith("-after.json")] or sys.argv[1:-1]   # ignore the end-of-run snapshot of a previous invocation
f=sorted(cands)[-1]; want=sys.argv[-1]; d=json.load(open(f))
mp=d.get("model_path") or d.get("server_args",{}).get("model_path"); sd=d.get("sampling_defaults") or d.get("server_args",{}).get("sampling_defaults")
assert mp==want, f"model_path {mp!r} != {want!r}"
assert sd=="openai", f"sampling_defaults {sd!r} != 'openai'"
print(f"server_info OK: model_path={mp} sampling_defaults={sd}")
PY
  if [ -f "$OUT/server-$name.log" ]; then
    grep -aq "sampling_defaults='openai'" "$OUT/server-$name.log" || { log "FAILED: server $name log does not show sampling_defaults='openai'"; exit 1; }
    grep -aq "Using default chat sampling params from model generation config" "$OUT/server-$name.log" && { log "FAILED: server $name applied generation_config sampling"; exit 1; }
  fi
  THINKING="$THINKING" python3 - "$port" "$name" <<'PY' || { log "FAILED: thinking probe on $name@$port"; exit 1; }
import json,sys,os,urllib.request
port,name=sys.argv[1],sys.argv[2]; mode=os.environ.get("THINKING","off")
def call(extra):
    body={"model":name,"temperature":0.0,"max_tokens":600,"messages":[{"role":"user","content":"Reply with exactly the word OK."}]}; body.update(extra)
    r=urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",data=json.dumps(body).encode(),headers={"Content-Type":"application/json"}),timeout=180)
    d=json.loads(r.read()); m=d["choices"][0]["message"]; return (m.get("content") or ""), d["usage"]["completion_tokens"], len(m.get("reasoning_content") or m.get("reasoning") or "")
if mode=="off":
    off,n_off,rc_off=call({})                                                # PhysGym-style request: no kwargs -> server default must be thinking OFF
    on,n_on,rc_on=call({"chat_template_kwargs":{"enable_thinking":True}})    # false-alarm test: thinking must be switchable ON
    print(f"{name}@{port}: default(off) -> {n_off} tokens {off[:30]!r} reasoning_chars={rc_off} | forced-on -> {n_on} tokens reasoning_chars={rc_on} think_tag={'<think>' in on}")
    assert n_off<=20 and "OK" in off and rc_off==0 and "<think>" not in off, "default request leaked reasoning or was not a short direct answer"
    # no reasoning parser on these servers: forced-on thinking shows up as extra tokens / a <think> tag in content, not as reasoning_content
    assert rc_on>0 or "<think>" in on or n_on>=n_off+5, "thinking could not be switched on -> the template kwargs path is dead, the OFF default is not proven live"
else:
    on,n_on,rc_on=call({})                                                   # PhysGym-style request -> server default must be thinking ON, parser splits it out
    off,n_off,rc_off=call({"chat_template_kwargs":{"enable_thinking":False}}) # false-alarm test: thinking must be switchable OFF
    print(f"{name}@{port}: default(on) -> {n_on} tokens content={on[:30]!r} reasoning_chars={rc_on} | forced-off -> {n_off} tokens reasoning_chars={rc_off}")
    assert rc_on>0 and "OK" in on and "<think>" not in on, "default request did not return reasoning_content with a clean answer (parser/template kwargs missing)"
    assert rc_off==0 and n_off<=20, "thinking could not be switched off -> the template kwargs path is dead, the ON default is not proven live"
PY
done
log "servers healthy; server_info + sampling_defaults=openai read back; thinking default-$THINKING two-sided probe PASS on $PORTS"

# per-(arm,rep) frozen working dirs; api_keys.env (read from CWD) carries the server URL, timeout, and (optionally) the judge key
mk_env(){ # dir port
  { echo "VLLM_BASE_URL=http://127.0.0.1:$2"; echo "LLM_TIMEOUT=$TIMEOUT";
    [ -n "${OPENROUTER_API_KEY:-}" ] && [ "${PHYSGYM_KEY_IN_FILE:-1}" = 1 ] && echo "OPENROUTER_API_KEY=$OPENROUTER_API_KEY"; } > "$1/api_keys.env"; chmod 600 "$1/api_keys.env"   # PHYSGYM_KEY_IN_FILE=0: the judge reads the exported env var instead (load_dotenv(override=True) keeps env vars the file does not set), so the key never lands on the shared drive
}
JOBS="$OUT/jobs.txt"; : > "$JOBS"
for pr in $PAIRS; do
  IFS=: read -r port arm _ck <<<"$pr"
  for k in $REPS; do
    W="$OUT/runs/$arm/rep$k/experiments"; mkdir -p "$W"; mk_env "$W" "$port"
    for m in $MODES; do
      # resume hygiene: drop half-finished problems (history without results) so they restart clean instead of continuing
      H="$OUT/runs/$arm/rep$k/histories/baseline/$m/vllm_$arm-rep$k"
      if [ -d "$H" ]; then for f in "$H"/experiment_*_history.json*; do [ -e "$f" ] || continue; id=${f##*/experiment_}; id=${id%%_*}; [ -f "$H/experiment_${id}_results.json" ] || { rm -f "$H"/experiment_${id}_*; echo "$arm rep$k $m env$id" >> "$OUT/half_finished_removed.txt"; }; done; fi
      for i in $(seq "$IDX_START" $((IDX_END-1))); do echo "$arm $port $k $m $i" >> "$JOBS"; done
    done
  done
done
log "fan-out: $(wc -l < "$JOBS") (arm,rep,mode,problem) jobs; $PAR_PER_SERVER concurrent per server; half-finished removed: $( [ -f "$OUT/half_finished_removed.txt" ] && wc -l < "$OUT/half_finished_removed.txt" || echo 0 )"
run_one(){ # arm port rep mode idx
  local arm="$1" port="$2" k="$3" m="$4" i="$5"
  local W="$OUT/runs/$arm/rep$k/experiments"; local L="$OUT/runs/$arm/rep$k/logs/$m"; mkdir -p "$L"
  ( cd "$W" && PHYSGYM_AGENT_TEMPERATURE="$TEMP" PHYSGYM_AGENT_MAX_TOKENS="$MAXTOK" timeout "$PROBLEM_TIMEOUT" \
      "$VENV/bin/python" "$REPO/experiments/run_baseline.py" --api-provider vllm --base-url "http://127.0.0.1:$port" \
      --llm-model "$arm-rep$k" --mode "$m" --idx-start "$i" --idx-end $((i+1)) --max-iterations "$MAX_ITER" --sample-quota "$QUOTA" \
      --experiments-per-iteration "$EXP_PER_ITER" --api-key-file api_keys.env > "$L/idx$i.log" 2>&1 ) || echo "$arm rep$k $m idx$i rc=$?" >> "$OUT/failed_jobs.txt"
}
export -f run_one; export OUT REPO VENV MAX_ITER QUOTA EXP_PER_ITER TEMP MAXTOK PROBLEM_TIMEOUT
XPIDS=""
for arm in $ARMS; do
  grep "^$arm " "$JOBS" | xargs -P "$PAR_PER_SERVER" -L 1 bash -c 'run_one "$@"' _ &
  XPIDS="$XPIDS $!"
done
wait $XPIDS   # only the two xargs pools — a bare `wait` would also wait for the setsid'd servers (children of this shell) forever
log "all jobs finished; failed_jobs: $( [ -f "$OUT/failed_jobs.txt" ] && wc -l < "$OUT/failed_jobs.txt" || echo 0 )"
if [ -s "$OUT/failed_jobs.txt" ]; then log "WARNING: $(wc -l < "$OUT/failed_jobs.txt") jobs FAILED (rc=124 = PROBLEM_TIMEOUT=$PROBLEM_TIMEOUT hit) — the summary below counts them as failures; re-run the same command (resume) with a larger PROBLEM_TIMEOUT before using the numbers"; echo "INCOMPLETE_FAILED_JOBS=$(wc -l < "$OUT/failed_jobs.txt")" > "$OUT/INCOMPLETE.txt"; else rm -f "$OUT/INCOMPLETE.txt"; fi
nres=$(find "$OUT/runs" -name 'experiment_*_results.json' | wc -l); log "results files: $nres / $(wc -l < "$JOBS") jobs"
if grep -rlaE "vLLM generation failed|Read timed out|Connection refused|HTTPError" "$OUT"/runs/*/rep*/logs >/dev/null 2>&1; then
  log "WARN: LLM transport errors in $(grep -rlaE 'vLLM generation failed|Read timed out|Connection refused|HTTPError' "$OUT"/runs/*/rep*/logs | wc -l) problem logs — inspect before scoring"
fi
for pr in $PAIRS; do IFS=: read -r port name _ck <<<"$pr"; curl -s -m 20 "http://127.0.0.1:$port/server_info" > "$OUT/server_info-$name-after.json" || true; done
"$VENV/bin/python" "$SUMMARY_PY" --root "$OUT" --step0 "$ARM_STEP0" --step10 "$ARM_STEP10" 2>&1 | tee -a "$LOG" || log "WARN: summary failed"
if [ "$KEEP_SERVERS" != 1 ] && [ "$SKIP_SERVERS" != 1 ]; then for name in $ARMS; do kill "$(cat "$OUT/server-$name.pid")" 2>/dev/null; done; sleep 5; log "servers stopped"; fi
log "DONE $TAG"
