#!/usr/bin/env bash
# launch_f2f.sh — Failing-to-Falsify (Wason 2-4-6 numeric, arXiv 2604.02485) transfer-eval launcher.
#   Each arm (a model) is served by SGLang (one TP=1 server per arm; started with transfer/serve.sh),
#   80 fixed OOD-test episodes (16 rules x 5 seed triples) x REPS repetitions per arm, arms interleaved.
#
#   launch_f2f.sh play         --root ROOT --think on|off --ports base=30001,step20=30002
#                              [--arms base,step20] [--reps 1-5] [--par 24] [--items all|N] [--max-new-tokens N]
#   launch_f2f.sh judge-guess  --root ROOT --think on|off --judge 'http://127.0.0.1:30100/v1#llama-3.3-70b-instruct' [--par 6]
#   launch_f2f.sh judge-compat --root ROOT --think on|off --judge 'http://127.0.0.1:30101/v1#qwen2.5-coder-32b-instruct' [--sandbox-user nobody] [--par 6]
#   launch_f2f.sh analyze      --root ROOT --think on|off            # official analyzer -> ROOT/results/... (paper-format summary.txt)
#   launch_f2f.sh summarize    --root ROOT                           # summarize_f2f.py -> ROOT/F2F_SUMMARY.{md,json}
#   launch_f2f.sh status       --root ROOT --think on|off
# Env: REPO (patched wason_experiments dir; default transfer/work/repos/llm-confirmation-bias/wason_experiments)
#      PY (python3), F2F_JUDGE_API_KEY_FILE / F2F_JUDGE_PROVIDER (OpenRouter judge only; key never printed)
# Layout: ROOT/think<on|off>/rep<k>/ood_test/<served-name>/baseline/<RULE>/inst<i>/{<stem>.jsonl,.txt,.model_params.json,
#         <stem>_judge_guesses.jsonl,<stem>_judge_compatibility.jsonl,calls.ndjson}; ROOT/logs; ROOT/provenance; ROOT/_incomplete
# Long runs: setsid nohup bash launch_f2f.sh play ... > ROOT/logs/play.out 2>&1 < /dev/null &
#   The launcher pid (= process-group leader under setsid) is written to ROOT/logs/<cmd>_think<T>.pid; stop everything
#   it started with   kill -- -$(cat ROOT/logs/play_thinkoff.pid)   (by pid/pgid only, never by pattern).
# Markers: <inst dir>/FMTABORT = 10 format retries exhausted (transcript kept + judged, NEVER re-run, even on a later
#   launcher invocation); stale partial transcripts of infra failures -> ROOT/_incomplete (not judged, not scored).
# Preflight: per-arm /get_server_info readback (served name + enable_thinking) AND a cross-arm diff of the condition fields
#   (sampling_defaults, reasoning_parser, context_length, attention backend, page size, ... see .bin/server_info_consistency.py):
#   any hard-field difference refuses to start; the diff table is saved to ROOT/provenance/server_info_consistency_*.txt.
# Stop strings: OFF sends ["```"] (upstream OpenAI path = paper's no-think Qwen3 wrapper); ON sends none (paper's THINKING
#   Qwen3 wrapper has no ``` stop; a ``` inside the reasoning stream would cut the answer off). Override: F2F_STOPS='[...]'.
set -u
CMD=${1:-}; shift || true
PKG_DIR=$(cd "$(dirname "$0")" && pwd)
REPO=${REPO:-${TRANSFER_WORK:-$(cd "$PKG_DIR/.." && pwd)/work}/repos/llm-confirmation-bias/wason_experiments}
PY=${PY:-${TRANSFER_WORK:-$(cd "$PKG_DIR/.." && pwd)/work}/venv/bin/python}
PKG=$(cd "$(dirname "$0")" && pwd)
ROOT=; THINK=; PORTS=; ARMS=base,step20; REPS=1-5; PAR=24; ITEMS=all; MAXTOK=; JUDGE=; SANDBOX_USER=; EXTRA=
while [ $# -gt 0 ]; do
  case $1 in
    --root) ROOT=$2; shift 2;; --think) THINK=$2; shift 2;; --ports) PORTS=$2; shift 2;; --arms) ARMS=$2; shift 2;;
    --reps) REPS=$2; shift 2;; --par) PAR=$2; shift 2;; --items) ITEMS=$2; shift 2;; --max-new-tokens) MAXTOK=$2; shift 2;;
    --judge) JUDGE=$2; shift 2;; --sandbox-user) SANDBOX_USER=$2; shift 2;;
    *) if [ "$CMD" = summarize ]; then EXTRA="$EXTRA $1"; shift; else echo "unknown arg $1"; exit 2; fi;;
  esac
done
[ -n "$ROOT" ] || { echo "--root required"; exit 2; }
case $CMD in play|judge-guess|judge-compat|analyze|status) [ "$THINK" = on ] || [ "$THINK" = off ] || { echo "--think on|off required"; exit 2; };; esac
[ -d "$REPO/src" ] || { echo "REPO=$REPO has no src/ (patched wason_experiments dir expected)"; exit 2; }
mkdir -p "$ROOT/logs" "$ROOT/provenance" "$ROOT/.bin"
case $CMD in play|judge-guess|judge-compat) echo $$ > "$ROOT/logs/${CMD}_think${THINK}.pid";; esac
log(){ echo "$(date -u +%FT%TZ) $*"; }
arm_name(){ case $1 in base) echo qwen3.8-27b-base;; step20) echo witnessgym-qwen3.8-27b-rl-broadcast-step20;; *) echo "$1";; esac; }   # any other arm key is used as the served model name itself
arm_port(){ echo ",$PORTS," | tr ',' '\n' | grep "^$1=" | head -1 | cut -d= -f2; }
expand_reps(){ case $1 in *-*) seq "${1%-*}" "${1#*-}";; *) echo "$1" | tr ',' '\n';; esac; }
RULES="O1_1 O1_2 O1_3 O1_4 O2_1 O2_2 O2_3 O2_4 O3_1 O3_2 O3_3 O3_4 O4_1 O4_2 O4_3 O4_4"   # the 16 OOD-test rules (rules/rules.yaml)
INSTS="1 2 3 4 5"                                                                          # 5 seed triples per group (data/instances.csv, ood_test)
md5f(){ if command -v md5sum >/dev/null; then md5sum "$1" | cut -d' ' -f1; else md5 -q "$1"; fi; }

# ---- helper: is an instance dir complete? (latest transcript has 91 model turns = 1 announce + 45 x (test, announce))
cat > "$ROOT/.bin/f2f_complete.py" <<'PY'
import glob, json, os, sys
d = sys.argv[1]
files = []
for p in sorted(glob.glob(os.path.join(d, "*.jsonl"))):
    if p.endswith(("_judge_guesses.jsonl", "_judge_compatibility.jsonl")): continue
    try:
        if open(p, encoding="utf-8").readline().strip().startswith('{"meta"'): files.append(p)
    except Exception: pass
if not files: sys.exit(1)
rows = [json.loads(l) for l in open(files[-1], encoding="utf-8") if l.strip()]
sys.exit(0 if sum(1 for r in rows if r.get("role") == "model") >= 91 else 1)
PY

# ---- helper: the arms must be served under IDENTICAL conditions. Compares the saved /get_server_info snapshots field by
#      field. HARD fields must be equal across all arms (refuse to
#      start otherwise); SOFT fields are printed for the record (memory-derived or per-process values that may differ).
cat > "$ROOT/.bin/server_info_consistency.py" <<'PY'
import json, sys
HARD = ["sampling_defaults", "default_chat_template_kwargs", "reasoning_parser", "tool_call_parser", "context_length",
        "chat_template", "dtype", "quantization", "kv_cache_dtype", "tp_size", "max_running_requests", "attention_backend",
        "page_size", "mamba_radix_cache_strategy", "max_mamba_cache_size", "mem_fraction_static", "preferred_sampling_params",
        "json_model_override_args", "speculative_algorithm", "enable_cache_report", "schedule_policy", "chunked_prefill_size",
        "stream_interval", "grammar_backend", "version"]
SOFT = ["max_total_tokens", "max_prefill_tokens", "cuda_graph_max_bs", "random_seed", "model_path", "served_model_name"]
files = sys.argv[1:]
infos = {f: json.load(open(f)) for f in files}
def val(d, k):
    return json.dumps(d.get(k, "<absent>"), sort_keys=True)
bad = []; lines = []
for k in HARD + SOFT:
    vals = {f: val(d, k) for f, d in infos.items()}
    if all(v == '"<absent>"' for v in vals.values()):
        continue
    same = len(set(vals.values())) == 1
    tag = "HARD" if k in HARD else "soft"
    lines.append(f"{tag:4s} {k:30s} {'same' if same else 'DIFFERS'}  " + ("" if same else " | ".join(f"{f.rsplit('/',1)[-1]}={v[:80]}" for f, v in vals.items())))
    if k in HARD and not same:
        bad.append(k)
print("\n".join(lines))
print(f"checked {len(files)} servers, {len([l for l in lines if l.startswith('HARD')])} hard fields, {len(bad)} hard mismatch(es): {bad}")
sys.exit(1 if bad else 0)
PY

# ---- server readback: served name + thinking flag must match; save the JSON
readback(){ # readback PORT NAME THINK OUTFILE
  local info; info=$(curl -sf -m 10 "http://127.0.0.1:$1/get_server_info") || { echo "no /get_server_info on port $1"; return 1; }
  echo "$info" > "$4"
  echo "$info" | "$PY" -c '
import json,sys
d=json.load(sys.stdin); name,think=sys.argv[1],sys.argv[2]
ok = d.get("served_model_name")==name and bool((d.get("default_chat_template_kwargs") or {}).get("enable_thinking"))==(think=="on")
print("OK" if ok else "MISMATCH served=%s enable_thinking=%s expected %s think=%s" % (d.get("served_model_name"), (d.get("default_chat_template_kwargs") or {}).get("enable_thinking"), name, think))
sys.exit(0 if ok else 1)' "$2" "$3"
}

provenance(){ # provenance TAG extra-json
  local out="$ROOT/provenance/${1}_$(date -u +%Y%m%d-%H%M%S).json"
  "$PY" - "$REPO" "$out" "$0" "$2" <<'PY'
import hashlib, json, os, subprocess, sys, socket, time
repo, out, launcher, extra = sys.argv[1:5]
def sh(c):
    try: return subprocess.check_output(c, shell=True, cwd=repo, text=True, stderr=subprocess.STDOUT).strip()
    except Exception as e: return f"ERR {e}"
def md5(p):
    try: return hashlib.md5(open(p, "rb").read()).hexdigest()
    except Exception: return None
patched = ["src/run_experiment.py", "src/models/gpt_4o.py", "src/models/openai_compat_judge.py", "src/run_judge_guess.py",
           "src/run_judge_compatibility.py", "src/experiments/baseline.py", "src/experiments/parse_utils.py", "src/analyze_variant_metrics.py",
           "rules/rules.yaml", "data/instances.csv", "prompts/judge_equivalence_wason.txt", "prompts/generate_hypothesis_function.txt"]
d = {"ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "host": socket.gethostname(), "repo": repo,
     "commit": sh("git rev-parse HEAD"), "status": sh("git status --porcelain"), "diffstat": sh("git diff --stat"),
     "md5": {p: md5(os.path.join(repo, p)) for p in patched}, "launcher_md5": md5(launcher),
     "python": sys.version, "openai_sdk": sh(f"{sys.executable} -c 'import openai;print(openai.__version__)'"), "extra": json.loads(extra)}
json.dump(d, open(out, "w"), indent=1); print(out)
PY
}

case $CMD in
# =============================================================================================================
play)
  [ -n "$PORTS" ] || { echo "--ports base=P,bcast=P,rl=P required"; exit 2; }
  if [ -z "$MAXTOK" ]; then if [ "$THINK" = on ]; then MAXTOK=16384; else MAXTOK=1024; fi; fi
  if [ -z "${F2F_STOPS:-}" ]; then if [ "$THINK" = on ]; then F2F_STOPS='[]'; else F2F_STOPS='["```"]'; fi; fi   # see header
  TAG="play_think$THINK"
  # pre-flight: every arm's server must serve the right model with the right thinking flag
  INFOS=""; TS=$(date -u +%Y%m%d-%H%M%S)
  for arm in $(echo "$ARMS" | tr ',' ' '); do
    port=$(arm_port "$arm"); [ -n "$port" ] || { echo "no port for arm $arm in --ports"; exit 2; }
    name=$(arm_name "$arm"); info="$ROOT/provenance/server_info_${arm}_think${THINK}_${TS}.json"
    r=$(readback "$port" "$name" "$THINK" "$info") || { echo "READBACK FAILED arm=$arm port=$port: $r"; exit 3; }
    log "readback arm=$arm port=$port $name think=$THINK: $r"; INFOS="$INFOS $info"
  done
  # shellcheck disable=SC2086
  if [ "$(echo $INFOS | wc -w)" -ge 2 ]; then   # arms must be served under identical conditions (hard fields), else refuse
    cons="$ROOT/provenance/server_info_consistency_think${THINK}_${TS}.txt"
    if "$PY" "$ROOT/.bin/server_info_consistency.py" $INFOS > "$cons" 2>&1; then log "server conditions identical across arms: $(tail -1 "$cons") -> $cons"
    else echo "SERVER CONDITIONS DIFFER ACROSS ARMS (refusing to start):"; cat "$cons"; exit 3; fi
  fi
  provenance "$TAG" "{\"cmd\":\"play\",\"think\":\"$THINK\",\"max_new_tokens\":$MAXTOK,\"ports\":\"$PORTS\",\"arms\":\"$ARMS\",\"reps\":\"$REPS\",\"par\":$PAR,\"items\":\"$ITEMS\",\"temperature\":0.6,\"top_p\":0.95,\"stop\":$F2F_STOPS,\"http_timeout_s\":${F2F_HTTP_TIMEOUT:-1800}}"
  # job lines: REP RULE INST ARM NAME PORT (item-major so all arms progress in lockstep; one pool per arm)
  JOBS="$ROOT/.bin/jobs_think${THINK}_$(date -u +%Y%m%d-%H%M%S).txt"; : > "$JOBS"
  n=0
  for rep in $(expand_reps "$REPS"); do for rule in $RULES; do for inst in $INSTS; do
    n=$((n+1)); if [ "$ITEMS" != all ] && [ $(( (n-1) % 80 )) -ge "$ITEMS" ]; then continue; fi
    for arm in $(echo "$ARMS" | tr ',' ' '); do echo "$rep $rule $inst $arm $(arm_name "$arm") $(arm_port "$arm")" >> "$JOBS"; done
  done; done; done
  log "jobs: $(wc -l < "$JOBS") lines -> $JOBS"
  run_item(){ # REP RULE INST ARM NAME PORT
    local rep=$1 rule=$2 inst=$3 arm=$4 name=$5 port=$6
    local outdir="$ROOT/think$THINK/rep$rep" dir="$ROOT/think$THINK/rep$rep/ood_test/$name/baseline/$rule/inst$inst"
    if [ -e "$dir/FMTABORT" ]; then echo "skip  $(date -u +%T)Z $name rep$rep $rule/inst$inst (FMTABORT marker: format retries exhausted earlier; kept, never re-run)"; return 0; fi
    if [ -d "$dir" ] && "$PY" "$ROOT/.bin/f2f_complete.py" "$dir"; then echo "skip  $(date -u +%T)Z $name rep$rep $rule/inst$inst (complete)"; return 0; fi
    local attempt=1
    while [ $attempt -le 2 ]; do
      if [ -d "$dir" ] && ls "$dir"/*.jsonl >/dev/null 2>&1; then   # stale partial transcript -> quarantine (not judged, not scored)
        local q="$ROOT/_incomplete/think$THINK/rep$rep/$name/$rule/inst$inst/attempt${attempt}_$(date -u +%Y%m%d-%H%M%S)"; mkdir -p "$q"; mv "$dir"/* "$q"/ 2>/dev/null
      fi
      mkdir -p "$dir"
      echo "start $(date -u +%T)Z $name rep$rep $rule/inst$inst attempt=$attempt port=$port"
      local so="$dir/stdout_attempt${attempt}_$(date -u +%Y%m%d-%H%M%S).txt"
      ( cd "$REPO" && env OPENAI_BASE_URL="http://127.0.0.1:$port/v1" OPENAI_API_KEY=x OPENAI_MODEL_OVERRIDE="$name" \
          F2F_MAX_NEW_TOKENS="$MAXTOK" F2F_STOPS="$F2F_STOPS" F2F_CALL_LOG="$dir/calls.ndjson" \
          "$PY" -m src.run_experiment --model "gpt-$name" --model-dir-name "$name" --variant baseline \
          --outdir "$outdir" --testcases "$rule" --instances "$inst" ) > "$so" 2>&1
      local rc=$?
      if "$PY" "$ROOT/.bin/f2f_complete.py" "$dir"; then echo "end   $(date -u +%T)Z $name rep$rep $rule/inst$inst rc=$rc complete"; return 0; fi
      if grep -q "Expected a single-line" "$so"; then
        touch "$dir/FMTABORT"
        echo "FMTABORT $(date -u +%T)Z $name rep$rep $rule/inst$inst (10 format retries exhausted; partial transcript kept + judged, marker written, NOT re-run; counted invalid in the gate, success/IF from the analyzer on the partial transcript)"; return 0
      fi
      echo "INFRA $(date -u +%T)Z $name rep$rep $rule/inst$inst rc=$rc attempt=$attempt: $(grep -E 'Error|error|Exception' "$so" | tail -1 | cut -c1-160)"
      attempt=$((attempt+1)); sleep 15
    done
    echo "FAILED $(date -u +%T)Z $name rep$rep $rule/inst$inst after 2 attempts (see $dir)"; return 0
  }
  export -f run_item; export ROOT THINK REPO PY MAXTOK F2F_STOPS
  for arm in $(echo "$ARMS" | tr ',' ' '); do
    ( grep " $arm " "$JOBS" | xargs -P "$PAR" -L 1 bash -c 'run_item "$@"' _ ) > "$ROOT/logs/play_${arm}_think${THINK}_$(date -u +%Y%m%d-%H%M%S).log" 2>&1 &
  done
  wait
  log "play done think=$THINK"; bash "$0" status --root "$ROOT" --think "$THINK" --arms "$ARMS" --reps "$REPS"
  ;;
# =============================================================================================================
judge-guess|judge-compat)
  [ -n "$JUDGE" ] || { echo "--judge 'http://host:port/v1#model' required"; exit 2; }
  jurl=${JUDGE%%#*}; jname=${JUDGE##*#}
  case $jname in qwen3.8-27b-*) echo "refusing: judge '$jname' is an evaluated arm; the judge must be a separate model/server"; exit 3;; esac
  case $jurl in
    *openrouter.ai*) : "${F2F_JUDGE_API_KEY_FILE:=/root/.openrouter_key}"; export F2F_JUDGE_API_KEY_FILE
       [ -r "$F2F_JUDGE_API_KEY_FILE" ] || { echo "OpenRouter judge: key file $F2F_JUDGE_API_KEY_FILE unreadable"; exit 3; }
       [ -n "${F2F_JUDGE_PROVIDER:-}" ] || { echo "OpenRouter judge: set F2F_JUDGE_PROVIDER='{\"order\":[\"Crusoe\"],\"allow_fallbacks\":false}' (pin a bf16 provider)"; exit 3; }
       log "judge=OpenRouter $jname provider=$F2F_JUDGE_PROVIDER (key file present; NOT the paper's local vLLM — disclose)";;
    *) info=$(curl -sf -m 10 "${jurl%/v1}/get_server_info") || { echo "judge server readback failed at ${jurl%/v1}/get_server_info"; exit 3; }
       echo "$info" > "$ROOT/provenance/judge_info_${CMD}_think${THINK}_$(date -u +%Y%m%d-%H%M%S).json"
       echo "$info" | grep -q "\"served_model_name\": *\"$jname\"" || { echo "judge served_model_name != $jname: $(echo "$info" | head -c 300)"; exit 3; }
       log "judge readback OK: $jname at $jurl";;
  esac
  TAG="${CMD}_think$THINK"; provenance "$TAG" "{\"cmd\":\"$CMD\",\"think\":\"$THINK\",\"judge\":\"$JUDGE\",\"arms\":\"$ARMS\",\"reps\":\"$REPS\",\"sandbox_user\":\"$SANDBOX_USER\"}"
  JOBS="$ROOT/.bin/jobs_${CMD}_think${THINK}_$(date -u +%Y%m%d-%H%M%S).txt"; : > "$JOBS"
  for rep in $(expand_reps "$REPS"); do for arm in $(echo "$ARMS" | tr ',' ' '); do
    d="$ROOT/think$THINK/rep$rep/ood_test/$(arm_name "$arm")/baseline"; [ -d "$d" ] && echo "$d" >> "$JOBS"
  done; done
  log "judge dirs: $(wc -l < "$JOBS")"
  run_judge(){ # EXPDIR
    local d=$1 rc
    if [ "$CMD" = judge-guess ]; then
      ( cd "$REPO" && env F2F_JUDGE_CALL_LOG="$d/judge_guess_calls.ndjson" "$PY" -m src.run_judge_guess --expdir "$d" --model-path "$JUDGE" --prompts-dir prompts --judge-temp 0.0 --judge-max-tokens 12 ) > "$d/judge_guess_stdout_$(date -u +%Y%m%d-%H%M%S).txt" 2>&1; rc=$?
    else
      local pre=""
      if [ -n "$SANDBOX_USER" ]; then
        if [ "$(id -u)" = 0 ] && command -v setpriv >/dev/null; then
          chmod -R a+rwX "$d"; pre="setpriv --reuid=$(id -u "$SANDBOX_USER") --regid=$(id -g "$SANDBOX_USER") --clear-groups env HOME=/tmp"
        else echo "WARN: --sandbox-user requested but not root or no setpriv; running compat judge as $(id -un)"; fi
      fi
      ( cd "$REPO" && ulimit -v 16000000 2>/dev/null; env F2F_JUDGE_CALL_LOG="$d/judge_compat_calls.ndjson" $pre "$PY" -m src.run_judge_compatibility --expdir "$d" --model-path "$JUDGE" --prompt-file prompts/generate_hypothesis_function.txt ) > "$d/judge_compat_stdout_$(date -u +%Y%m%d-%H%M%S).txt" 2>&1; rc=$?
    fi
    echo "$(date -u +%T)Z $CMD rc=$rc $d"
  }
  export -f run_judge; export CMD REPO PY JUDGE SANDBOX_USER F2F_JUDGE_PROVIDER F2F_JUDGE_API_KEY_FILE 2>/dev/null
  xargs -P "$PAR" -L 1 bash -c 'run_judge "$@"' _ < "$JOBS" 2>&1 | tee -a "$ROOT/logs/${CMD}_think${THINK}.log" | tee "$JOBS.out"
  # a per-directory rc!=0 fails the stage (otherwise a judge that failed everywhere would still report success)
  nfail=$(grep -c " $CMD rc=[1-9]" "$JOBS.out" || true)
  if [ "${nfail:-0}" -gt 0 ]; then log "$CMD FAILED for $nfail/$(wc -l < "$JOBS") dirs (see per-dir judge_*_stdout_*.txt)"; exit 1; fi
  log "$CMD done ($(wc -l < "$JOBS") dirs, 0 failures)"
  ;;
# =============================================================================================================
analyze)
  for rep in $(expand_reps "$REPS"); do for arm in $(echo "$ARMS" | tr ',' ' '); do
    d="$ROOT/think$THINK/rep$rep/ood_test/$(arm_name "$arm")/baseline"; [ -d "$d" ] || continue
    ( cd "$REPO" && "$PY" -m src.analyze_variant_metrics --expdir "$d" --outdir "$ROOT/results/think$THINK/rep$rep" ) > "$ROOT/logs/analyze_think${THINK}_rep${rep}_${arm}.log" 2>&1 && log "analyze ok rep$rep $arm -> $ROOT/results/think$THINK/rep$rep/ood_test/$(arm_name "$arm")/baseline/summary.txt" || log "analyze FAILED rep$rep $arm (see log)"
  done; done
  ;;
summarize)
  # shellcheck disable=SC2086
  "$PY" "$PKG/summarize_f2f.py" --root "$ROOT" --repo "$REPO" $EXTRA
  ;;
status)
  for rep in $(expand_reps "$REPS"); do for arm in $(echo "$ARMS" | tr ',' ' '); do
    name=$(arm_name "$arm"); base="$ROOT/think$THINK/rep$rep/ood_test/$name/baseline"; [ -d "$base" ] || { echo "rep$rep $name: (no dir)"; continue; }
    tot=0; done_=0; jg=0; jc=0; fa=0
    for d in "$base"/O*_*/inst*; do [ -d "$d" ] || continue; tot=$((tot+1))
      "$PY" "$ROOT/.bin/f2f_complete.py" "$d" && done_=$((done_+1)); [ -e "$d/FMTABORT" ] && fa=$((fa+1))
      ls "$d"/*_judge_guesses.jsonl >/dev/null 2>&1 && jg=$((jg+1)); ls "$d"/*_judge_compatibility.jsonl >/dev/null 2>&1 && jc=$((jc+1))
    done
    echo "think$THINK rep$rep $name: dirs=$tot complete=$done_ fmtabort=$fa judged_guess=$jg judged_compat=$jc"
  done; done
  ;;
*) sed -n 2,20p "$0"; exit 2;;
esac
