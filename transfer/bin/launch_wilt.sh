#!/usr/bin/env bash
# launch_wilt.sh — WILT (Wason Inductive Logic Test, arXiv 2410.10998, riotgames/wilt@6344f04) transfer-eval launcher.
#   Each arm (a model) is served by SGLang (one TP=1 server per arm; transfer/serve.sh), full split =
#   50 fixed hidden lambdas x REPS repetitions per arm. One `main.py` process per (arm, rep) with --multi episodes in
#   flight; reps run sequentially per arm, arms in parallel.
#
#   launch_wilt.sh play      --root ROOT --think on|off --ports base=30001,step20=30002
#                            [--arms base,step20] [--reps 1-5] [--multi 25] [--max-tokens N] [--sandbox-user nobody] [--split full|lite]
#   launch_wilt.sh summarize --root ROOT [summarize_wilt.py args]
#   launch_wilt.sh status    --root ROOT --think on|off
# Env: REPO (patched riotgames/wilt checkout; default transfer/work/repos/wilt), PY (default transfer/work/venv/bin/python)
# Layout: ROOT/think<on|off>/<served-name>/rep<k>/{prompts->REPO/prompts, checkpoints/, results/sglang__<name>_full_context.json,
#         calls.ndjson, stdout_*.txt}; ROOT/logs; ROOT/provenance
# Long runs: setsid nohup bash launch_wilt.sh play ... > ROOT/logs/play.out 2>&1 < /dev/null &
#   The launcher pid (= process-group leader under setsid) is written to ROOT/logs/play_think<T>.pid; stop everything it
#   started with   kill -- -$(cat ROOT/logs/play_thinkoff.pid)   (by pid/pgid only, never by pattern).
# Markers: <rep dir>/INFRA_TAINTED = at least one episode of that attempt ended in an ENDPOINT failure (SglangModel raised
#   EndpointFailure after 6 retries; the harness stores infra_errors>0 in the results JSON and the model file logs the error
#   to calls.ndjson) -> the rep is treated as incomplete and re-run as a whole (max 2 attempts per invocation, again on the
#   next invocation); the superseded calls.ndjson is kept as calls_superseded_<ts>.ndjson. Whole-rep re-run, not --single N:
#   upstream --single patches full_context[N-1] by list position, which is not the episode index under --multi.
# Preflight: readback of every arm's /get_server_info (served name + enable_thinking) + a cross-arm diff of the condition
#   fields (any hard-field difference refuses to start; table saved to ROOT/provenance/server_info_consistency_*.txt) + a
#   sandbox self-test (true rule must score, wrong lambda must not, a dunder payload must be rejected) run with the same PY
#   and WILT_SANDBOX_USER as the real episodes; any of the three failing refuses to start.
set -u
CMD=${1:-}; shift || true
PKG_DIR=$(cd "$(dirname "$0")" && pwd)
REPO=${REPO:-${TRANSFER_WORK:-$(cd "$PKG_DIR/.." && pwd)/work}/repos/wilt}
PY=${PY:-${TRANSFER_WORK:-$(cd "$PKG_DIR/.." && pwd)/work}/venv/bin/python}
PKG=$(cd "$(dirname "$0")" && pwd)
ROOT=; THINK=; PORTS=; ARMS=base,step20; REPS=1-5; MULTI=25; MAXTOK=; SANDBOX_USER=; SPLIT=full; EXTRA=
while [ $# -gt 0 ]; do
  case $1 in
    --root) ROOT=$2; shift 2;; --think) THINK=$2; shift 2;; --ports) PORTS=$2; shift 2;; --arms) ARMS=$2; shift 2;;
    --reps) REPS=$2; shift 2;; --multi) MULTI=$2; shift 2;; --max-tokens) MAXTOK=$2; shift 2;; --sandbox-user) SANDBOX_USER=$2; shift 2;;
    --split) SPLIT=$2; shift 2;;
    *) if [ "$CMD" = summarize ]; then EXTRA="$EXTRA $1"; shift; else echo "unknown arg $1"; exit 2; fi;;
  esac
done
[ -n "$ROOT" ] || { echo "--root required"; exit 2; }
case $CMD in play|status) [ "$THINK" = on ] || [ "$THINK" = off ] || { echo "--think on|off required"; exit 2; };; esac
[ -f "$REPO/main.py" ] && [ -f "$REPO/harness/sandbox.py" ] || { echo "REPO=$REPO is not the PATCHED wilt checkout (main.py + harness/sandbox.py expected)"; exit 2; }
mkdir -p "$ROOT/logs" "$ROOT/provenance" "$ROOT/.bin"
[ "$CMD" = play ] && echo $$ > "$ROOT/logs/play_think${THINK}.pid"
log(){ echo "$(date -u +%FT%TZ) $*"; }
arm_name(){ case $1 in base) echo qwen3.8-27b-base;; step20) echo witnessgym-qwen3.8-27b-rl-broadcast-step20;; *) echo "$1";; esac; }   # any other arm key is used as the served model name itself
arm_port(){ echo ",$PORTS," | tr ',' '\n' | grep "^$1=" | head -1 | cut -d= -f2; }
expand_reps(){ case $1 in *-*) seq "${1%-*}" "${1#*-}";; *) echo "$1" | tr ',' '\n';; esac; }
case $SPLIT in full) NRULES=50; QUAL="";; lite) NRULES=10; QUAL="_lite";; *) echo "--split full|lite"; exit 2;; esac
md5f(){ if command -v md5sum >/dev/null; then md5sum "$1" | cut -d' ' -f1; else md5 -q "$1"; fi; }

cat > "$ROOT/.bin/wilt_complete.py" <<'PY'
# exit 0: results JSON has test_index 1..N and no episode carries infra_errors>0
# exit 2: all episodes present but >=1 is infra-tainted (EndpointFailure after retries; see models/sglang_model.py)
# exit 1: missing/short/unreadable
import json, sys
p, n = sys.argv[1], int(sys.argv[2])
try: d = json.load(open(p))
except Exception: sys.exit(1)
eps = [e for e in d if isinstance(e, dict) and "test_index" in e]
idx = sorted({int(e.get("test_index", -1)) for e in eps})
if idx != list(range(1, n + 1)): sys.exit(1)
tainted = [int(e["test_index"]) for e in eps if int(e.get("infra_errors") or 0) > 0]
if tainted:
    print(f"infra-tainted episodes: {tainted}"); sys.exit(2)
sys.exit(0)
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

readback(){ # PORT NAME THINK OUTFILE
  local info; info=$(curl -sf -m 10 "http://127.0.0.1:$1/get_server_info") || { echo "no /get_server_info on port $1"; return 1; }
  echo "$info" > "$4"
  echo "$info" | "$PY" -c '
import json,sys
d=json.load(sys.stdin); name,think=sys.argv[1],sys.argv[2]
ok = d.get("served_model_name")==name and bool((d.get("default_chat_template_kwargs") or {}).get("enable_thinking"))==(think=="on")
print("OK" if ok else "MISMATCH served=%s enable_thinking=%s expected %s think=%s" % (d.get("served_model_name"), (d.get("default_chat_template_kwargs") or {}).get("enable_thinking"), name, think))
sys.exit(0 if ok else 1)' "$2" "$3"
}

provenance(){ # TAG extra-json
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
patched = ["main.py", "models/model_factory.py", "models/sglang_model.py", "utils/metrics.py", "harness/llm_reasoning_harness.py",
           "harness/sandbox.py", "harness/test_cases.py", "harness/hypothesis_space.py", "prompts/instruction.txt"]
d = {"ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "host": socket.gethostname(), "repo": repo,
     "commit": sh("git rev-parse HEAD"), "status": sh("git status --porcelain"), "diffstat": sh("git diff --stat"),
     "md5": {p: md5(os.path.join(repo, p)) for p in patched}, "launcher_md5": md5(launcher),
     "python": sys.version, "openai_sdk": sh(f"{sys.executable} -c 'import openai;print(openai.__version__)'"), "extra": json.loads(extra)}
json.dump(d, open(out, "w"), indent=1); print(out)
PY
}

case $CMD in
play)
  [ -n "$PORTS" ] || { echo "--ports base=P,bcast=P,rl=P required"; exit 2; }
  if [ -z "$MAXTOK" ]; then if [ "$THINK" = on ]; then MAXTOK=32768; else MAXTOK=4096; fi; fi
  INFOS=""; TS=$(date -u +%Y%m%d-%H%M%S)
  for arm in $(echo "$ARMS" | tr ',' ' '); do
    port=$(arm_port "$arm"); [ -n "$port" ] || { echo "no port for arm $arm in --ports"; exit 2; }
    info="$ROOT/provenance/server_info_${arm}_think${THINK}_${TS}.json"
    r=$(readback "$port" "$(arm_name "$arm")" "$THINK" "$info") || { echo "READBACK FAILED arm=$arm port=$port: $r"; exit 3; }
    log "readback arm=$arm port=$port $(arm_name "$arm") think=$THINK: $r"; INFOS="$INFOS $info"
  done
  # shellcheck disable=SC2086
  if [ "$(echo $INFOS | wc -w)" -ge 2 ]; then   # arms must be served under identical conditions (hard fields), else refuse
    cons="$ROOT/provenance/server_info_consistency_think${THINK}_${TS}.txt"
    if "$PY" "$ROOT/.bin/server_info_consistency.py" $INFOS > "$cons" 2>&1; then log "server conditions identical across arms: $(tail -1 "$cons") -> $cons"
    else echo "SERVER CONDITIONS DIFFER ACROSS ARMS (refusing to start):"; cat "$cons"; exit 3; fi
  fi
  if [ -n "$SANDBOX_USER" ]; then
    if [ "$(id -u)" = 0 ] && command -v setpriv >/dev/null; then id "$SANDBOX_USER" >/dev/null || { echo "no such user $SANDBOX_USER"; exit 2; }; log "sandbox: final-guess lambdas evaluated as uid $(id -u "$SANDBOX_USER") via setpriv (+ rlimits, timeout)"
    else log "WARN: --sandbox-user given but not root or no setpriv: sandbox = child process + rlimits + timeout only"; fi
  else log "sandbox: child process + AST whitelist + rlimits + timeout (no uid drop; pass --sandbox-user nobody on the pod)"; fi
  # sandbox self-test with the exact runtime configuration (same PY, same uid drop). Catches e.g. a venv python that
  # `nobody` cannot execute, which would otherwise turn EVERY final guess into "Invalid lambda function." and look like
  # model behaviour (the instrument must prove it rings before we trust its silence).
  ( cd "$REPO" && env PYTHONPATH="$REPO" ${SANDBOX_USER:+WILT_SANDBOX_USER="$SANDBOX_USER"} "$PY" - <<'SELFTEST'
import sys
from harness.sandbox import check_guess
from harness.test_cases import TESTS_FULL
rule = TESTS_FULL["2"]                      # lambda x, y, z: x < y < z
good = check_guess("lambda x, y, z: x < y < z", rule, 3)
wrong = check_guess("lambda x, y, z: x <= y <= z", rule, 3)
evil = check_guess("lambda x, y, z: __import__('os').system('true') or True", rule, 3)
ok = bool(good.get("ok") and good.get("all_correct") and wrong.get("ok") and not wrong.get("all_correct") and not evil.get("ok"))
print("sandbox selftest:", "OK" if ok else "FAIL", {"good": good, "wrong": wrong, "evil": evil.get("error")})
sys.exit(0 if ok else 1)
SELFTEST
  ) || { echo "SANDBOX SELF-TEST FAILED (see above); refusing to start"; exit 3; }
  provenance "play_think$THINK" "{\"cmd\":\"play\",\"think\":\"$THINK\",\"max_tokens\":$MAXTOK,\"temperature\":\"server default (openai: 1.0/1.0)\",\"ports\":\"$PORTS\",\"arms\":\"$ARMS\",\"reps\":\"$REPS\",\"multi\":$MULTI,\"split\":\"$SPLIT\",\"sandbox_user\":\"$SANDBOX_USER\"}"
  run_arm(){ # ARM NAME PORT
    local arm=$1 name=$2 port=$3 rep
    for rep in $(expand_reps "$REPS"); do
      local wd="$ROOT/think$THINK/$name/rep$rep" res
      mkdir -p "$wd"; [ -e "$wd/prompts" ] || ln -s "$REPO/prompts" "$wd/prompts"
      res="$wd/results/sglang__${name}${QUAL}_full_context.json"
      if [ ! -e "$wd/INFRA_TAINTED" ] && "$PY" "$ROOT/.bin/wilt_complete.py" "$res" "$NRULES" >/dev/null 2>&1; then echo "skip  $(date -u +%T)Z $name rep$rep (complete)"; continue; fi
      [ -e "$wd/INFRA_TAINTED" ] && echo "redo  $(date -u +%T)Z $name rep$rep (INFRA_TAINTED marker from an earlier attempt: endpoint failure after retries; whole rep re-run)"
      local attempt=1
      while [ $attempt -le 2 ]; do
        rm -rf "$wd/checkpoints" "$wd/results" "$wd/INFRA_TAINTED"   # no --resume: with --multi the upstream checkpoint index is not monotonic
        # calls.ndjson must describe the attempt that produced results/: rotate the previous attempt's log (kept for forensics)
        [ -f "$wd/calls.ndjson" ] && mv "$wd/calls.ndjson" "$wd/calls_superseded_$(date -u +%Y%m%d-%H%M%S).ndjson"
        echo "start $(date -u +%T)Z $name rep$rep attempt=$attempt port=$port multi=$MULTI max_tokens=$MAXTOK"
        local so="$wd/stdout_attempt${attempt}_$(date -u +%Y%m%d-%H%M%S).txt"
        ( cd "$wd" && env PYTHONPATH="$REPO" SGLANG_BASE_URL="http://127.0.0.1:$port/v1" WILT_MAX_TOKENS="$MAXTOK" WILT_CALL_LOG="$wd/calls.ndjson" \
            ${SANDBOX_USER:+WILT_SANDBOX_USER="$SANDBOX_USER"} "$PY" "$REPO/main.py" --model "sglang/$name" --split "$SPLIT" --multi "$MULTI" ) > "$so" 2>&1
        local rc=$?
        local crc; "$PY" "$ROOT/.bin/wilt_complete.py" "$res" "$NRULES" > "$wd/.complete_check" 2>&1; crc=$?
        # infra taint = results JSON carries infra_errors>0 (EndpointFailure caught by the harness) OR the model file logged
        # an exhausted retry (belt and braces; the print precedes the raise). Either way the rep is re-run, never scored.
        if [ $crc -eq 2 ] || grep -q "Error in SglangModel after retries" "$so"; then
          touch "$wd/INFRA_TAINTED"
          echo "INFRA $(date -u +%T)Z $name rep$rep rc=$rc attempt=$attempt: $(grep -c 'Error in SglangModel after retries' "$so") exhausted API retries, $(cat "$wd/.complete_check" | tr '\n' ' ') -> rep marked INFRA_TAINTED, will be re-run"
        elif [ $crc -eq 0 ]; then echo "end   $(date -u +%T)Z $name rep$rep rc=$rc complete"; break
        else echo "INCOMPLETE $(date -u +%T)Z $name rep$rep rc=$rc attempt=$attempt (results missing/short); $(grep -iE 'error|exception|traceback' "$so" | tail -1 | cut -c1-160)"; fi
        attempt=$((attempt+1)); sleep 15
      done
    done
  }
  export -f run_arm expand_reps; export ROOT THINK REPO PY MAXTOK MULTI REPS SANDBOX_USER SPLIT QUAL NRULES
  for arm in $(echo "$ARMS" | tr ',' ' '); do
    ( run_arm "$arm" "$(arm_name "$arm")" "$(arm_port "$arm")" ) > "$ROOT/logs/play_${arm}_think${THINK}_$(date -u +%Y%m%d-%H%M%S).log" 2>&1 &
  done
  wait; log "play done think=$THINK"; bash "$0" status --root "$ROOT" --think "$THINK" --arms "$ARMS" --reps "$REPS" --split "$SPLIT"
  ;;
summarize)
  # shellcheck disable=SC2086
  "$PY" "$PKG/summarize_wilt.py" --root "$ROOT" --repo "$REPO" --split "$SPLIT" $EXTRA
  ;;
status)
  for arm in $(echo "$ARMS" | tr ',' ' '); do name=$(arm_name "$arm"); for rep in $(expand_reps "$REPS"); do
    res="$ROOT/think$THINK/$name/rep$rep/results/sglang__${name}${QUAL}_full_context.json"
    "$PY" "$ROOT/.bin/wilt_complete.py" "$res" "$NRULES" >/dev/null 2>&1; crc=$?
    if [ -e "$ROOT/think$THINK/$name/rep$rep/INFRA_TAINTED" ] || [ $crc -eq 2 ]; then s=INFRA_TAINTED; elif [ $crc -eq 0 ]; then s=complete; elif [ -f "$res" ]; then s=short; else s=missing; fi
    ck="$ROOT/think$THINK/$name/rep$rep/checkpoints"; n=0; [ -d "$ck" ] && n=$("$PY" -c 'import glob,json,sys; fs=glob.glob(sys.argv[1]+"/*.json"); print(json.load(open(fs[0])).get("total_answers",0) if fs else 0)' "$ck" 2>/dev/null || echo 0)
    echo "think$THINK $name rep$rep: $s (checkpoint episodes=$n)"
  done; done
  ;;
*) sed -n 2,14p "$0"; exit 2;;
esac
