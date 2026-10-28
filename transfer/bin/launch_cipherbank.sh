#!/usr/bin/env bash
# launch_cipherbank.sh — CipherBank (ACL 2025 Findings, arXiv 2504.19093; Leey21/CipherBank @ 05b450c + transfer/patches/cipherbank.diff;
#   items = HF yu0226/CipherBank @ d2f8f7cc, 262 plaintexts x 9 ciphers = 2,358) transfer-eval launcher.
#   Each arm (a model) is served by SGLang 0.5.19 (one TP=1 server per arm, byte-identical flags; see transfer/serve.sh). The
#   thinking dialect is a property of the SERVER (--default-chat-template-kwargs); this launcher only checks it.
#
#   launch_cipherbank.sh setup     [--repo REPO] [--hf-dir DIR]     clone+pin the repo, git apply patches/cipherbank.diff, fetch the 9 HF files at the
#                                                                    pinned revision (or copy them from --hf-dir), build data/test_hf.jsonl,
#                                                                    verify every md5, check the venv deps
#   launch_cipherbank.sh play      --root ROOT --think on|off --ports base=30001,step20=30002
#                                  [--arms base,step20] [--reps 1-2] [--par 32] [--rows all|N] [--max-tokens N] [--hint]
#   launch_cipherbank.sh status    --root ROOT [--think on|off] [--hint] [--rows all|N]
#   launch_cipherbank.sh seedcheck --root ROOT --think on|off --ports base=P[,...] [--arms base] [--hint] [--n 2]   (smoke: is `seed` honoured?)
#   launch_cipherbank.sh summarize --root ROOT [--think on|off] [--hint] [--rows N] [summarize_cipherbank.py args]
# Env: REPO (patched CipherBank checkout; default transfer/work/repos/CipherBank), PY (default transfer/work/venv/bin/python),
#      CB_MAX_TOKENS (overrides the dialect default: ON 32768 / OFF 4096), CB_HTTP_TIMEOUT (s, default 3600), CB_RETRY_SLEEP (s, 10)
# Layout: ROOT/think<on|off>/<nohint|hint>/<served-name>/rep<k>/{results.ndjson,calls.ndjson,outputs.ndjson}; ROOT/logs; ROOT/provenance; ROOT/.bin
# Long runs: setsid nohup bash launch_cipherbank.sh play ... > ROOT/logs/play_on.out 2>&1 < /dev/null &
#   The launcher pid (= process-group leader under setsid) is written to ROOT/logs/play_think<T>_<mode>.pid; stop everything it
#   started with   kill -- -$(cat ROOT/logs/play_thinkon_nohint.pid)   (by pid/pgid only, never by pattern), or softly with
#   touch ROOT/STOP (checked by every worker before it starts an item; running items finish; the launcher then exits 5).
# Resume: re-run the same command; an item whose LAST results record has error==null AND max_tokens == the current budget is never
#   re-run. Endpoint failures are re-run once inside the run, then scored 0 (error recorded, counted per arm by the summarizer) and
#   re-run by the next invocation. A DRIVER-ERROR leaves the item without a record -> driver exit 6 -> launcher exit 4 (never "play done").
# One play per (ROOT, dialect, mode): ROOT/logs/play_think<T>_<mode>.lock.d holds the live launcher pid; a second invocation while
#   it is alive refuses with exit 6 (a stale lock from a dead pid is taken over).
# Preflight: pinned commit + md5 of the patched files and the data; per-arm /get_server_info readback (served_model_name == arm,
#   default_chat_template_kwargs.enable_thinking is a real bool == (--think on) [absent/null = MISMATCH], sampling_defaults == openai);
#   cross-arm diff of the hard condition fields (.bin/server_info_consistency.py, copied from the failing-to-falsify package): any
#   hard-field difference refuses to start; table saved to ROOT/provenance/.
# Client sends: model, messages, max_tokens, seed (per (rep,item), same for every arm), stop=[] — never temperature/top_p.
set -u
CMD=${1:-}; shift || true
PKG=$(cd "$(dirname "$0")" && pwd)
REPO=${REPO:-${TRANSFER_WORK:-$(cd "$PKG/.." && pwd)/work}/repos/CipherBank}
PY=${PY:-${TRANSFER_WORK:-$(cd "$PKG/.." && pwd)/work}/venv/bin/python}
REMOTE=https://github.com/Leey21/CipherBank.git
EXPECTED_COMMIT=05b450ca8f11b966fb73d06278dea213e87540f9
HF_REV=d2f8f7cc7ca6a0c1b8aa25898747a9b80ed67729
CIPHERS="Rot13 Atbash Polybius Vigenere Reverse SwapPairs ParityShift DualAvgCode WordShift"
# md5 pins (2026-09-10): patched files, prompt template, shots, merged HF data, the 9 HF files, the patch itself
MD5_TEST_PY=552318b81aef2f6cd4db1d2f122dc948; MD5_TOOLS_PY=62de52e01411323be85046848034ce73; MD5_PROMPT_PY=41bdf6dcc821462928269f2a813cce99
MD5_SHOTS=bef9df36965bcb34b5789256e3956213; MD5_DATA=592d88ac06e810c4de37f531dba8dbdd; MD5_PATCH=65a573552ad5bff22035ac4e69c45c17
hf_md5(){ case $1 in Rot13) echo fa66f18823251c2a7ab24feefa9ba20d;; Atbash) echo cff0205851649c31bff6d58918df3afe;; Polybius) echo 76e636c5f7f0c34f8f79d9528ea549ef;;
  Vigenere) echo dfc1bb8b25f03957f13c35d129797dcc;; Reverse) echo e3c578e7637d1c61d1910627f840424a;; SwapPairs) echo 442bff4f64be8c8a800ef91599ecca88;;
  ParityShift) echo 3dc563eaf6ea576c2e2b25d6b34d371a;; DualAvgCode) echo 5eb44632b4015b7ca55b9d9c746c49a6;; WordShift) echo d96a1fbe45b66648ec746c7e43e9a199;; esac; }
ROOT=; THINK=; PORTS=; ARMS=base,step20; REPS=1-2; PAR=32; ROWS=all; MAXTOK=${CB_MAX_TOKENS:-}; HINT=0; HF_DIR=; EXTRA=; NCHECK=2
while [ $# -gt 0 ]; do
  case $1 in
    --root) ROOT=$2; shift 2;; --think) THINK=$2; shift 2;; --ports) PORTS=$2; shift 2;; --arms) ARMS=$2; shift 2;;
    --reps) REPS=$2; shift 2;; --par) PAR=$2; shift 2;; --rows) ROWS=$2; shift 2;; --max-tokens) MAXTOK=$2; shift 2;;
    --hint) HINT=1; shift;; --repo) REPO=$2; shift 2;; --hf-dir) HF_DIR=$2; shift 2;; --n) NCHECK=$2; shift 2;;
    *) if [ "$CMD" = summarize ]; then EXTRA="$EXTRA $1"; shift; else echo "unknown arg $1"; exit 2; fi;;
  esac
done
MODE=nohint; [ "$HINT" = 1 ] && MODE=hint
if [ -n "$ROOT" ]; then [ "$CMD" = play ] && { mkdir -p "$ROOT" || exit 2; }; [ -d "$ROOT" ] && ROOT=$(cd "$ROOT" && pwd); fi   # absolute: the driver chdir()s into REPO
log(){ echo "$(date -u +%FT%TZ) $*"; }
md5f(){ if command -v md5sum >/dev/null; then md5sum "$1" | cut -d' ' -f1; else md5 -q "$1"; fi; }
arm_name(){ case $1 in base) echo qwen3.8-27b-base;; step20) echo witnessgym-qwen3.8-27b-rl-broadcast-step20;; *) echo "$1";; esac; }   # any other arm key is used as the served model name itself
arm_port(){ echo ",$PORTS," | tr ',' '\n' | grep "^$1=" | head -1 | cut -d= -f2; }
expand_reps(){ case $1 in *-*) seq "${1%-*}" "${1#*-}" | tr '\n' ',' | sed 's/,$//';; *) echo "$1";; esac; }
check_md5(){ # check_md5 FILE EXPECTED LABEL
  local got; got=$(md5f "$1" 2>/dev/null || echo missing)
  if [ "$got" = "$2" ]; then echo "  ok   $3 $got"; else echo "  BAD  $3 got=$got expected=$2 ($1)"; return 1; fi; }
verify_repo(){ # pinned commit + patched files + data; refuse otherwise
  local head bad=0; head=$(git -C "$REPO" rev-parse HEAD 2>/dev/null || echo none)
  [ "$head" = "$EXPECTED_COMMIT" ] || { echo "REPO=$REPO HEAD=$head != pinned $EXPECTED_COMMIT"; bad=1; }
  check_md5 "$REPO/test.py" "$MD5_TEST_PY" "test.py (patched)" || bad=1
  check_md5 "$REPO/utils/tools.py" "$MD5_TOOLS_PY" "utils/tools.py (patched)" || bad=1
  check_md5 "$REPO/prompt.py" "$MD5_PROMPT_PY" "prompt.py (upstream, unpatched)" || bad=1
  check_md5 "$REPO/data/shot_case.jsonl" "$MD5_SHOTS" "data/shot_case.jsonl (3 shots)" || bad=1
  check_md5 "$REPO/data/test_hf.jsonl" "$MD5_DATA" "data/test_hf.jsonl (HF @ ${HF_REV:0:8}, merged)" || bad=1
  check_md5 "$PKG/../patches/cipherbank.diff" "$MD5_PATCH" "patches/cipherbank.diff" || bad=1
  return $bad; }

case $CMD in
# =============================================================================================================
setup)
  if [ ! -d "$REPO/.git" ]; then log "cloning $REMOTE -> $REPO"; git clone --quiet "$REMOTE" "$REPO" || exit 3; fi
  git -C "$REPO" checkout --quiet "$EXPECTED_COMMIT" 2>/dev/null || { echo "cannot checkout $EXPECTED_COMMIT"; exit 3; }
  if git -C "$REPO" apply --check --reverse "$PKG/../patches/cipherbank.diff" 2>/dev/null; then log "patch already applied"
  elif git -C "$REPO" apply --check "$PKG/../patches/cipherbank.diff" 2>/dev/null; then git -C "$REPO" apply "$PKG/../patches/cipherbank.diff" && log "patch applied"
  else echo "patch neither applied nor applicable to $REPO (dirty tree?)"; git -C "$REPO" status --short; exit 3; fi
  mkdir -p "$REPO/data/hf"
  for c in $CIPHERS; do
    f="$REPO/data/hf/$c.jsonl"
    if [ -n "$HF_DIR" ]; then cp "$HF_DIR/$c.jsonl" "$f"
    elif [ ! -s "$f" ]; then curl -sf -m 120 -L "https://huggingface.co/datasets/yu0226/CipherBank/resolve/$HF_REV/data/$c.jsonl" -o "$f" || { echo "HF download failed: $c"; exit 3; }; fi
    check_md5 "$f" "$(hf_md5 "$c")" "HF data/$c.jsonl" || exit 3
  done
  "$PY" "$PKG/cb_driver.py" build-data --repo "$REPO" --hf-dir "$REPO/data/hf" || exit 3
  log "verifying pins"; verify_repo || exit 3
  "$PY" -c 'import openai, Levenshtein, tqdm, numpy; print("deps ok: openai", openai.__version__, "numpy", numpy.__version__)' || { echo "missing deps -> $PY -m pip install Levenshtein tqdm   (openai==1.109.1 and numpy come with transfer/setup.sh)"; exit 3; }
  log "setup ok: REPO=$REPO"
  ;;
# =============================================================================================================
play)
  [ -n "$ROOT" ] || { echo "--root required"; exit 2; }
  [ "$THINK" = on ] || [ "$THINK" = off ] || { echo "--think on|off required"; exit 2; }
  [ -n "$PORTS" ] || { echo "--ports base=P,step20=P required"; exit 2; }
  if [ -z "$MAXTOK" ]; then if [ "$THINK" = on ]; then MAXTOK=32768; else MAXTOK=4096; fi; fi
  mkdir -p "$ROOT/logs" "$ROOT/provenance" "$ROOT/.bin"
  # ---- one play per (ROOT, dialect, mode): mkdir is atomic on every FS we use (no flock on macOS for the dry run); stale = dead pid
  LOCKD="$ROOT/logs/play_think${THINK}_${MODE}.lock.d"; PIDF="$ROOT/logs/play_think${THINK}_${MODE}.pid"
  take_lock(){ mkdir "$LOCKD" 2>/dev/null && { echo $$ > "$LOCKD/pid"; echo $$ > "$PIDF"; return 0; }; return 1; }
  if ! take_lock; then
    other=$(cat "$LOCKD/pid" 2>/dev/null || echo none)
    if [ "$other" != none ] && kill -0 "$other" 2>/dev/null; then
      echo "ANOTHER PLAY think=$THINK mode=$MODE IS RUNNING on $ROOT (launcher pid $other, lock $LOCKD) — refusing to start (exit 6)"; exit 6
    fi
    log "stale lock (pid $other not alive) -> taking over"; rm -rf "$LOCKD"; take_lock || { echo "cannot take lock $LOCKD"; exit 6; }
  fi
  trap 'rm -rf "$LOCKD"' EXIT
  log "verifying repo pins"; verify_repo || { echo "REPO pin check failed (run setup)"; exit 3; }
  # ---- helper: the arms must be served under IDENTICAL conditions (copied from failing-to-falsify/launch_f2f.sh)
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
  readback(){ # readback PORT NAME THINK OUTFILE — strict: enable_thinking must be a real bool (absent/null/{} = MISMATCH; Qwen3
    #   templates THINK by default when the kwarg is absent), sampling_defaults must be "openai" (the pre-registered temperature 1.0)
    local info; info=$(curl -sf -m 10 "http://127.0.0.1:$1/get_server_info") || { echo "no /get_server_info on port $1"; return 1; }
    echo "$info" > "$4"
    echo "$info" | "$PY" -c '
import json,sys
d=json.load(sys.stdin); name,think=sys.argv[1],sys.argv[2]
kw = d.get("default_chat_template_kwargs")
if not isinstance(kw, dict): kw = d.get("chat_template_kwargs")
et = kw.get("enable_thinking", "<absent>") if isinstance(kw, dict) else "<absent>"
bad = []
if d.get("served_model_name") != name: bad.append("served_model_name=%s (want %s)" % (d.get("served_model_name"), name))
if not isinstance(et, bool) or et != (think == "on"): bad.append("enable_thinking=%s (want %s for --think %s)" % (et, think == "on", think))
if d.get("sampling_defaults") != "openai": bad.append("sampling_defaults=%s (want openai)" % d.get("sampling_defaults", "<absent>"))
print("OK" if not bad else "MISMATCH " + "; ".join(bad))
sys.exit(0 if not bad else 1)' "$2" "$3"
  }
  TS=$(date -u +%Y%m%d-%H%M%S); INFOS=""
  for arm in $(echo "$ARMS" | tr ',' ' '); do
    port=$(arm_port "$arm"); [ -n "$port" ] || { echo "no port for arm $arm in --ports"; exit 2; }
    name=$(arm_name "$arm"); info="$ROOT/provenance/server_info_${arm}_think${THINK}_${TS}.json"
    r=$(readback "$port" "$name" "$THINK" "$info") || { echo "READBACK FAILED arm=$arm port=$port: $r"; exit 3; }
    log "readback arm=$arm port=$port $name think=$THINK: $r"; INFOS="$INFOS $info"
  done
  # shellcheck disable=SC2086
  if [ "$(echo $INFOS | wc -w)" -ge 2 ]; then
    cons="$ROOT/provenance/server_info_consistency_think${THINK}_${TS}.txt"
    if "$PY" "$ROOT/.bin/server_info_consistency.py" $INFOS > "$cons" 2>&1; then log "server conditions identical across arms: $(tail -1 "$cons") -> $cons"
    else echo "SERVER CONDITIONS DIFFER ACROSS ARMS (refusing to start):"; cat "$cons"; exit 3; fi
  fi
  REPLIST=$(expand_reps "$REPS")
  # ---- provenance: env, server_info per arm, commit, md5s, config
  "$PY" - "$REPO" "$ROOT/provenance/play_think${THINK}_${MODE}_${TS}.json" "$0" "$PKG" "$ROOT" "$THINK" "$MODE" "$MAXTOK" "$PORTS" "$ARMS" "$REPLIST" "$PAR" "$ROWS" "$TS" <<'PY'
import glob, hashlib, json, os, platform, socket, subprocess, sys, time
repo, out, launcher, pkg, root, think, mode, maxtok, ports, arms, reps, par, rows, ts = sys.argv[1:15]
def sh(c):
    try: return subprocess.check_output(c, shell=True, cwd=repo, text=True, stderr=subprocess.STDOUT).strip()
    except Exception as e: return f"ERR {e}"
def md5(p):
    try: return hashlib.md5(open(p, "rb").read()).hexdigest()
    except Exception: return None
servers = {os.path.basename(p).split("_")[2]: json.load(open(p)) for p in glob.glob(f"{root}/provenance/server_info_*_think{think}_{ts}.json")}
d = {"bench": "cipherbank", "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "host": socket.gethostname(),
     "repo": repo, "repo_remote": "https://github.com/Leey21/CipherBank", "repo_head": sh("git rev-parse HEAD"), "repo_status": sh("git status --porcelain"),
     "hf_dataset": "yu0226/CipherBank", "hf_revision": "d2f8f7cc7ca6a0c1b8aa25898747a9b80ed67729",
     "md5": {p: md5(os.path.join(repo, p)) for p in ["test.py", "utils/tools.py", "prompt.py", "cipher/encryption.py", "data/shot_case.jsonl", "data/test_hf.jsonl", "data/test.jsonl"]},
     "patch_md5": md5(os.path.join(pkg, "..", "patches", "cipherbank.diff")), "launcher_md5": md5(launcher), "driver_md5": md5(os.path.join(pkg, "cb_driver.py")),
     "summarizer_md5": md5(os.path.join(pkg, "summarize_cipherbank.py")),
     "config": {"think": think, "mode": mode, "max_tokens": int(maxtok), "ports": ports, "arms": arms, "reps": reps, "par": int(par), "rows": rows,
                "temperature": "not sent (server default)", "top_p": "not sent (server default)",
                "seed": "1_000_000*rep + 10*row + cipher_index (sent; determinism verified only by a seedcheck provenance file, if present)",
                "stop": [], "http_timeout_s": float(os.getenv("CB_HTTP_TIMEOUT", "3600")), "shots": 3, "format_type": "markdown", "number": False},
     "servers": servers,
     "env": {"python": sys.version, "platform": platform.platform(), "openai_sdk": sh(f"{sys.executable} -c 'import openai;print(openai.__version__)'"),
             "numpy": sh(f"{sys.executable} -c 'import numpy;print(numpy.__version__)'"), "Levenshtein": sh(f"{sys.executable} -c 'import Levenshtein;print(Levenshtein.__version__)'"),
             "sandbox": "none needed: this benchmark executes no model-written code"}}
json.dump(d, open(out, "w"), indent=1); print(out)
PY
  log "play think=$THINK mode=$MODE max_tokens=$MAXTOK reps=$REPLIST rows=$ROWS par=$PAR arms=$ARMS"
  PIDS=""; RCFILE="$ROOT/logs/play_rc_think${THINK}_${MODE}_${TS}.txt"; : > "$RCFILE"
  for arm in $(echo "$ARMS" | tr ',' ' '); do
    port=$(arm_port "$arm"); name=$(arm_name "$arm")
    ( "$PY" "$PKG/cb_driver.py" run --repo "$REPO" --root "$ROOT" --think "$THINK" --mode "$MODE" --arm "$arm" --name "$name" --port "$port" \
        --reps "$REPLIST" --par "$PAR" --max-tokens "$MAXTOK" --rows "$ROWS" > "$ROOT/logs/play_${arm}_think${THINK}_${MODE}_${TS}.log" 2>&1; echo "$arm $?" >> "$RCFILE" ) &
    PIDS="$PIDS $!"
  done
  wait
  log "drivers exited: $(tr '\n' ' ' < "$RCFILE")"
  bash "$0" status --root "$ROOT" --think "$THINK" --rows "$ROWS" $( [ "$HINT" = 1 ] && echo --hint )
  if grep -q " 6$" "$RCFILE"; then log "DRIVER-ERRORS in $(grep -c ' 6$' "$RCFILE") driver(s): items left WITHOUT a record (see DRIVER-ERROR lines in $ROOT/logs/play_*_think${THINK}_${MODE}_${TS}.log); NOT done"; exit 4; fi
  if grep -q " 5$" "$RCFILE"; then log "STOPPED by $ROOT/STOP (remove it and re-run the same command to resume)"; exit 5; fi
  grep -qv " 0$" "$RCFILE" && { log "some driver exited non-zero (see logs)"; exit 4; }
  log "play done think=$THINK mode=$MODE"
  ;;
# =============================================================================================================
status)
  [ -n "$ROOT" ] || { echo "--root required"; exit 2; }
  "$PY" "$PKG/cb_driver.py" status --root "$ROOT" --rows "$ROWS" ${THINK:+--think "$THINK"} $( [ "$HINT" = 1 ] && echo --mode hint )
  ;;
seedcheck)
  [ -n "$ROOT" ] || { echo "--root required"; exit 2; }
  [ "$THINK" = on ] || [ "$THINK" = off ] || { echo "--think on|off required"; exit 2; }
  [ -n "$PORTS" ] || { echo "--ports base=P[,...] required"; exit 2; }
  rc=0
  for arm in $(echo "$ARMS" | tr ',' ' '); do
    port=$(arm_port "$arm"); [ -n "$port" ] || continue
    "$PY" "$PKG/cb_driver.py" seedcheck --repo "$REPO" --root "$ROOT" --think "$THINK" --mode "$MODE" --arm "$arm" --name "$(arm_name "$arm")" --port "$port" --n "$NCHECK" || rc=$?
  done
  exit $rc
  ;;
summarize)
  [ -n "$ROOT" ] || { echo "--root required"; exit 2; }
  # forward the launcher-level options explicitly (they are consumed by the option parser above, so they never reach EXTRA)
  # shellcheck disable=SC2086
  "$PY" "$PKG/summarize_cipherbank.py" --root "$ROOT" ${THINK:+--think "$THINK"} $( [ "$HINT" = 1 ] && echo --mode hint ) $( [ "$ROWS" != all ] && echo --expected-rows "$ROWS" ) $EXTRA
  ;;
*) sed -n 2,31p "$0"; exit 2;;
esac
