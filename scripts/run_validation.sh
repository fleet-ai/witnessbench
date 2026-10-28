#!/usr/bin/env bash
# Validate one Hugging Face checkpoint: serve it with serve_policy.sh, wait until the server is
# healthy, check that it uses the OpenAI sampling defaults, run the evaluator against it, and stop
# the evaluator and the server on exit, also when the script is ended by a signal.
#
#   scripts/run_validation.sh HF_DIR OUT_DIR [evaluator options...]
#
# OUT_DIR must not exist; it receives server.log and the evaluator's outputs. PORT (default 30000)
# selects the port, CUDA_VISIBLE_DEVICES the GPU. The exit status is the evaluator's; 2 = bad arguments,
# 3 = the server did not start or does not use the OpenAI sampling defaults.
set -euo pipefail
usage="usage: $0 HF_DIR OUT_DIR [evaluator options...]"
case "${1:-}" in
  -h | --help)
    echo "$usage"
    echo "Serve HF_DIR with serve_policy.sh, run python -m witness_agent.evaluate against it with the evaluator"
    echo "options, and write server.log and the results to OUT_DIR, which must not exist. The environment"
    echo "variables PORT (default 30000) and CUDA_VISIBLE_DEVICES select the port and the GPU."
    exit 0 ;;
esac
if [ "$#" -lt 2 ]; then
  echo "$usage" >&2
  exit 2
fi
hf_dir=$1
out=$2
shift 2
port=${PORT:-30000}
if [ ! -f "$hf_dir/config.json" ]; then
  echo "error: $hf_dir has no config.json" >&2
  exit 2
fi
if [ -e "$out" ]; then
  echo "error: $out already exists" >&2
  exit 2
fi
mkdir -p "$out"
log=$out/server.log

set -m  # background jobs get their own process group, so the whole server tree can be stopped
"$(dirname "$0")/serve_policy.sh" "$hf_dir" "$port" >"$log" 2>&1 &
server=$!
evaluator=
stop_server() {
  disown "$server" 2>/dev/null || true  # no job-status message when the group is stopped
  kill -TERM -- "-$server" 2>/dev/null || return 0
  for _ in $(seq 30); do  # wait for the whole group: server workers can outlive the launcher process
    kill -0 -- "-$server" 2>/dev/null || return 0
    sleep 1
  done
  kill -KILL -- "-$server" 2>/dev/null || true
}
stop_all() {
  if [ -n "$evaluator" ]; then
    kill -TERM "$evaluator" 2>/dev/null || true
    wait "$evaluator" 2>/dev/null || true
  fi
  stop_server
}
trap stop_all EXIT

healthy=0
for _ in $(seq 60); do
  if [ "$(curl -s -m 5 -o /dev/null -w '%{http_code}' "http://127.0.0.1:$port/health")" = 200 ]; then
    healthy=1
    break
  fi
  if ! kill -0 "$server" 2>/dev/null; then
    echo "error: the server exited; see $log" >&2
    exit 3
  fi
  sleep 15
done
if [ "$healthy" != 1 ]; then
  echo "error: the server was not healthy after 15 minutes; see $log" >&2
  exit 3
fi
# SGLang logs its arguments once at startup, in the line `server_args=ServerArgs(...)`; it holds
# sampling_defaults='openai' when the --sampling-defaults openai of serve_policy.sh took effect.
if ! grep -q "sampling_defaults='openai'" "$log"; then
  echo "error: $log does not show sampling_defaults='openai'" >&2
  exit 3
fi
# A second, independent check: SGLang logs this sentence when it takes the default sampling
# parameters of chat requests from the checkpoint's generation_config.
if grep -q "Using default chat sampling params from model generation config" "$log"; then
  echo "error: the server samples with the checkpoint's generation_config; see $log" >&2
  exit 3
fi
# The evaluator runs in the background, in the script's process group (job control off), and is
# waited for: the EXIT trap knows its pid and stops it when a signal ends the script.
set +m
python3 -m witness_agent.evaluate --base-url "http://127.0.0.1:$port/v1" --out "$out" "$@" &
evaluator=$!
status=0
wait "$evaluator" || status=$?
evaluator=
exit "$status"
