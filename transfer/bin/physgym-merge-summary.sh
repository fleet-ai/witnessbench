#!/usr/bin/env bash
# physgym-merge-summary.sh <merged-tag> <tag1> [<tag2> ...] — merge several physgym-eval.sh campaign directories that hold
# DISJOINT repetition numbers (e.g. main-r12 reps 1 2, main-r34 reps 3 4, main-r5 rep 5) into one root via symlinks
# (runs/<arm>/rep<k> -> ../../../<tag>/runs/<arm>/rep<k>) and run physgym_summary.py over the merged root.
# Refuses to merge if two tags carry the same (arm, rep).
set -uo pipefail
TW="${TRANSFER_WORK:-$(cd "$(dirname "$0")/.." && pwd)/work}"
ROOT="${ROOT:-$TW/results/physgym}"
VENV="${VENV:-$TW/venv-physgym}"
SUMMARY_PY="${SUMMARY_PY:-$(cd "$(dirname "$0")" && pwd)/physgym_summary.py}"
[ $# -ge 2 ] || { echo "usage: $0 <merged-tag> <tag1> [<tag2> ...]" >&2; exit 1; }
M="$ROOT/$1"; shift
mkdir -p "$M/runs"; : > "$M/MERGED_FROM.txt"
for t in "$@"; do
  [ -d "$ROOT/$t/runs" ] || { echo "FAILED: $ROOT/$t/runs missing" >&2; exit 1; }
  for r in "$ROOT/$t"/runs/*/rep*; do
    arm=$(basename "$(dirname "$r")"); rep=$(basename "$r")
    mkdir -p "$M/runs/$arm"
    if [ -e "$M/runs/$arm/$rep" ]; then echo "FAILED: $arm/$rep present twice ($(readlink "$M/runs/$arm/$rep") and $r)" >&2; exit 1; fi
    ln -s "$r" "$M/runs/$arm/$rep"
    echo "$arm/$rep <- $t" >> "$M/MERGED_FROM.txt"
  done
done
cat "$M/MERGED_FROM.txt"
ARGS=(--root "$M"); [ -n "${STEP0:-}" ] && ARGS+=(--step0 "$STEP0"); [ -n "${STEP10:-}" ] && ARGS+=(--step10 "$STEP10")
"$VENV/bin/python" "$SUMMARY_PY" "${ARGS[@]}"
