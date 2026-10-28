#!/usr/bin/env bash
# setup.sh — prepare the four out-of-domain transfer benchmarks exactly as they were run for the paper:
#   clone each public repo, check out the pinned commit, apply transfer/patches/<bench>.diff, verify the md5 of every
#   patched file, fetch CipherBank's data at its pinned Hugging Face revision, and build the two Python environments.
# Usage: bash transfer/setup.sh            (everything goes to transfer/work/, or $TRANSFER_WORK if set)
# Needs: git, curl, python3.12 with venv. Recommended: run inside the SGLang image the paper used (see README), whose
#        system Python already carries numpy/tqdm/pyyaml/Levenshtein; the eval venv is created on top of it.
set -euo pipefail
T=$(cd "$(dirname "$0")" && pwd)
W=${TRANSFER_WORK:-$T/work}; mkdir -p "$W/repos"
log(){ echo "$(date -u +%FT%TZ) $*"; }
md5f(){ if command -v md5sum >/dev/null; then md5sum "$1" | cut -d' ' -f1; else md5 -q "$1"; fi; }

# bench | repo URL | pinned commit | checkout dir under work/repos
PINS="cipherbank https://github.com/Leey21/CipherBank.git 05b450ca8f11b966fb73d06278dea213e87540f9 CipherBank
wilt https://github.com/riotgames/wilt.git 6344f04b3eb9119664c87c4d1b6cd2a227f70e8d wilt
failing-to-falsify https://github.com/ayushjhaveri/llm-confirmation-bias.git 97f64338f88e75cdf3f5572ee6cd899f8b95c2b9 llm-confirmation-bias
physgym https://github.com/principia-ai/PhysGym.git fe68079c0921029dde679ed44bc3192dd3b270ab PhysGym"

echo "$PINS" | while read -r bench url sha dir; do
  r="$W/repos/$dir"; p="$T/patches/$bench.diff"
  [ -d "$r/.git" ] || { log "clone $url"; git clone --quiet "$url" "$r"; }
  git -C "$r" checkout --quiet "$sha"
  if git -C "$r" apply --check --reverse "$p" 2>/dev/null; then log "$bench: patch already applied"
  elif git -C "$r" apply --check "$p" 2>/dev/null; then git -C "$r" apply "$p"; log "$bench: patch applied"
  else echo "$bench: $p neither applied nor applicable to $r (local changes?)"; git -C "$r" status --short; exit 3; fi
done

# md5 of every patched file must match the files that produced the paper's numbers
dir_of(){ case $1 in cipherbank) echo CipherBank;; wilt) echo wilt;; f2f) echo llm-confirmation-bias;; physgym) echo PhysGym;; esac; }
bad=0
while read -r bench rel want; do
  got=$(md5f "$W/repos/$(dir_of "$bench")/$rel")
  [ "$got" = "$want" ] || { echo "MD5 MISMATCH $bench/$rel: $got != $want"; bad=1; }
done < "$T/patched_md5.txt"
[ $bad = 0 ] || exit 3; log "all patched files match patched_md5.txt"

# eval venv (CipherBank, WILT, Failing-to-Falsify): the paper's runs used the image's Python 3.12 + these packages
if [ ! -x "$W/venv/bin/python" ]; then
  python3 -m venv --system-site-packages "$W/venv"
  "$W/venv/bin/pip" install --quiet openai==1.109.1 Levenshtein==0.27.5 rapidfuzz==3.14.6 tqdm==4.70.0 pyyaml==6.0.1 numpy==2.3.5
fi
"$W/venv/bin/python" -c 'import openai, Levenshtein, tqdm, numpy, yaml; print("eval venv ok: openai", openai.__version__, "numpy", numpy.__version__)'

# PhysGym venv: its own isolated environment (versions frozen from the paper's run)
if [ ! -x "$W/venv-physgym/bin/python" ]; then
  python3 -m venv "$W/venv-physgym"
  "$W/venv-physgym/bin/pip" install --quiet -r "$T/requirements-physgym.txt"
  "$W/venv-physgym/bin/pip" install --quiet --no-deps -e "$W/repos/PhysGym"
fi
"$W/venv-physgym/bin/python" -c 'import physgym, openai, sympy; print("physgym venv ok: openai", openai.__version__, "sympy", sympy.__version__)'

# CipherBank data: 9 files at the pinned HF dataset revision, md5-checked, merged into the repo's data layout
REPO="$W/repos/CipherBank" PY="$W/venv/bin/python" bash "$T/bin/launch_cipherbank.sh" setup
log "setup done: $W"
