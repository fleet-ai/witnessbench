#!/usr/bin/env bash
# Serve a Hugging Face checkpoint with SGLang on one GPU for validation.
#
#   scripts/serve_policy.sh HF_DIR [PORT]
#
# The server listens on 127.0.0.1:PORT (default 30000) under the model name "witness-policy".
# --sampling-defaults openai makes a request that omits top_p/top_k sample with top_p 1.0 and
# top_k -1 instead of the checkpoint's generation_config values. The default chat-template kwargs
# render every prompt with thinking off. A prompt plus the 8192 completion tokens a request may
# ask for must fit in --context-length, and the 24 episodes of a default validation run (8 games
# x 3 seeds) run at once, under --max-running-requests. CUDA_VISIBLE_DEVICES selects the GPU.
set -euo pipefail
usage="usage: $0 HF_DIR [PORT]"
case "${1:-}" in
  -h | --help)
    echo "$usage"
    echo "Serve the Hugging Face checkpoint HF_DIR with SGLang on one GPU at 127.0.0.1:PORT (default 30000),"
    echo "under the model name witness-policy. CUDA_VISIBLE_DEVICES selects the GPU."
    exit 0 ;;
esac
if [ "$#" -lt 1 ] || [ "$#" -gt 2 ]; then
  echo "$usage" >&2
  exit 2
fi
exec python3 -m sglang.launch_server \
  --model-path "$1" \
  --served-model-name witness-policy \
  --tp 1 \
  --port "${2:-30000}" \
  --host 127.0.0.1 \
  --mem-fraction-static 0.85 \
  --context-length 32768 \
  --sampling-defaults openai \
  --default-chat-template-kwargs '{"enable_thinking": false}' \
  --max-running-requests 64
