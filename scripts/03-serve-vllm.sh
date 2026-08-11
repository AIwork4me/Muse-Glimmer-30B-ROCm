#!/usr/bin/env bash
# Launch vLLM (OpenAI-compatible server) for Muse-Glimmer-30B on gfx1151.
# Source of truth for the flags is configs/serve-args.conf; the runtime env is
# configs/vllm-gfx1151.env. Both are CI-checked by tests/test_serve_args.py.
#
# CRITICAL: uses `uv run --no-sync`. vLLM was source-installed editable
# (scripts/01-build-vllm.sh, --no-build-isolation) and is NOT in uv.lock. A bare
# `uv run` would re-sync and DELETE the editable vLLM. Never drop --no-sync here.
set -euo pipefail

HERE="$(cd "$(dirname "$0")/.." && pwd)"
cd "$HERE"

# uv lives in ~/.local/bin on this host.
export PATH="$HOME/.local/bin:$PATH"

# shellcheck source=/dev/null
source "$HERE/configs/vllm-gfx1151.env"

MODEL_DIR="${MODEL_DIR:-$HERE/models/Muse-Glimmer-30B}"
if [ ! -f "$MODEL_DIR/config.json" ]; then
    echo "ERROR: $MODEL_DIR/config.json missing — run scripts/02-fetch-model.sh first." >&2
    exit 1
fi

# Comments stripped from serve-args.conf; flags passed positionally.
# shellcheck disable=SC2046
exec uv run --no-sync vllm serve "$MODEL_DIR" \
    $(grep -v '^[[:space:]]*#' "$HERE/configs/serve-args.conf" | xargs)
