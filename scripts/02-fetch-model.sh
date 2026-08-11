#!/usr/bin/env bash
# Download Meta's Muse-Glimmer-30B (BF16, ~55 GiB weights) into ./models/.
# Apache 2.0, NOT gated — no HF token needed. Two safetensors shards
# (~46.5 GiB + ~9.0 GiB); expect a multi-hour download on a slow CDN link.
#
# Two-phase fetch:
#   1. `hf download --exclude '*.safetensors'`  — small files (config, tokenizer,
#      chat template, license). Tiny; the stock tool is fine here.
#   2. scripts/hf_parallel_get.py                — the two weight shards via
#      N-way parallel HTTP range requests (see that file for why the stock tool
#      is single-stream and painfully slow per connection, ~0.2 MiB/s).
#
# Critical: every `uv run` MUST pass --no-sync. vLLM was source-installed
# (Task 3, editable, --no-build-isolation) and is NOT tracked by uv.lock; a bare
# `uv run` would re-sync and DELETE the editable vLLM. See pyproject.toml.
#
# CLI note: huggingface_hub >=0.28 deprecated `huggingface-cli`; the venv has
# 1.27.0, so we invoke `hf download` directly.
#
# Mirror: goes through https://hf-mirror.com (HF_ENDPOINT) by default.
# huggingface.co is slow/blocked from this host. NOTE: the mirror only proxies
# the HF API + /resolve metadata — LFS blobs 302-redirect to the SAME signed
# CloudFront URL (us.aws.cdn.hf.co) either way, so the mirror does not speed up
# the weights; hf_parallel_get.py's parallelism is what does. Override the
# endpoint with HF_ENDPOINT. The mirror has no xet CAS server, so xet must stay
# off for the small-file phase (HF_HUB_DISABLE_XET=1).
#
# Set USE_HF_DOWNLOAD=1 to skip the parallel downloader and use the stock
# single-stream `hf download` for everything (much slower here, but simpler and
# dependency-free if hf_parallel_get.py is unavailable).
#
# Resumable: both phases pick up where they left off. Retry loop for transient
# network blips.
set -euo pipefail

MODEL_ID="meta-models/Muse-Glimmer-30B"   # BF16, Apache 2.0, NOT gated
DEST="models/Muse-Glimmer-30B"
SHARDS=(model-00001-of-00002.safetensors model-00002-of-00002.safetensors)

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# uv lives in ~/.local/bin on this host.
export PATH="$HOME/.local/bin:$PATH"
# Mirror endpoint (overridable) + classic HTTP for the small-file phase.
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_DISABLE_XET=1
export NCONNS="${NCONNS:-24}"   # parallel range connections for the shards

mkdir -p models

# Preflight: fail fast with a clear message if the endpoint is unreachable.
if ! curl -sS -o /dev/null --max-time 20 \
     "${HF_ENDPOINT}/meta-models/Muse-Glimmer-30B/resolve/main/config.json"; then
    echo "ERROR: cannot reach HF_ENDPOINT=$HF_ENDPOINT (model resolve)." >&2
    echo "       Set HF_ENDPOINT to a working mirror or direct huggingface.co." >&2
    exit 1
fi

echo "Fetching $MODEL_ID -> $DEST"
echo "  endpoint  : $HF_ENDPOINT"
echo "  small files: hf download (classic HTTP)"
if [ "${USE_HF_DOWNLOAD:-0}" = "1" ]; then
    echo "  weights   : hf download (single-stream; set USE_HF_DOWNLOAD=0 for parallel)"
else
    echo "  weights   : hf_parallel_get.py ($NCONNS parallel range connections)"
fi
START=$(( $(date +%s) ))

run_with_retries() {  # $1 = max attempts, rest = command
    local n="$1"; shift
    local i=0
    until "$@"; do
        i=$(( i + 1 )); [ "$i" -ge "$n" ] && { echo "ERROR: failed after $n attempts" >&2; return 1; }
        echo "  attempt $i failed; retrying in 10s (resumable)..." >&2; sleep 10
    done
}

# Phase 1: small files (everything but the weight shards).
run_with_retries 10 \
    uv run --no-sync hf download "$MODEL_ID" --local-dir "$DEST" --exclude "*.safetensors"

# Phase 2: the two weight shards.
if [ "${USE_HF_DOWNLOAD:-0}" = "1" ]; then
    run_with_retries 10 uv run --no-sync hf download "$MODEL_ID" \
        --local-dir "$DEST" --include "*.safetensors"
else
    run_with_retries 10 \
        uv run --no-sync python scripts/hf_parallel_get.py "$MODEL_ID" \
            "${SHARDS[@]}" --local-dir "$DEST" --concurrency "$NCONNS"
fi

ELAPSED=$(( $(date +%s) - START ))
BYTES=$(du -sb "$DEST" | cut -f1)
GB=$(awk -v b="$BYTES" 'BEGIN { printf "%.2f", b / 1024 / 1024 / 1024 }')
echo "OK: model at $DEST"
echo "Summary: ${GB} GiB, ${ELAPSED}s elapsed"
