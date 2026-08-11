#!/usr/bin/env bash
# GGUF quick-start: build llama.cpp for gfx1151 (once), fetch Meta's official
# K-quant GGUF, and serve it with llama-server. This is the "chat in minutes,
# no vLLM compile" path — independent of the vLLM venv (needs only system
# python3 + curl + cmake/g++).
#
# Meta ships its own calibrated quants under meta-models/Muse-Glimmer-30B-GGUF
# (custom "kquant" names, not standard llama.cpp Q4_K_M):
#   muse-glimmer-30B-kquant-17gb.gguf     ~17 GiB, fits 24/32 GB envelopes (default)
#   muse-glimmer-30B-kquant-dynamic.gguf  ~under 20 GiB, higher fidelity
#   mmproj-kquant.gguf                    multimodal projector (vision)
#   dflash-kquant.gguf                    DFlash speculative-drafter
#
# Override the file with GGUF_FILE=... ; enable vision with WITH_MMProj=1.
# Downloads go through https://hf-mirror.com (HF_ENDPOINT) via the project's
# parallel range downloader (scripts/hf_parallel_get.py) — same slow-CDN logic
# as the BF16 fetch; set USE_HF_DOWNLOAD=1 for the stock single-stream tool.
set -euo pipefail

HERE="$(cd "$(dirname "$0")/.." && pwd)"
cd "$HERE"
export PATH="$HOME/.local/bin:$PATH"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

LLAMA="$HERE/third_party/llama.cpp"
GGUF_REPO="meta-models/Muse-Glimmer-30B-GGUF"
GGUF_FILE="${GGUF_FILE:-muse-glimmer-30B-kquant-17gb.gguf}"
MMPROJ_FILE="mmproj-kquant.gguf"
DEST="models"
mkdir -p "$DEST"

# 1. Build llama.cpp for gfx1151 (once).
if [ ! -x "$LLAMA/build/bin/llama-server" ]; then
    echo "Building llama.cpp (HIP, gfx1151) ..."
    rm -rf "$LLAMA"
    git clone --depth 1 https://github.com/ggml-org/llama.cpp "$LLAMA"
    cmake -S "$LLAMA" -B "$LLAMA/build" \
        -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx1151 -DCMAKE_BUILD_TYPE=Release
    cmake --build "$LLAMA/build" -j
fi

# 2. Fetch the GGUF (+ mmproj if requested).
fetch_file() {  # $1 = filename
    local f="$1"
    if [ -f "$DEST/$f" ]; then echo "  have $f"; return; fi
    echo "  fetching $f via $HF_ENDPOINT ..."
    if [ "${USE_HF_DOWNLOAD:-0}" = "1" ]; then
        uv run --no-sync hf download "$GGUF_REPO" "$f" --local-dir "$DEST"
    else
        python3 "$HERE/scripts/hf_parallel_get.py" "$GGUF_REPO" "$f" \
            --local-dir "$DEST" --concurrency "${NCONNS:-24}"
    fi
}
fetch_file "$GGUF_FILE"
MMPROJ_ARGS=()
if [ "${WITH_MMProj:-0}" = "1" ]; then
    fetch_file "$MMPROJ_FILE"
    MMPROJ_ARGS=(--mmproj "$DEST/$MMPROJ_FILE")
fi

# 3. Serve. Text-focused quick-start; with WITH_MMProj=1 the vision projector is
#    attached (llama.cpp has first-class muse_glimmer arch support, so it loads
#    natively). Add -np <slots> (e.g. -np 16 -c 16384) to raise concurrent
#    throughput — the default 4 slots plateau at ~22 tok/s (see
#    docs/results/benchmark.md).
echo "Serving on http://127.0.0.1:8080 ..."
exec "$LLAMA/build/bin/llama-server" \
    -m "$DEST/$GGUF_FILE" "${MMPROJ_ARGS[@]}" \
    -ngl 999 -c 32768 --port 8080
