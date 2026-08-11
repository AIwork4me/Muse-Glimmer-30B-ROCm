#!/usr/bin/env bash
# Throughput + peak-VRAM benchmark against a running OpenAI-compatible server
# (vLLM :8000 by default; point BASE at llama.cpp :8080 to compare). Writes a
# JSON record to docs/results/ for each run (gitignored runtime artifacts).
#
# Usage:  BASE=http://127.0.0.1:8000 bash scripts/benchmark.sh
set -euo pipefail

HERE="$(cd "$(dirname "$0")/.." && pwd)"
cd "$HERE"
export PATH="$HOME/.local/bin:$PATH"

BASE="${BASE:-http://127.0.0.1:8000}"
STAMP=$(date +%s)
OUT="docs/results/$(basename "$BASE" | tr : -)-${STAMP}.json"
mkdir -p docs/results

echo "Benchmarking $BASE (concurrency 1, 4, 16; 512 out tokens each) ..."
declare -a ROWS=()
for C in 1 4 16; do
    echo "  c=$C ..."
    ROWS+=("$(uv run --no-sync python "$HERE/scripts/bench_client.py" "$BASE" "$C" 512)")
done

# Peak VRAM snapshot across all GPUs (rocm-smi JSON, or {} if unavailable).
VRAM="$(rocm-smi --showmeminfo vram --json 2>/dev/null || echo '{}')"

printf '{"engine_base":"%s","timestamp":%s,"vram_peak":%s,"runs":[%s]}\n' \
    "$BASE" "$STAMP" "$VRAM" "$(IFS=,; echo "${ROWS[*]}")" | tee "$OUT"
echo "wrote $OUT"
