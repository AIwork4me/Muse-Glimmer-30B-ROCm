#!/usr/bin/env bash
# Run the GGUF benchmark matrix against the side-by-side ROCm 7.14.0 install
# (~/rocm-7.14.0, official stable gfx1151 tarball) and the llama.cpp build linked
# against it (third_party/llama.cpp/build-714/). Writes cells to
# docs/results/matrix-714/. Everything else — source commit, flags, weights, prompt
# set, seeds, reps — is identical to the 7.2.1 run; ONLY ROCm differs.
#
# Scope (decided 2026-08-13): skip ALL c=16 cells on this first pass (hard-freeze
# risk #6165 under sustained load on a newer ROCm). Override with EXCLUDE_NPS=""
# to run c=16 too (add it back only if this pass is stable).
#
# Usage: run-gguf-matrix-714.sh [--dry-run] [study1|study2|study3|all]
#   --dry-run is invaluable to confirm the reduced cell count (17) before running.
#
# Non-destructive: 7.2.1 (/opt/rocm) stays the system default; this only puts the
# 7.14.0 prefix first in THIS process's PATH/LD_LIBRARY_PATH. Revert by not running it.
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"; cd "$HERE"

ROCM_PREFIX="${ROCM_PREFIX:-$HOME/rocm-7.14.0}"
if [ ! -x "$ROCM_PREFIX/bin/hipcc" ]; then
  echo "ERROR: $ROCM_PREFIX/bin/hipcc not found — install 7.14.0 first (see the Part 2 plan, S1)." >&2
  exit 1
fi
if [ ! -x "$HERE/third_party/llama.cpp/build-714/bin/llama-server" ]; then
  echo "ERROR: build-714 server not found — rebuild llama.cpp against 7.14.0 first (plan S3)." >&2
  exit 1
fi

export PATH="$ROCM_PREFIX/bin:$HOME/.local/bin:$PATH"
export LD_LIBRARY_PATH="$ROCM_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export LLAMA_BIN="$HERE/third_party/llama.cpp/build-714/bin/llama-server"
export MATRIX_OUTDIR="docs/results/matrix-714"
export EXCLUDE_NPS="${EXCLUDE_NPS:-16}"   # skip c=16 on first pass (freeze risk)

HIPV=$(hipcc --version 2>/dev/null | grep -iE 'HIP version' || true)
echo "=== ROCm 7.14.0 matrix run ==="
echo "  ROCM_PREFIX  = $ROCM_PREFIX"
echo "  hipcc        = ${HIPV:-unknown}"
echo "  LLAMA_BIN    = $LLAMA_BIN"
echo "  MATRIX_OUTDIR= $MATRIX_OUTDIR"
echo "  EXCLUDE_NPS  = '$EXCLUDE_NPS'  (empty = run c=16 too)"
echo

exec bash "$HERE/scripts/run-gguf-matrix.sh" "$@"
