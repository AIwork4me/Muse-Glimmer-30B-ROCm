#!/usr/bin/env bash
# Method 2 (bare metal, NO Docker) — reproduce via python/shell on the host.
#
# You provide a llama.cpp `llama-server` built for gfx1100 (llama.cpp >= b10353,
# which added the muse-glimmer arch + DFlash) via LLAMA_BIN_HOST. The SAME driver
# then runs directly on the host — no container.
#
# Host prereqs: ROCm runtime + a gfx1100 `llama-server`; python3 with `aiohttp`;
# `curl`; models fetched via 00_prepare.sh.
#
#   LLAMA_BIN_HOST=/path/to/llama.cpp/build/bin/llama-server  bash run_host.sh
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"; . "$HERE/config.env"
mkdir -p "$OUT_HOST"

LLAMA_BIN="${LLAMA_BIN:-$LLAMA_BIN_HOST}"
{ [ -n "$LLAMA_BIN" ] && [ -x "$LLAMA_BIN" ]; } \
  || { echo "ERROR: set LLAMA_BIN_HOST to your gfx1100 llama-server binary (llama.cpp >= b10353)."; exit 1; }
[ -f "$REPO_DIR/scripts/gguf-bench-cell.sh" ] || { echo "ERROR: harness not found under $REPO_DIR/scripts"; exit 1; }
[ -f "$MODELS_HOST/muse-glimmer-30B-kquant-dynamic.gguf" ] || { echo "ERROR: models missing — run: bash 00_prepare.sh"; exit 1; }
command -v curl >/dev/null || { echo "ERROR: curl required"; exit 1; }
command -v python3 >/dev/null || { echo "ERROR: python3 required"; exit 1; }
python3 -c 'import aiohttp' 2>/dev/null || { echo "ERROR: python3 'aiohttp' required (pip install --user aiohttp)"; exit 1; }
if command -v uv >/dev/null 2>&1 && [ "$(command -v uv)" != "$HERE/_uvshim/uv" ]; then
  echo "NOTE: a real 'uv' is on PATH; the harness will prefer it. Ensure its env has aiohttp,"
  echo "      or remove it so the bundled shim (-> python3) is used."
fi

echo "=== Method 2 bare-metal reproduction (host) ==="
echo "  llama-server = $LLAMA_BIN"
REPO_DIR="$REPO_DIR" \
GLUE_DIR="$HERE" \
MATRIX_OUTDIR="$OUT_HOST" \
LLAMA_BIN="$LLAMA_BIN" \
STUDY_CONF="$STUDY_CONF" \
IMAGE_TAG="host-llama.cpp" \
RESUME="${RESUME:-1}" \
  bash "$HERE/_repro_driver.sh"

echo "Results: $OUT_HOST/"
