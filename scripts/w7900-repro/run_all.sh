#!/usr/bin/env bash
# Method 1 (image) — reproduce the W7900 Study-2 matrix end-to-end in a container.
# Long (~2-3 h). Recommended:  nohup bash run_all.sh > run_all.out 2>&1 &
#
# (Re)starts an isolated single-W7900 container with the repo mounted at
# /workspace, then runs _repro_driver.sh inside it. Results (idempotent,
# resumable) land in _out/matrix-w7900-gfx1100/. Overrides come from config.env.
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"; . "$HERE/config.env"
mkdir -p "$OUT_HOST"

# --- preconditions ---
[ -f "$MODELS_HOST/muse-glimmer-30B-kquant-dynamic.gguf" ] || { echo "models missing — run:  bash 00_prepare.sh"; exit 1; }
[ -f "$REPO_DIR/scripts/gguf-bench-cell.sh" ] || { echo "harness not found under $REPO_DIR/scripts"; exit 1; }
command -v docker >/dev/null || { echo "docker not found"; exit 1; }

# --- (re)create the isolated container (single W7900) ---
echo "=== (re)start container '$CONTAINER' (single W7900, HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-0}) ==="
docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
# shellcheck disable=SC2046
docker run -d --name "$CONTAINER" $(rocm_device_flags) \
  -v "$REPO_DIR":"$WORKROOT_CONT" -w "$WORKROOT_CONT" \
  --entrypoint sleep "$IMAGE" infinity

echo "=== GPU preflight (gfx1100) ==="
docker exec "$CONTAINER" bash -lc 'rocminfo 2>/dev/null | grep -qm1 gfx1100' \
  || { echo "ERROR: gfx1100 (W7900) not visible in the container"; exit 1; }

# --- run the reproduction inside the container ---
echo "=== reproducing 12 cells (Study-2 matrix + c=32) ==="
docker exec \
  -e REPO_DIR="$WORKROOT_CONT" \
  -e GLUE_DIR="$WORKROOT_CONT/$GLUE_REL" \
  -e MATRIX_OUTDIR="$MATRIX_OUTDIR_CONT" \
  -e LLAMA_BIN="$LLAMA_BIN_CONT" \
  -e STUDY_CONF="$STUDY_CONF" \
  -e IMAGE_TAG="$IMAGE" \
  -e RESUME="${RESUME:-1}" \
  "$CONTAINER" bash "$WORKROOT_CONT/$GLUE_REL/_repro_driver.sh"

echo
echo "Results (host): $OUT_HOST/"
echo "  headline.md  matrix.md  cell-study2-*.json (12)  llama-server-version.txt"
echo "Tear down:  bash 99_teardown.sh"
