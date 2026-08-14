#!/usr/bin/env bash
# Optional confirmed one-command entry point for the ROCm 7.14 GGUF path.
set -euo pipefail

HERE="$(cd "$(dirname "$0")/.." && pwd)"
cd "$HERE"

# shellcheck source=scripts/lib/rocm.sh
source "$HERE/scripts/lib/rocm.sh"

assume_yes=0
while [ "$#" -gt 0 ]; do
    case "$1" in
        --yes) assume_yes=1 ;;
        -h|--help)
            echo "Usage: bash scripts/quickstart.sh [--yes]"
            echo "  --yes  show the plan, then continue without prompting"
            exit 0
            ;;
        *)
            echo "ERROR: unknown argument: $1" >&2
            echo "Usage: bash scripts/quickstart.sh [--yes]" >&2
            exit 2
            ;;
    esac
    shift
done

mapfile -t PLAN_VALUES < <(python3 - \
    "$HERE/configs/rocm-7.14-gguf-validation.json" \
    "$HERE/configs/artifact-manifest.json" <<'PY'
import json
import sys

rocm = json.load(open(sys.argv[1], encoding="utf-8"))
artifacts = json.load(open(sys.argv[2], encoding="utf-8"))
model = next(
    item for item in artifacts["sets"]["gguf"]["files"]
    if item["path"] == "muse-glimmer-30B-kquant-17gb.gguf"
)
print(rocm["host"]["rocm_version"])
print(rocm["host"]["archive"]["size_bytes"])
print(model["path"])
print(model["size_bytes"])
PY
)
ROCM_VERSION="${PLAN_VALUES[0]}"
ROCM_BYTES="${PLAN_VALUES[1]}"
MODEL_FILE="${PLAN_VALUES[2]}"
MODEL_BYTES="${PLAN_VALUES[3]}"
MODEL_DIR="${MODEL_DEST:-models}"
RECOMMENDED_PREFIX="$HOME/rocm-7.14.0"

gib() {
    python3 - "$1" <<'PY'
import sys
print(f"{int(sys.argv[1]) / 1024**3:.1f} GiB")
PY
}

install_rocm=0
if [ -n "${ROCM_PREFIX:-}" ] || [ -n "${ROCM_PATH:-}" ]; then
    resolve_rocm_prefix || exit 1
    selected_rocm="$(detect_rocm_version "$ROCM_PREFIX") at $ROCM_PREFIX (explicit override)"
elif rocm_prefix_is_valid "$RECOMMENDED_PREFIX"; then
    selected_rocm="$ROCM_VERSION at $RECOMMENDED_PREFIX (installed)"
else
    selected_rocm="$ROCM_VERSION at $RECOMMENDED_PREFIX (will install side-by-side)"
    install_rocm=1
fi

if [ -f "$MODEL_DIR/$MODEL_FILE" ]; then
    model_download="already present; exact size and SHA256 will be verified"
else
    model_download="$(gib "$MODEL_BYTES")"
fi
if [ "$install_rocm" -eq 1 ]; then
    rocm_download="$(gib "$ROCM_BYTES")"
else
    rocm_download="not needed"
fi

echo "Muse-Glimmer RDNA Quick Start"
echo
echo "ROCm:          $selected_rocm"
echo "Backend:       llama.cpp HIP / gfx1151"
echo "Model:         Muse-Glimmer-30B K-Quant"
echo "Model download: $model_download"
echo "ROCm download:  $rocm_download"
echo

if [ "$assume_yes" -ne 1 ]; then
    printf 'Continue? [y/N] '
    reply=""
    read -r reply || true
    case "$reply" in
        y|Y|yes|YES|Yes) ;;
        *)
            echo "Cancelled; no installer, environment check, or model download was started."
            exit 0
            ;;
    esac
fi

if [ "$install_rocm" -eq 1 ]; then
    bash "$HERE/scripts/install-rocm-7.14.sh"
fi
bash "$HERE/scripts/00-check-env.sh" --profile gguf
exec bash "$HERE/scripts/gguf-quickstart.sh"
