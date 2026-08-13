#!/usr/bin/env bash
# Reproducible GGUF quick-start for gfx1151.
#
# Defaults use the exact llama.cpp and GGUF revisions recorded in
# configs/validated-stack.json and verify every selected model artifact.
# Expert overrides are explicit and experimental:
#   LLAMA_CPP_REF=master GGUF_REVISION=main bash scripts/gguf-quickstart.sh
#
# Optional features:
#   WITH_MMPROJ=1  fetch and attach the validated vision projector
#   WITH_DFLASH=1  fetch and enable the validated DFlash drafter
set -euo pipefail

HERE="$(cd "$(dirname "$0")/.." && pwd)"
cd "$HERE"
export PATH="$HOME/.local/bin:$PATH"

# ROCm toolchain: the official 7.14.0 gfx1151 install at ~/rocm-7.14.0
# (set up by scripts/install-rocm-7.14.sh) is the DEFAULT; if it is absent we
# fall back to the system /opt/rocm (7.2.1, the historical reference). Override
# explicitly with ROCM_PREFIX=/path.
if [ -n "${ROCM_PREFIX:-}" ]; then
    : # explicit override
elif [ -x "$HOME/rocm-7.14.0/bin/hipcc" ]; then
    ROCM_PREFIX="$HOME/rocm-7.14.0"
else
    ROCM_PREFIX="/opt/rocm"
fi
if [ ! -x "$ROCM_PREFIX/bin/hipcc" ]; then
    echo "ERROR: no hipcc at $ROCM_PREFIX/bin/hipcc." >&2
    [ "$ROCM_PREFIX" = "$HOME/rocm-7.14.0" ] && \
        echo "       Install it: bash scripts/install-rocm-7.14.sh   (or ROCM_PREFIX=/opt/rocm to use 7.2.1)" >&2
    exit 1
fi
export PATH="$ROCM_PREFIX/bin:$PATH"
export LD_LIBRARY_PATH="$ROCM_PREFIX/lib:${LD_LIBRARY_PATH:-}"

for cmd in cmake curl git python3; do
    command -v "$cmd" >/dev/null 2>&1 || {
        echo "ERROR: required command not found: $cmd" >&2
        exit 1
    }
done

stack_value() {
    python3 - "$HERE/configs/validated-stack.json" "$1" <<'PY'
import json, sys
value = json.load(open(sys.argv[1]))
for key in sys.argv[2].split('.'):
    value = value[key]
print(value)
PY
}

LLAMA="$HERE/third_party/llama.cpp"
LLAMA_CPP_REPO="${LLAMA_CPP_REPO:-$(stack_value llama_cpp.source_repo)}"
VALIDATED_LLAMA_CPP_REF="$(stack_value llama_cpp.commit)"
LLAMA_CPP_REF="${LLAMA_CPP_REF:-$VALIDATED_LLAMA_CPP_REF}"
# Default build dir tracks the selected ROCm so a 7.14 build and a 7.2.1 build
# never clobber each other (build-714 vs build). LLAMA_CPP_BUILD_DIR overrides.
case "$ROCM_PREFIX" in
    */rocm-7.14.0) DEFAULT_BUILD_DIR="$LLAMA/build-714" ;;
    *)             DEFAULT_BUILD_DIR="$LLAMA/build" ;;
esac
BUILD_DIR="${LLAMA_CPP_BUILD_DIR:-$DEFAULT_BUILD_DIR}"
BUILD_STAMP="$BUILD_DIR/.muse-llama-ref"

GGUF_REPO="$(stack_value model.gguf_id)"
VALIDATED_GGUF_REVISION="$(stack_value model.gguf_revision)"
GGUF_REVISION="${GGUF_REVISION:-$VALIDATED_GGUF_REVISION}"
GGUF_FILE="${GGUF_FILE:-muse-glimmer-30B-kquant-17gb.gguf}"
MMPROJ_FILE="mmproj-kquant.gguf"
DFLASH_FILE="dflash-kquant.gguf"
DEST="${MODEL_DEST:-models}"
PORT="${PORT:-8080}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}"

echo "llama.cpp source: $LLAMA_CPP_REPO"
echo "llama.cpp ref   : $LLAMA_CPP_REF"
if [ "$LLAMA_CPP_REF" = "$VALIDATED_LLAMA_CPP_REF" ]; then
    echo "llama.cpp track : validated reference"
else
    echo "llama.cpp track : latest/experimental override"
fi
echo "HF endpoint     : $HF_ENDPOINT"
echo "GGUF revision   : $GGUF_REVISION"
if [ "$GGUF_REVISION" = "$VALIDATED_GGUF_REVISION" ]; then
    echo "GGUF track      : validated reference (hash verification enabled)"
else
    echo "GGUF track      : latest/experimental override"
fi

# Clone once, then fetch and detach at the selected ref on every run. Existing
# uncommitted llama.cpp changes are never deleted; checkout fails instead.
if [ ! -d "$LLAMA/.git" ]; then
    if [ -e "$LLAMA" ] && [ -n "$(find "$LLAMA" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]; then
        echo "ERROR: $LLAMA exists but is not a git checkout; move it aside first." >&2
        exit 1
    fi
    mkdir -p "$HERE/third_party"
    git clone --filter=blob:none --no-checkout "$LLAMA_CPP_REPO" "$LLAMA"
fi
CURRENT_LLAMA_CPP_COMMIT="$(git -C "$LLAMA" rev-parse HEAD 2>/dev/null || true)"
if [ "$CURRENT_LLAMA_CPP_COMMIT" != "$LLAMA_CPP_REF" ]; then
    if ! git -C "$LLAMA" diff --quiet --ignore-submodules HEAD -- ||
       ! git -C "$LLAMA" diff --cached --quiet; then
        echo "ERROR: $LLAMA has tracked changes; refusing to change commits." >&2
        exit 1
    fi
    git -C "$LLAMA" fetch --depth 1 "$LLAMA_CPP_REPO" "$LLAMA_CPP_REF"
    git -C "$LLAMA" checkout --detach FETCH_HEAD
elif ! git -C "$LLAMA" diff --quiet --ignore-submodules HEAD -- ||
     ! git -C "$LLAMA" diff --cached --quiet; then
    echo "ERROR: validated llama.cpp checkout has tracked modifications." >&2
    exit 1
else
    echo "llama.cpp checkout already at validated commit; no fetch needed"
fi
ACTUAL_LLAMA_CPP_COMMIT="$(git -C "$LLAMA" rev-parse HEAD)"
if [[ "$LLAMA_CPP_REF" =~ ^[0-9a-fA-F]{40}$ ]] &&
   [ "$ACTUAL_LLAMA_CPP_COMMIT" != "$LLAMA_CPP_REF" ]; then
    echo "ERROR: requested $LLAMA_CPP_REF but checked out $ACTUAL_LLAMA_CPP_COMMIT" >&2
    exit 1
fi
echo "llama.cpp commit: $ACTUAL_LLAMA_CPP_COMMIT"

previous_build_commit=""
[ -f "$BUILD_STAMP" ] && previous_build_commit="$(<"$BUILD_STAMP")"
if [ ! -x "$BUILD_DIR/bin/llama-server" ] ||    [ "$previous_build_commit" != "$ACTUAL_LLAMA_CPP_COMMIT" ]; then
    echo "Building llama.cpp (HIP, gfx1151) ..."
    cmake -S "$LLAMA" -B "$BUILD_DIR"         -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx1151 -DCMAKE_BUILD_TYPE=Release \
        -DROCM_PATH="$ROCM_PREFIX" -Dhip_DIR="$ROCM_PREFIX/lib/cmake/hip"
    cmake --build "$BUILD_DIR" -j "${BUILD_JOBS:-$(nproc)}"
    printf '%s\n' "$ACTUAL_LLAMA_CPP_COMMIT" > "$BUILD_STAMP"
else
    echo "llama.cpp build already matches $ACTUAL_LLAMA_CPP_COMMIT"
fi

mkdir -p "$DEST"
is_recorded_artifact() {
    python3 - "$HERE/configs/artifact-manifest.json" "$1" <<'PY'
import json, sys
files = json.load(open(sys.argv[1]))["sets"]["gguf"]["files"]
raise SystemExit(0 if any(item["path"] == sys.argv[2] for item in files) else 1)
PY
}

fetch_file() {
    local filename="$1"
    local can_verify=0
    if [ "$GGUF_REVISION" = "$VALIDATED_GGUF_REVISION" ] &&        is_recorded_artifact "$filename"; then
        can_verify=1
    fi

    if [ ! -f "$DEST/$filename" ]; then
        echo "fetching $filename via $HF_ENDPOINT ..."
        if [ "${USE_HF_DOWNLOAD:-0}" = "1" ]; then
            command -v uv >/dev/null 2>&1 || {
                echo "ERROR: uv is required with USE_HF_DOWNLOAD=1" >&2
                exit 1
            }
            uv run --no-sync hf download "$GGUF_REPO" "$filename"                 --revision "$GGUF_REVISION" --local-dir "$DEST"
        else
            python3 "$HERE/scripts/hf_parallel_get.py" "$GGUF_REPO" "$filename"                 --revision "$GGUF_REVISION" --local-dir "$DEST"                 --concurrency "${NCONNS:-24}"
        fi
    else
        echo "have $filename"
    fi

    if [ "$can_verify" -eq 1 ]; then
        if ! python3 "$HERE/scripts/verify_artifacts.py" gguf "$DEST" "$filename"; then
            suffix="$(date +%s)"
            quarantine="$DEST/$filename.corrupt.$suffix"
            mv "$DEST/$filename" "$quarantine"
            [ ! -e "$DEST/$filename.parts.json" ] ||
                mv "$DEST/$filename.parts.json" "$DEST/$filename.parts.json.invalid.$suffix"
            echo "ERROR: invalid artifact quarantined at $quarantine; rerun to fetch it." >&2
            exit 1
        fi
    else
        echo "WARNING: $filename is outside the validated artifact set; hash not asserted." >&2
    fi
}

fetch_file "$GGUF_FILE"
SERVER_ARGS=(-m "$DEST/$GGUF_FILE")
if [ "${WITH_MMPROJ:-0}" = "1" ]; then
    fetch_file "$MMPROJ_FILE"
    SERVER_ARGS+=(--mmproj "$DEST/$MMPROJ_FILE")
fi
if [ "${WITH_DFLASH:-0}" = "1" ]; then
    fetch_file "$DFLASH_FILE"
    SERVER_ARGS+=(
        -md "$DEST/$DFLASH_FILE" -ngld 99
        --spec-type draft-dflash --spec-draft-n-max 16
    )
fi

if ! python3 - "$PORT" <<'PY_PORT'
import socket
import sys
sock = socket.socket()
try:
    sock.bind(("127.0.0.1", int(sys.argv[1])))
finally:
    sock.close()
PY_PORT
then
    echo "ERROR: port $PORT is already in use; choose PORT=<free-port>." >&2
    exit 1
fi

echo "Serving on http://127.0.0.1:$PORT ..."
exec "$BUILD_DIR/bin/llama-server" "${SERVER_ARGS[@]}"     -ngl 999 -c 32768 --port "$PORT" --jinja
