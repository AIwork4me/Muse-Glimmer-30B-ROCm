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

# Use the same selection policy and diagnostics as the environment checker.
# shellcheck source=scripts/lib/rocm.sh
source "$HERE/scripts/lib/rocm.sh"
# shellcheck source=scripts/lib/llama_build.sh
source "$HERE/scripts/lib/llama_build.sh"

# F-16/F-11: gate on the serving port before anything expensive - the plan
# header, ROCm resolution, tool checks, clone/checkout guards, build
# fingerprinting and artifact re-verification all used to run before this
# refusal, costing seconds on a warm tree and minutes on a rebuild-needed one.
# The probe needs python3; without it, fall through to the required-tools
# guard below instead of misreading a missing interpreter as a busy port.
PORT="${PORT:-8080}"
if command -v python3 >/dev/null 2>&1 && ! python3 - "$PORT" <<'PY_PORT'
import socket
import sys
sock = socket.socket()
try:
    sock.bind(("127.0.0.1", int(sys.argv[1])))
except OSError:
    # F-11: exit cleanly so the outer `if !` prints the actionable ERROR
    # line; an unhandled bind error used to leak a traceback right before it.
    sys.exit(1)
finally:
    sock.close()
PY_PORT
then
    echo "ERROR: port $PORT is already in use; choose PORT=<free-port>." >&2
    exit 1
fi

resolve_rocm_prefix || exit 1
rocm_ver="$(detect_rocm_version "$ROCM_PREFIX")"
print_selected_rocm "$rocm_ver"
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
AMDGPU_TARGET="gfx1151"
ROCM_BUILD_PREFIX="$(canonical_rocm_prefix "$ROCM_PREFIX")"
BUILD_DIR="$(llama_build_dir "$LLAMA" "$ROCM_PREFIX" "$rocm_ver" "${LLAMA_CPP_BUILD_DIR:-}")"
BUILD_FINGERPRINT="$BUILD_DIR/.muse-build-fingerprint.json"

GGUF_REPO="$(stack_value model.gguf_id)"
VALIDATED_GGUF_REVISION="$(stack_value model.gguf_revision)"
GGUF_REVISION="${GGUF_REVISION:-$VALIDATED_GGUF_REVISION}"
GGUF_FILE="${GGUF_FILE:-muse-glimmer-30B-kquant-17gb.gguf}"
MMPROJ_FILE="mmproj-kquant.gguf"
DFLASH_FILE="dflash-kquant.gguf"
DEST="${MODEL_DEST:-models}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}"

# F-03: refuse up front instead of dying mid-download (or mid-build) with a
# raw "No space left on device". Floors come from the artifact manifest the
# fetch step already verifies against:
#   - every selected artifact that is not yet present locally contributes its
#     manifest size_bytes (WITH_DFLASH/WITH_MMPROJ add the drafter/projector).
#     Artifacts already on disk are skipped: they occupy their space already,
#     and re-requiring it would lock warm reruns out of serving on any disk
#     that filled up after the download. If hash verification quarantines one,
#     the rerun sees it absent and this preflight re-engages, so a refetch can
#     never start without this check having passed.
#   - the llama.cpp checkout/build allowance covers the ~206 MiB validated
#     worktree plus the ~1022 MiB build-714 tree measured on the validated
#     stack, rounded up to 1.5 GiB for ref/build churn. It always applies:
#     a fingerprint change can force a rebuild on any run.
# models/ and the build dir can live on different filesystems, so each is
# checked against its own floor; when they share a mount the floors are
# combined and checked once (two separate checks against the same mount
# would undercount the shared space).
DISK_BUILD_ALLOWANCE_BYTES=$((512 * 1024 * 1024 * 3))

artifact_size_bytes() {  # artifact_size_bytes <filename> -> manifest size_bytes
    python3 - "$HERE/configs/artifact-manifest.json" "$1" <<'PY'
import json, sys
files = json.load(open(sys.argv[1]))["sets"]["gguf"]["files"]
sizes = {item["path"]: item["size_bytes"] for item in files}
default = "muse-glimmer-30B-kquant-17gb.gguf"
print(sizes.get(sys.argv[2], sizes.get(default, 0)))
PY
}

fs_of_dir() {  # fs_of_dir <dir> -> "<mount> <avail_bytes>" of the filesystem holding <dir>
    local dir="$1"
    # Walk up to the nearest existing ancestor: models/ and the build dir do
    # not exist yet this early in the run.
    while [ ! -d "$dir" ]; do
        dir="$(dirname "$dir")"
    done
    df -Pk "$dir" | awk 'NR==2 {print $NF, $4 * 1024}'
}

require_disk_bytes() {  # require_disk_bytes <what> <dir> <floor_bytes> <remedy>
    local what="$1" dir="$2" floor_bytes="$3" remedy="$4" mount avail
    read -r mount avail <<<"$(fs_of_dir "$dir")"
    if [ "$avail" -lt "$floor_bytes" ]; then
        printf 'ERROR: not enough disk space for %s: filesystem %s (holding %s) has %s GiB available, need at least %s GiB.\n' \
            "$what" "$mount" "$dir" \
            "$(awk -v b="$avail" 'BEGIN {printf "%.1f", b / 1073741824}')" \
            "$(awk -v b="$floor_bytes" 'BEGIN {printf "%.1f", b / 1073741824}')" >&2
        printf '       %s\n' "$remedy" >&2
        exit 1
    fi
}

DISK_MODEL_REMEDY="Free space on that filesystem, or set MODEL_DEST to a path on a larger filesystem, then rerun (MODEL_DEST also reuses the model across clones; see README)."
selected_artifacts=("$GGUF_FILE")
if [ "${WITH_MMPROJ:-0}" = "1" ]; then
    selected_artifacts+=("$MMPROJ_FILE")
fi
if [ "${WITH_DFLASH:-0}" = "1" ]; then
    selected_artifacts+=("$DFLASH_FILE")
fi
disk_model_floor_bytes=0
for artifact in "${selected_artifacts[@]}"; do
    if [ ! -f "$DEST/$artifact" ]; then
        artifact_bytes="$(artifact_size_bytes "$artifact")"
        disk_model_floor_bytes=$((disk_model_floor_bytes + artifact_bytes))
    fi
done

if [ "$(fs_of_dir "$DEST" | awk '{print $1}')" = "$(fs_of_dir "$BUILD_DIR" | awk '{print $1}')" ]; then
    require_disk_bytes "the model download and the llama.cpp checkout/build" \
        "$DEST" "$((disk_model_floor_bytes + DISK_BUILD_ALLOWANCE_BYTES))" \
        "$DISK_MODEL_REMEDY"
else
    require_disk_bytes "the model download" "$DEST" "$disk_model_floor_bytes" \
        "$DISK_MODEL_REMEDY"
    require_disk_bytes "the llama.cpp checkout/build" "$BUILD_DIR" \
        "$DISK_BUILD_ALLOWANCE_BYTES" \
        "Free space on that filesystem, then rerun."
fi

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
echo "llama.cpp build : $BUILD_DIR"

# Clone once, then fetch and detach at the selected ref on every run. Existing
# uncommitted llama.cpp changes are never deleted; the guard refuses instead.
if [ ! -d "$LLAMA/.git" ]; then
    if [ -e "$LLAMA" ] && [ -n "$(find "$LLAMA" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]; then
        echo "ERROR: $LLAMA exists but is not a git checkout; move it aside first." >&2
        exit 1
    fi
    mkdir -p "$HERE/third_party"
    git clone --filter=blob:none --no-checkout "$LLAMA_CPP_REPO" "$LLAMA"
    # F-04: this clone is seconds old and empty by construction (no index, no
    # worktree), so it cannot hold user changes. Reach the selected ref now,
    # before any dirty-tree guard can misread that state as every tracked
    # path staged-deleted and dead-end each cold start.
    git -C "$LLAMA" fetch --depth 1 "$LLAMA_CPP_REPO" "$LLAMA_CPP_REF"
    git -C "$LLAMA" checkout --detach FETCH_HEAD
fi
CURRENT_LLAMA_CPP_COMMIT="$(git -C "$LLAMA" rev-parse HEAD 2>/dev/null || true)"
if [ "$CURRENT_LLAMA_CPP_COMMIT" != "$LLAMA_CPP_REF" ]; then
    if llama_has_tracked_changes "$LLAMA"; then
        llama_refuse_dirty_checkout "$LLAMA" \
            "switch llama.cpp from $CURRENT_LLAMA_CPP_COMMIT to $LLAMA_CPP_REF"
        exit 1
    fi
    git -C "$LLAMA" fetch --depth 1 "$LLAMA_CPP_REPO" "$LLAMA_CPP_REF"
    git -C "$LLAMA" checkout --detach FETCH_HEAD
elif llama_has_tracked_changes "$LLAMA"; then
    llama_refuse_dirty_checkout "$LLAMA" \
        "reuse the llama.cpp checkout at $CURRENT_LLAMA_CPP_COMMIT"
    exit 1
else
    # Ref: Fix-A review cosmetic - this also prints right after the fresh-clone
    # branch above did its own fetch+checkout, so it must not claim no fetch
    # happened; state the checkout's actual condition instead.
    echo "llama.cpp checkout at $LLAMA_CPP_REF; working tree clean"
fi
ACTUAL_LLAMA_CPP_COMMIT="$(git -C "$LLAMA" rev-parse HEAD)"
if [[ "$LLAMA_CPP_REF" =~ ^[0-9a-fA-F]{40}$ ]] &&
   [ "$ACTUAL_LLAMA_CPP_COMMIT" != "$LLAMA_CPP_REF" ]; then
    echo "ERROR: requested $LLAMA_CPP_REF but checked out $ACTUAL_LLAMA_CPP_COMMIT" >&2
    exit 1
fi
echo "llama.cpp commit: $ACTUAL_LLAMA_CPP_COMMIT"

mkdir -p "$BUILD_DIR"
EXPECTED_BUILD_FINGERPRINT="$(mktemp "$BUILD_DIR/.muse-build-fingerprint.expected.XXXXXX")"
trap 'rm -f "$EXPECTED_BUILD_FINGERPRINT"' EXIT
write_llama_build_fingerprint "$EXPECTED_BUILD_FINGERPRINT" \
    "$ACTUAL_LLAMA_CPP_COMMIT" "$ROCM_PREFIX" "$rocm_ver" "$AMDGPU_TARGET"

if [ ! -x "$BUILD_DIR/bin/llama-server" ]; then
    echo "llama.cpp build missing; configuring HIP for $AMDGPU_TARGET"
    rebuild=1
elif ! llama_build_fingerprint_matches "$EXPECTED_BUILD_FINGERPRINT" "$BUILD_FINGERPRINT"; then
    echo "llama.cpp build fingerprint changed; reconfiguring HIP for $AMDGPU_TARGET"
    rebuild=1
else
    rebuild=0
fi

if [ "$rebuild" -eq 1 ]; then
    cmake -S "$LLAMA" -B "$BUILD_DIR" -DGGML_HIP=ON \
        -DAMDGPU_TARGETS="$AMDGPU_TARGET" -DCMAKE_BUILD_TYPE=Release \
        -DROCM_PATH="$ROCM_BUILD_PREFIX" \
        -Dhip_DIR="$ROCM_BUILD_PREFIX/lib/cmake/hip"
    cmake --build "$BUILD_DIR" -j "${BUILD_JOBS:-$(nproc)}"
    mv "$EXPECTED_BUILD_FINGERPRINT" "$BUILD_FINGERPRINT"
else
    echo "llama.cpp build fingerprint matches; no rebuild needed"
    rm -f "$EXPECTED_BUILD_FINGERPRINT"
fi
trap - EXIT

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

echo "Serving on http://127.0.0.1:$PORT ..."
exec "$BUILD_DIR/bin/llama-server" "${SERVER_ARGS[@]}"     -ngl 999 -c 32768 --port "$PORT" --jinja
