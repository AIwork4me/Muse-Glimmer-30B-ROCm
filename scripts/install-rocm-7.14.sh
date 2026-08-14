#!/usr/bin/env bash
# Install AMD's official ROCm 7.14.0 gfx1151 tarball side-by-side at
# ~/rocm-7.14.0 (default), WITHOUT touching the system /opt/rocm (7.2.1).
#
# URL, byte size and SHA256 are read from configs/rocm-7.14-gguf-validation.json
# so the manifest is the single source of truth (no hardcoded hash here).
#
# Idempotent: if ~/rocm-7.14.0/bin/hipcc already exists, this is a no-op.
# Usage: bash scripts/install-rocm-7.14.sh [ROCM714_PREFIX]
#   ROCM714_PREFIX=/path    override the install prefix
#   ROCM714_ARCHIVE=/path   override the download location
#   ROCM714_MANIFEST=/path  override the manifest (test seam for the harness)
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
MANIFEST="${ROCM714_MANIFEST:-$HERE/configs/rocm-7.14-gguf-validation.json}"

# F-12: the manifest reader (python3) and the downloader (curl) must exist
# before first use; without this check a bare OS dies inside read_field with a
# raw bash "command not found" instead of an actionable error.
for REQUIRED_TOOL in python3 curl; do
    if ! command -v "$REQUIRED_TOOL" >/dev/null 2>&1; then
        echo "ERROR: required command not found: $REQUIRED_TOOL" >&2
        echo "  Debian/Ubuntu:  sudo apt-get install $REQUIRED_TOOL" >&2
        echo "  Fedora/RHEL:    sudo dnf install $REQUIRED_TOOL" >&2
        echo "  Arch:           sudo pacman -S $REQUIRED_TOOL" >&2
        exit 1
    fi
done

read_field() {
    python3 - "$MANIFEST" "$1" <<'PY'
import json, sys
v = json.load(open(sys.argv[1]))
for k in sys.argv[2].split('.'):
    v = v[k]
print(v)
PY
}

PREFIX="${ROCM714_PREFIX:-${1:-$HOME/rocm-7.14.0}}"
URL="$(read_field host.archive.url)"
SIZE="$(read_field host.archive.size_bytes)"
SHA256="$(read_field host.archive.sha256)"
ROCM_VER="$(read_field host.rocm_version)"

if [ -x "$PREFIX/bin/hipcc" ]; then
    echo "ROCm $ROCM_VER already installed at $PREFIX (bin/hipcc present); nothing to do."
    exit 0
fi
if [ -e "$PREFIX" ]; then
    echo "ERROR: $PREFIX exists but has no bin/hipcc (incomplete install)." >&2
    echo "       Move it aside and rerun." >&2
    exit 1
fi

ARCHIVE="${ROCM714_ARCHIVE:-${TMPDIR:-/tmp}/therock-dist-linux-gfx1151-7.14.0.tar.gz}"
PARENT="$(dirname "$PREFIX")"
mkdir -p "$PARENT"

# F-03: refuse to start instead of dying mid-install with a raw curl/tar
# "No space left on device". The download and the extraction can target
# different filesystems ($TMPDIR vs the prefix parent), so each is checked
# against its own worst case:
#   - the archive filesystem must hold the tarball: the manifest's own
#     size_bytes (1,713,449,440 = ~1.6 GiB for ROCm 7.14.0);
#   - the prefix filesystem must hold the extracted tree: the manifest has no
#     extracted size, so this floor is derived from the validated install
#     (an 8.3 GiB ~/rocm-7.14.0 tree from the 1.6 GiB tarball), rounded up to
#     9 GiB to stay honest across patch releases.
EXTRACTED_FLOOR_BYTES=$((9 * 1024 * 1024 * 1024))

require_available_bytes() {  # require_available_bytes <dir> <floor_bytes> <what> <remedy>
    local dir="$1" floor_bytes="$2" what="$3" remedy="$4"
    if [ ! -d "$dir" ]; then
        echo "ERROR: directory $dir does not exist; create it and rerun." >&2
        exit 1
    fi
    local df_line avail_bytes mount
    df_line="$(df -Pk "$dir" | awk 'NR==2')"
    avail_bytes="$(( $(printf '%s\n' "$df_line" | awk '{print $4}') * 1024 ))"
    mount="$(printf '%s\n' "$df_line" | awk '{print $NF}')"
    if [ "$avail_bytes" -lt "$floor_bytes" ]; then
        printf 'ERROR: not enough disk space for %s: filesystem %s (holding %s) has %s GiB available, need at least %s GiB.\n' \
            "$what" "$mount" "$dir" \
            "$(awk -v b="$avail_bytes" 'BEGIN {printf "%.1f", b / 1073741824}')" \
            "$(awk -v b="$floor_bytes" 'BEGIN {printf "%.1f", b / 1073741824}')" >&2
        printf '       %s\n' "$remedy" >&2
        exit 1
    fi
}

require_available_bytes "$(dirname "$ARCHIVE")" "$SIZE" \
    "the ROCm $ROCM_VER archive download (staged at $ARCHIVE)" \
    "Free space on that filesystem, or set TMPDIR (or ROCM714_ARCHIVE) to a path on a larger filesystem, then rerun."
require_available_bytes "$PARENT" "$EXTRACTED_FLOOR_BYTES" \
    "the extracted ROCm tree at $PREFIX" \
    "Free space on that filesystem, or set ROCM714_PREFIX to a path on a larger filesystem, then rerun."

echo "Downloading ROCm $ROCM_VER gfx1151 tarball (~$((SIZE/1024/1024)) MiB) ..."
echo "  $URL"
curl --fail --location --retry 5 --retry-all-errors --connect-timeout 20 \
    --output "$ARCHIVE" "$URL"

echo "Verifying size + SHA256 against $MANIFEST ..."
[ "$(stat -c %s "$ARCHIVE")" -eq "$SIZE" ] || { echo "ERROR: size mismatch" >&2; exit 1; }
printf '%s  %s\n' "$SHA256" "$ARCHIVE" | sha256sum -c -

STAGE="$(mktemp -d "$PARENT/.rocm-7.14.0.XXXXXX")"
trap 'rm -rf -- "$STAGE"' EXIT
echo "Extracting → $PREFIX ..."
tar -xf "$ARCHIVE" -C "$STAGE"
[ -x "$STAGE/bin/hipcc" ] || { echo "ERROR: extracted tarball has no bin/hipcc" >&2; exit 1; }
mv "$STAGE" "$PREFIX"
trap - EXIT

# F-02: the verified tarball is a transient download, not a cache - on the
# default /tmp (often tmpfs on UMA hosts) leaving it behind silently pins
# ~1.6 GiB of RAM-backed storage. Remove it once verification and extraction
# succeeded; failure paths above keep it for inspection/retry.
rm -f -- "$ARCHIVE"
echo "Cleaned up archive $ARCHIVE (deleted after successful verification)."

echo "Installed ROCm $ROCM_VER at $PREFIX. Activate in a shell with:"
echo "  export PATH=\"$PREFIX/bin:\$PATH\""
echo "  export LD_LIBRARY_PATH=\"$PREFIX/lib:\${LD_LIBRARY_PATH:-}\""
# F-01: this tail used to pipe hipcc --version into `head -1` - the only
# pipeline in the script. hipcc emits several lines, head exits after the
# first and closes the pipe, hipcc takes SIGPIPE, and `set -o pipefail`
# promoted it to exit 141 after a fully successful install. Capture the
# output first, then print line 1 with no pipe in sight.
head -1 <<<"$("$PREFIX/bin/hipcc" --version)"
