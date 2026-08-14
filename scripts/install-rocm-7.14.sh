#!/usr/bin/env bash
# Install AMD's official ROCm 7.14.0 gfx1151 tarball side-by-side at
# ~/rocm-7.14.0 (default), WITHOUT touching the system /opt/rocm (7.2.1).
#
# URL, byte size and SHA256 are read from configs/rocm-7.14-gguf-validation.json
# so the manifest is the single source of truth (no hardcoded hash here).
#
# Idempotent: if ~/rocm-7.14.0/bin/hipcc already exists, this is a no-op.
# Usage: bash scripts/install-rocm-7.14.sh [ROCM714_PREFIX]
#   ROCM714_PREFIX=/path   override the install prefix
#   ROCM714_ARCHIVE=/path  override the download location
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
MANIFEST="$HERE/configs/rocm-7.14-gguf-validation.json"

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
echo "Downloading ROCm $ROCM_VER gfx1151 tarball (~$((SIZE/1024/1024)) MiB) ..."
echo "  $URL"
curl --fail --location --retry 5 --retry-all-errors --connect-timeout 20 \
    --output "$ARCHIVE" "$URL"

echo "Verifying size + SHA256 against $MANIFEST ..."
[ "$(stat -c %s "$ARCHIVE")" -eq "$SIZE" ] || { echo "ERROR: size mismatch" >&2; exit 1; }
printf '%s  %s\n' "$SHA256" "$ARCHIVE" | sha256sum -c -

PARENT="$(dirname "$PREFIX")"
mkdir -p "$PARENT"
STAGE="$(mktemp -d "$PARENT/.rocm-7.14.0.XXXXXX")"
trap 'rm -rf -- "$STAGE"' EXIT
echo "Extracting → $PREFIX ..."
tar -xf "$ARCHIVE" -C "$STAGE"
[ -x "$STAGE/bin/hipcc" ] || { echo "ERROR: extracted tarball has no bin/hipcc" >&2; exit 1; }
mv "$STAGE" "$PREFIX"
trap - EXIT

echo "Installed ROCm $ROCM_VER at $PREFIX. Activate in a shell with:"
echo "  export PATH=\"$PREFIX/bin:\$PATH\""
echo "  export LD_LIBRARY_PATH=\"$PREFIX/lib:\${LD_LIBRARY_PATH:-}\""
"$PREFIX/bin/hipcc" --version | head -1
