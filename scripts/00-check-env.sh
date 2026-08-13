#!/usr/bin/env bash
set -euo pipefail

# uv usually lives at ~/.local/bin; make sure later tasks/services can find it
# even when invoked with a minimal PATH (systemd, cron, subprocess from pytest).
command -v uv >/dev/null 2>&1 || export PATH="$HOME/.local/bin:$PATH"

fail() { echo "FAIL: $1" >&2; echo "    see docs/troubleshooting.md" >&2; exit 1; }
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck source=scripts/lib/version.sh
source "$ROOT/scripts/lib/version.sh"
read_manifest() {
    python3 - "$ROOT/configs/validated-stack.json" "$1" <<'PY'
import json, sys
value = json.load(open(sys.argv[1]))
for key in sys.argv[2].split('.'):
    value = value[key]
print(value)
PY
}

MIN_KERNEL="$(read_manifest host.minimum_kernel)"
MIN_VISIBLE_GIB="$(read_manifest host.gpu_visible_memory_min_gib)"


# ROCm version (accept 7.2.x)
rocm_ver="$(cat /opt/rocm/.info/version 2>/dev/null || echo none)"
echo "ROCm: $rocm_ver"
[[ "$rocm_ver" == 7.2.* ]] || fail "expected ROCm 7.2.x, got $rocm_ver"

# Kernel floor (fixes the 15.5 GB UMA bug). Compare major, minor, and patch.
krel="$(uname -r)"
echo "kernel: $krel"
version_at_least "$krel" "$MIN_KERNEL" \
  || fail "kernel >= $MIN_KERNEL required (see docs/troubleshooting.md#uma-bug); got $krel"

# gfx target.
# Buffer rocminfo output: `grep -q` exits on first match, and under `pipefail`
# a live `rocminfo` still writing gets SIGPIPE (exit 141), falsely failing the
# check. Capture first, then grep the buffer (no broken pipe).
rocminfo_out="$(rocminfo 2>/dev/null || true)"
grep -q "gfx1151" <<<"$rocminfo_out" || fail "gfx1151 not found in rocminfo"

# GPU-visible unified-memory pool. Parse the first coarse-grained global pool
# belonging to the gfx1151 agent. This matches torch's visible 80 GiB pool on
# the validated host and keeps the preflight usable before `uv sync`.
vram_kb="$(awk '
  /Name:[[:space:]]+gfx1151/ { gpu = 1 }
  gpu && /Segment:[[:space:]]+GLOBAL; FLAGS: COARSE GRAINED/ { coarse = 1; next }
  coarse && /Size:/ { size = $2; sub(/\(.*/, "", size); print size; exit }
' <<<"$rocminfo_out")"
[[ "$vram_kb" =~ ^[0-9]+$ ]] || fail "could not read gfx1151 global memory pool from rocminfo"
echo "GPU-visible pool: $(( vram_kb / 1024 / 1024 )) GiB"
min_visible_kib=$(( MIN_VISIBLE_GIB * 1024 * 1024 ))
[ "$vram_kb" -ge "$min_visible_kib" ] || fail "VRAM pool < $MIN_VISIBLE_GIB GiB; check UMA carve-out (docs/troubleshooting.md#uma-bug)"

echo "OK: environment ready for Muse-Glimmer-30B on gfx1151"
