#!/usr/bin/env bash
set -euo pipefail

# uv usually lives at ~/.local/bin; make sure later tasks/services can find it
# even when invoked with a minimal PATH (systemd, cron, subprocess from pytest).
command -v uv >/dev/null 2>&1 || export PATH="$HOME/.local/bin:$PATH"

fail() { echo "FAIL: $1" >&2; echo "    see docs/troubleshooting.md" >&2; exit 1; }

# ROCm version (accept 7.2.x)
rocm_ver="$(cat /opt/rocm/.info/version 2>/dev/null || echo none)"
echo "ROCm: $rocm_ver"
[[ "$rocm_ver" == 7.2.* ]] || fail "expected ROCm 7.2.x, got $rocm_ver"

# Kernel floor (fixes the 15.5 GB UMA bug)
krel="$(uname -r)"
echo "kernel: $krel"
kmajor="$(echo "$krel" | cut -d. -f1)"; kminor="$(cut -d. -f2 <<<"$krel" | cut -d- -f1)"
# NOTE: each `[ ... ]` inside a brace group must end with `;` before the `}` is
# recognized as a reserved word (brief had `] }`, which bash rejects as a syntax
# error). `;` added before the inner closing brace.
{ [ "$kmajor" -ge 7 ] || { [ "$kmajor" -eq 6 ] && [ "$kminor" -ge 16 ]; }; } \
  || fail "kernel >= 6.16.9 required (see docs/troubleshooting.md#uma-bug); got $krel"

# gfx target.
# Buffer rocminfo output: `grep -q` exits on first match, and under `pipefail`
# a live `rocminfo` still writing gets SIGPIPE (exit 141), falsely failing the
# check. Capture first, then grep the buffer (no broken pipe).
rocminfo_out="$(rocminfo 2>/dev/null || true)"
grep -q "gfx1151" <<<"$rocminfo_out" || fail "gfx1151 not found in rocminfo"

# VRAM pool visible to the runtime (warn, don't fail, below 60 GB).
# --no-sync so uv never re-syncs the venv (preserves the Task 3 source-built
# vLLM install when this script is re-run by later tasks).
vram_kb="$(uv run --no-sync python - <<'PY'
import torch
print(torch.cuda.get_device_properties(0).total_memory // 1024)
PY
)"
echo "VRAM visible: $(( vram_kb / 1024 / 1024 )) GB"
[ "$vram_kb" -ge 62914560 ] || fail "VRAM pool < 60 GB; check UMA carve-out (docs/troubleshooting.md#uma-bug)"

echo "OK: environment ready for Muse-Glimmer-30B on gfx1151"
