#!/usr/bin/env bash
set -euo pipefail

fail() {
    echo "FAIL: $1" >&2
    echo "    see docs/troubleshooting.md" >&2
    exit 1
}
warn() { echo "WARNING: $1" >&2; }
# F-12: wording kept in lockstep with the tool guard in install-rocm-7.14.sh.
missing_tool_fail() {
    echo "FAIL: required command not found: $1" >&2
    echo "  Debian/Ubuntu:  sudo apt-get install $1" >&2
    echo "  Fedora/RHEL:    sudo dnf install $1" >&2
    echo "  Arch:           sudo pacman -S $1" >&2
    echo "    see docs/troubleshooting.md" >&2
    exit 1
}
usage() {
    cat <<'EOF'
Usage: bash scripts/00-check-env.sh [--profile gguf|vllm|reference]

Profiles:
  gguf       Default llama.cpp/GGUF path; no Python package, uv, or PyTorch requirement.
  vllm       Advanced BF16 path using the validated ROCm 7.2.1 reference stack.
  reference  Certify the exact historical host/runtime envelope.
EOF
}

PROFILE="gguf"
while [ "$#" -gt 0 ]; do
    case "$1" in
        --profile)
            [ "$#" -ge 2 ] || { usage >&2; exit 2; }
            PROFILE="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "ERROR: unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done
case "$PROFILE" in
    gguf|vllm|reference) ;;
    *)
        echo "ERROR: invalid profile: $PROFILE" >&2
        usage >&2
        exit 2
        ;;
esac

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# F-12: the manifest reads below shell out to python3 before any diagnostic
# has printed; guard the interpreter first so a host without it gets an
# actionable FAIL instead of a raw bash "command not found".
command -v python3 >/dev/null 2>&1 || missing_tool_fail python3
# shellcheck source=scripts/lib/version.sh
source "$ROOT/scripts/lib/version.sh"
# shellcheck source=scripts/lib/rocm.sh
source "$ROOT/scripts/lib/rocm.sh"

read_manifest() {
    python3 - "$ROOT/configs/validated-stack.json" "$1" <<'PY'
import json, sys
value = json.load(open(sys.argv[1], encoding="utf-8"))
for key in sys.argv[2].split("."):
    value = value[key]
if isinstance(value, list):
    print("\n".join(value))
else:
    print(value)
PY
}
read_default_gguf_bytes() {
    python3 - "$ROOT/configs/artifact-manifest.json" <<'PY'
import json, sys
files = json.load(open(sys.argv[1], encoding="utf-8"))["sets"]["gguf"]["files"]
for item in files:
    if item["path"] == "muse-glimmer-30B-kquant-17gb.gguf":
        print(item["size_bytes"])
        break
else:
    raise SystemExit("default GGUF is absent from the artifact manifest")
PY
}

MIN_KERNEL="$(read_manifest host.minimum_kernel)"
MIN_VISIBLE_GIB="$(read_manifest host.gpu_visible_memory_min_gib)"
REFERENCE_ROCM="$(read_manifest host.rocm_toolchain)"
REFERENCE_PYTHON="$(read_manifest python.version)"
REFERENCE_TORCH="$(read_manifest pytorch.version)"
GGUF_MODEL_BYTES="$(read_default_gguf_bytes)"
# The published GGUF runs used the real 94 GiB Strix Halo host, whose gfx1151
# coarse-grained pool is 80 GiB. This is a warning boundary, not a minimum.
GGUF_REFERENCE_VISIBLE_GIB=80
# F-14: the same manifest-derived size, rendered as the GPU-visible floor the
# user sees next to the measured pool (GPU memory, explicitly not disk).
GGUF_FLOOR_GIB="$(awk -v b="$GGUF_MODEL_BYTES" 'BEGIN {printf "%.1f", b / 1073741824}')"

echo "Environment profile: $PROFILE"
case "$PROFILE" in
    gguf) echo "  path: default llama.cpp + GGUF" ;;
    vllm) echo "  path: optional advanced vLLM + BF16" ;;
    reference) echo "  path: exact historical vLLM/BF16 reference" ;;
esac

if [ "$PROFILE" = "gguf" ]; then
    # F-12: verify the README-declared host tools so an OK verdict means the
    # next step (scripts/gguf-quickstart.sh) will not die on a missing
    # command. Same list as its required-commands loop.
    echo "host tools:"
    for tool in git cmake curl python3; do
        if command -v "$tool" >/dev/null 2>&1; then
            echo "  tool $tool: $(command -v "$tool")"
        else
            missing_tool_fail "$tool"
        fi
    done
fi

resolve_rocm_prefix || exit 1
export PATH="$ROCM_PREFIX/bin:$PATH"
rocm_ver="$(detect_rocm_version "$ROCM_PREFIX")"
print_selected_rocm "$rocm_ver"

case "$PROFILE" in
    gguf)
        case "$rocm_ver" in
            7.14.*) ;;
            7.2.*)
                warn "using the historical ROCm $rocm_ver fallback; ROCm 7.14 is recommended."
                ;;
            *) fail "GGUF profile expects ROCm 7.14.x (recommended) or historical 7.2.x; got '$rocm_ver'. Run scripts/install-rocm-7.14.sh to install 7.14, or set ROCM_PREFIX to an existing 7.14.x/7.2.x prefix." ;;
        esac
        ;;
    vllm|reference)
        [ "$rocm_ver" = "$REFERENCE_ROCM" ] ||
            fail "$PROFILE profile requires the validated ROCm $REFERENCE_ROCM toolchain; set ROCM_PREFIX=/opt/rocm."
        ;;
esac

krel="$(uname -r)"
echo "kernel: $krel"
version_at_least "$krel" "$MIN_KERNEL" ||
    fail "project Strix Halo host floor is kernel >= $MIN_KERNEL (docs/troubleshooting.md#uma-bug); got $krel"
if [ "$PROFILE" = "reference" ]; then
    reference_kernel=0
    while IFS= read -r observed; do
        [ "$krel" = "$observed" ] && reference_kernel=1
    done < <(read_manifest host.observed_benchmark_kernels)
    [ "$reference_kernel" -eq 1 ] ||
        fail "reference profile requires a recorded benchmark kernel; got $krel"
fi

# Buffer rocminfo output. A live rocminfo piped to grep -q can receive SIGPIPE
# under pipefail, so parsing always uses the complete captured output.
rocminfo_out="$("$ROCM_PREFIX/bin/rocminfo" 2>/dev/null || true)"
if ! grep -q "gfx1151" <<<"$rocminfo_out"; then
    # F-15: name what the host actually reported, and where non-gfx1151
    # users should look, instead of a bare "not found".
    observed_gpus="$( { grep -oE 'gfx[0-9]+' <<<"$rocminfo_out" || true; } | sort -u | paste -sd' ' -)"
    fail "gfx1151 not found in $ROCM_PREFIX/bin/rocminfo output; observed GPU id(s): ${observed_gpus:-none}. This project is validated on gfx1151 (AMD Strix Halo) only — see docs/hardware-validation.md for non-gfx1151 platforms."
fi

# First coarse-grained global pool belonging to the gfx1151 agent.
vram_kb="$(awk '
  /Name:[[:space:]]+gfx1151/ { gpu = 1 }
  gpu && /Segment:[[:space:]]+GLOBAL; FLAGS: COARSE GRAINED/ { coarse = 1; next }
  coarse && /Size:/ { size = $2; sub(/\(.*/, "", size); print size; exit }
' <<<"$rocminfo_out")"
[[ "$vram_kb" =~ ^[0-9]+$ ]] ||
    fail "could not read gfx1151 global memory pool from rocminfo"
pool_gib=$(( vram_kb / 1024 / 1024 ))

if [ "$PROFILE" = "gguf" ]; then
    # F-14: state the thresholds with the number on the passing path so a
    # user can self-assess; both floors are GPU-visible memory, not disk.
    echo "GPU-visible pool: ${pool_gib} GiB (default GGUF needs >= ${GGUF_FLOOR_GIB} GiB GPU-visible; validated envelope ${GGUF_REFERENCE_VISIBLE_GIB} GiB GPU-visible — warning boundary, not a minimum)"
else
    echo "GPU-visible pool: ${pool_gib} GiB (this profile requires >= ${MIN_VISIBLE_GIB} GiB GPU-visible)"
fi

if [ "$PROFILE" = "gguf" ]; then
    min_gguf_kib=$(( (GGUF_MODEL_BYTES + 1023) / 1024 ))
    if [ "$vram_kb" -lt "$min_gguf_kib" ]; then
        fail "GPU-visible pool is ${pool_gib} GiB, but the default GGUF needs >= ${GGUF_FLOOR_GIB} GiB GPU-visible to fit; select a smaller quant with GGUF_FILE=<path> (see scripts/gguf-quickstart.sh) or increase GPU-visible memory in the BIOS."
    fi
    reference_visible_kib=$(( GGUF_REFERENCE_VISIBLE_GIB * 1024 * 1024 ))
    if [ "$vram_kb" -lt "$reference_visible_kib" ]; then
        warn "memory configuration is outside the validated 94 GiB Strix Halo / 80 GiB visible-pool reference envelope; the default GGUF may still run."
    fi
else
    min_visible_kib=$(( MIN_VISIBLE_GIB * 1024 * 1024 ))
    [ "$vram_kb" -ge "$min_visible_kib" ] ||
        fail "GPU-visible pool < $MIN_VISIBLE_GIB GiB required by the validated BF16 vLLM path."
fi

if [ "$PROFILE" != "gguf" ]; then
    command -v uv >/dev/null 2>&1 || export PATH="${HOME:?HOME is required}/.local/bin:$PATH"
    command -v uv >/dev/null 2>&1 ||
        fail "uv is required for the optional vLLM/BF16 profile."
    python_version="$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
    [ "$python_version" = "$REFERENCE_PYTHON" ] ||
        fail "$PROFILE profile requires Python $REFERENCE_PYTHON; got $python_version."
    if ! uv run --no-sync python - "$REFERENCE_TORCH" <<'PY'
import sys
import torch

expected = sys.argv[1]
if torch.__version__ != expected:
    raise SystemExit(f"expected torch {expected}, got {torch.__version__}")
if torch.version.hip is None:
    raise SystemExit("installed torch is not a ROCm build")
if not torch.cuda.is_available():
    raise SystemExit("TheRock PyTorch cannot access the GPU")
if not torch.cuda.is_bf16_supported():
    raise SystemExit("gfx1151 does not report BF16 support through PyTorch")
print(f"TheRock PyTorch: {torch.__version__} (HIP {torch.version.hip})")
PY
    then
        fail "pinned TheRock PyTorch/HIP/BF16 check failed; run 'uv sync --locked' for the optional vLLM path."
    fi
fi

echo "OK: $PROFILE environment ready for Muse-Glimmer-30B on gfx1151"
