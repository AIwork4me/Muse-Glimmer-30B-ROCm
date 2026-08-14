#!/usr/bin/env bash
# Build vLLM (PR #51655, muse_glimmer) from source for gfx1151 into the Task 1
# uv venv. Idempotent: re-runs clone+shim+build. Control parallelism via MAX_JOBS.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# --- Pinned Muse-Glimmer vLLM support (PR #51655). -------------------------
# Defaults come from the validated stack manifest. Overrides are intentionally
# supported for upstream experiments, but they no longer reproduce the
# published reference stack.
mapfile -t STACK_VLLM < <(python3 - "$ROOT/configs/validated-stack.json" <<'PY_STACK'
import json
import sys
data = json.load(open(sys.argv[1], encoding="utf-8"))
print(data["vllm"]["source_repo"])
print(data["vllm"]["commit"])
PY_STACK
)
VALIDATED_VLLM_REPO="${STACK_VLLM[0]}"
VALIDATED_VLLM_REF="${STACK_VLLM[1]}"
VLLM_REPO="${VLLM_REPO:-$VALIDATED_VLLM_REPO}"
VLLM_REF="${VLLM_REF:-$VALIDATED_VLLM_REF}"
SRC="${VLLM_SRC:-third_party/vllm}"

if [ "$VLLM_REPO" = "$VALIDATED_VLLM_REPO" ] && [ "$VLLM_REF" = "$VALIDATED_VLLM_REF" ]; then
  echo "vLLM source: validated reference"
else
  echo "vLLM source: EXPERIMENTAL override (published benchmark claims do not apply)"
fi
echo "  repo: $VLLM_REPO"
echo "  ref:  $VLLM_REF"

# uv lives in ~/.local/bin on this host.
export PATH="$HOME/.local/bin:$PATH"

# --- Toolchain selection -----------------------------------------------------
# Two ROCm installs coexist on this host:
#   * host /opt/rocm           -> ROCm 7.2.1, COMPLETE (hipcc + clang 22, every
#                                 cmake config: hip, amd_comgr, rocblas, miopen,
#                                 hiprand, rocprim, hipcub, rocthrust, ...).
#   * venv _rocm_sdk_core      -> ROCm 7.13.0a, the userspace torch ships. Has
#                                 hipcc + clang 23 + headers + versioned libs,
#                                 but NO cmake package configs and no dev .so
#                                 symlinks (it is a *runtime* install).
# torch's LoadHIP.cmake find_package()'s ~12 ROCm components (hip, amd_comgr,
# rocblas, miopen, ...) REQUIRED, so the build must resolve cmake configs.
# Only the host ROCm can. We therefore BUILD against the host ROCm 7.2.1 (its
# hipcc is gfx1151-capable) and rely on torch's 7.13.0a userspace at RUNTIME:
# vLLM's .so links libamdhip64.so.7 (SONAME); at runtime python loads torch's
# 7.13.0a libamdhip64.so.7 first (via torch's RPATH), and the already-loaded
# symbol set (a superset of 7.2.1's) satisfies vLLM. ROCM 7.x HIP runtime ABI
# is backward compatible, and vLLM only uses stable HIP APIs.
#
# vLLM/BF16 deliberately builds against the 7.2.1 host toolchain (/opt/rocm),
# NOT 7.14: this script reproduces the validated historical stack. The
# GGUF/llama.cpp path defaults to ~/rocm-7.14.0; ROCm 7.14 Muse-Glimmer vLLM
# validation is optional / not prioritized for v0.1 and pending. Current
# rocBLAS BF16-GEMM proxy results did not justify prioritizing that rebuild;
# they do not rule out value from a future cohesive 7.14 stack.
# ROCM_PATH is exported so torch.utils.cpp_extension.ROCM_HOME picks it up and
# setup.py forwards it to cmake as -DROCM_PATH=.
export ROCM_PATH="${ROCM_PATH:-/opt/rocm}"
export HIP_PATH="${HIP_PATH:-/opt/rocm}"
# Put /opt/rocm/bin first so hipcc/hipconfig/amdclang all come from the host
# ROCm (7.2.1) — a single internally-consistent toolchain.
export PATH="$ROCM_PATH/bin:$HOME/.local/bin:$PATH"

# amdsmi must be importable at RUNTIME: vLLM's ROCm platform plugin calls
# amdsmi.amdsmi_init() during platform detection, and our gfx1151 shim (below)
# prepends `import amdsmi` to vllm/__init__.py. The venv does not have amdsmi
# on sys.path by default; expose the TheRock-bundled amdsmi (26.4.0, wraps the
# libamd_smi.so.26 from the same 7.13.0a tree torch uses) via a .pth. Its
# wrapper resolves its .so via a path RELATIVE TO THE PACKAGE FILE, so it must
# stay at its original location (a `pip install` copy would break that).
VENV_SITE=".venv/lib/python3.12/site-packages"
VENV_ROCM="$VENV_SITE/_rocm_sdk_core"
AMD_PTH="$VENV_SITE/_amdsmi_therock.pth"
echo "$ROOT/$VENV_ROCM/share/amd_smi" > "$AMD_PTH"

echo "=== Toolchain ==="
echo "  hipcc:   $(command -v hipcc)"
echo "  ROCM_PATH=$ROCM_PATH"
hipcc --version | sed 's/^/  hipcc> /' | head -3

# --- Install build backend deps ---------------------------------------------
# --no-build-isolation means the build backend must already live in the venv.
echo "=== Installing build backend deps (no-build-isolation needs them in-env) ==="
uv pip install "setuptools-scm>=8.0" "cmake>=3.26.1" "ninja" "wheel" "setuptools-rust>=1.9.0"

# --- Fetch or reuse the pinned source safely ---------------------------------
# Never delete an existing checkout: it may contain developer work. A checkout
# at another commit must be clean before this script changes HEAD.
mkdir -p "$(dirname "$SRC")"
if [ -e "$SRC" ] && [ ! -d "$SRC/.git" ]; then
  echo "ERROR: $SRC exists but is not a git checkout; choose VLLM_SRC or move it." >&2
  exit 1
fi
if [ ! -d "$SRC/.git" ]; then
  echo "=== Cloning vLLM @ $VLLM_REF ==="
  git clone --depth 1 "$VLLM_REPO" "$SRC"
fi

CURRENT_REF="$(git -C "$SRC" rev-parse HEAD 2>/dev/null || true)"
if [ "$CURRENT_REF" != "$VLLM_REF" ]; then
  if ! git -C "$SRC" diff --quiet --ignore-submodules HEAD -- ||
     ! git -C "$SRC" diff --cached --quiet; then
    echo "ERROR: $SRC has tracked changes at $CURRENT_REF; use a clean VLLM_SRC for another ref." >&2
    exit 1
  fi
  git -C "$SRC" fetch --depth 1 "$VLLM_REPO" "$VLLM_REF"
  git -C "$SRC" checkout --detach FETCH_HEAD
else
  echo "  checkout already at validated commit; no fetch needed"
fi
ACTUAL_REF="$(git -C "$SRC" rev-parse HEAD)"
if [[ "$VLLM_REF" =~ ^[0-9a-fA-F]{40}$ ]] && [ "$ACTUAL_REF" != "$VLLM_REF" ]; then
  echo "ERROR: requested $VLLM_REF but checked out $ACTUAL_REF" >&2
  exit 1
fi
echo "  HEAD: $(git -C "$SRC" log --oneline -1)"

# --- Source patches (committed in patches/ and applied idempotently) ---------
apply_pinned_patch() {
  local patch="$1"
  if git -C "$SRC" apply --check "$patch"; then
    git -C "$SRC" apply "$patch"
    echo "  applied $(basename "$patch")"
  elif git -C "$SRC" apply --reverse --check "$patch"; then
    echo "  already applied $(basename "$patch")"
  else
    echo "ERROR: $(basename "$patch") neither applies nor matches the source tree." >&2
    exit 1
  fi
}

# Patch 1: torch 2.10 vs vLLM-2.13-era stable C++ API. The TheRock gfx1151
# index tops out at torch 2.10.0, but PR #51655 rides recent vLLM main that
# targets torch 2.13. cuda_view.cu uses two APIs that changed: stable::Tensor
# gained layout() in 2.13, and from_blob gained a deleter overload in 2.13.
# We pin layout to Strided (all stable tensors are Strided) and drop the
# deleter (its UVA cleanup path is unused by muse_glimmer inference). All
# other torch::stable call sites already match the 2.10 signature.
TORCH_PATCH="$ROOT/patches/vllm-torch210-compat.diff"
AMDSMI_PATCH="$ROOT/patches/vllm-amdsmi-import.diff"
for patch in "$TORCH_PATCH" "$AMDSMI_PATCH"; do
  if [ ! -f "$patch" ]; then
    echo "ERROR: required patch missing: $patch" >&2
    exit 1
  fi
  apply_pinned_patch "$patch"
done

# Fail if tracked modifications extend beyond the two documented patches.
mapfile -t PATCHED_FILES < <(git -C "$SRC" diff --name-only)
EXPECTED_FILES=("csrc/libtorch_stable/cuda_view.cu" "vllm/__init__.py")
if [ "${PATCHED_FILES[*]}" != "${EXPECTED_FILES[*]}" ] ||
   ! git -C "$SRC" diff --cached --quiet; then
  echo "ERROR: $SRC contains tracked changes beyond the two validated patches:" >&2
  git -C "$SRC" status --short >&2
  exit 1
fi
echo "  patches: $(git -C "$SRC" diff --stat | tail -1)"

# --- Build (editable, no isolation so the build sees installed torch+ROCm) ---
# --no-build-isolation is REQUIRED: an isolated build would not see the TheRock
# torch + host ROCm and would fail.
#
# The Rust frontend (vllm._rust_tool_parser / vllm-rs CLI) is OPTIONAL: lazily
# imported only for the generic Rust tool parser and `vllm bench serve`.
# Muse-Glimmer ships its own Python tool/reasoning parsers, so we skip the
# Rust build (avoids requiring a cargo toolchain). Set VLLM_REQUIRE_RUST_FRONTEND=1
# (and install rustup) for a full build.
export VLLM_REQUIRE_RUST_FRONTEND="${VLLM_REQUIRE_RUST_FRONTEND:-0}"
export PYTORCH_ROCM_ARCH=gfx1151
export FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE
export MAX_JOBS="${MAX_JOBS:-16}"

echo "=== Building (PYTORCH_ROCM_ARCH=$PYTORCH_ROCM_ARCH MAX_JOBS=$MAX_JOBS VLLM_REQUIRE_RUST_FRONTEND=$VLLM_REQUIRE_RUST_FRONTEND) ==="
uv pip install -e "$SRC" --no-build-isolation

# vLLM declares numpy unpinned; its editable install just pulled numpy 2.x. The
# TheRock torch wheel was built against numpy<2 (numpy 2.x breaks the torch C
# extension ABI), so force it back. vLLM itself works with numpy 1.26.x.
# vLLM also dragged in scipy 1.18.x (an optional bench/audio extra) which uses
# np.long (removed in numpy 1.24); pin a numpy-1.26-compatible scipy.
uv pip install "numpy<2" "scipy<1.14" >/dev/null && echo "  pinned numpy<2 ($(uv pip show numpy | awk '/^Version:/{print $2}')), scipy<1.14 ($(uv pip show scipy | awk '/^Version:/{print $2}'))"

echo "=== OK: vLLM built for gfx1151 ==="
echo "Verify with: uv run --no-sync python -c 'import vllm; print(vllm.__version__)'"
