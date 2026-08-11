#!/usr/bin/env bash
# Build vLLM (PR #51655, muse_glimmer) from source for gfx1151 into the Task 1
# uv venv. Idempotent: re-runs clone+shim+build. Control parallelism via MAX_JOBS.
set -euo pipefail

# --- Pinned Muse-Glimmer vLLM support (PR #51655). -------------------------
# Resolved via the GitHub REST API (no `gh` available on this host):
#   curl -s https://api.github.com/repos/vllm-project/vllm/pulls/51655
# PR head lives on the xianbaoqian/vllm fork, branch tiezhen/new-model-support.
# Update only after re-validation against gfx1151.
VLLM_REPO="${VLLM_REPO:-https://github.com/xianbaoqian/vllm.git}"
VLLM_REF="${VLLM_REF:-606a12cd701875012ffe78a54afd29f97b825dba}"
SRC="third_party/vllm"

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

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

# --- Clone the pinned PR head ------------------------------------------------
echo "=== Cloning vLLM @ $VLLM_REF ==="
rm -rf "$SRC"
mkdir -p third_party
git clone --depth 1 "$VLLM_REPO" "$SRC"
git -C "$SRC" fetch --depth 1 origin "$VLLM_REF"
git -C "$SRC" checkout "$VLLM_REF"
echo "  HEAD: $(git -C "$SRC" log --oneline -1)"

# --- Source patches (applied to the cloned tree; diffs are committed in
#     patches/ for review and re-applied here for reproducibility) ------------

# Patch 1: torch 2.10 vs vLLM-2.13-era stable C++ API. The TheRock gfx1151
# index tops out at torch 2.10.0, but PR #51655 rides recent vLLM main that
# targets torch 2.13. cuda_view.cu uses two APIs that changed: stable::Tensor
# gained layout() in 2.13, and from_blob gained a deleter overload in 2.13.
# We pin layout to Strided (all stable tensors are Strided) and drop the
# deleter (its UVA cleanup path is unused by muse_glimmer inference). All
# other torch::stable call sites already match the 2.10 signature.
TORCH_PATCH="$ROOT/patches/vllm-torch210-compat.diff"
if [ -f "$TORCH_PATCH" ]; then
  echo "  applying torch-2.10 compat patch"
  git -C "$SRC" apply "$TORCH_PATCH"
else
  echo "  WARNING: $TORCH_PATCH missing — torch 2.10 build will fail on cuda_view.cu"
fi

# Patch 2: amdsmi-before-torch shim (gfx1151 workaround). vLLM already imports
# amdsmi lazily in platforms/__init__.py, but on gfx1151 the first ROCm init
# must happen before torch grabs the device. The shim prepends `import amdsmi`
# to vllm/__init__.py.
INIT="$SRC/vllm/__init__.py"
if ! grep -q "import amdsmi" "$INIT"; then
  sed -i '1i import amdsmi  # gfx1151 workaround (see docs/troubleshooting.md#amdsmi)' "$INIT"
fi

# Regenerate both diffs from the patched tree (idempotency check against the
# committed copies). The amdsmi diff is the brief's required artifact.
git -C "$SRC" diff -- vllm/__init__.py > "$ROOT/patches/vllm-amdsmi-import.diff"
git -C "$SRC" diff -- csrc/libtorch_stable/cuda_view.cu > "$ROOT/patches/vllm-torch210-compat.diff"
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
