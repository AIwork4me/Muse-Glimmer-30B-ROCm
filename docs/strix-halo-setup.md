# Strix Halo setup (prerequisites)

Target box: **AMD Ryzen AI MAX+ PRO 395** w/ **Radeon 8060S** ("Strix Halo"),
gfx1151 (RDNA 3.5), **94 GiB unified LPDDR5X**. This guide gets the host to the
point where `scripts/00-check-env.sh` passes.

## 1. Kernel ≥ 6.16.9 (required)

Older kernels expose only **~15.5 GB** of the unified pool to ROCm — a KFD/HSA
UMA-handling bug ([ROCm/ROCm#5444][uma]). Anything ≥ 6.16.9 fixes it. This host
runs `6.17.0-1020-oem`.

```bash
uname -r            # must be >= 6.16.9
```

If you are below the floor, upgrade your kernel before going further — no amount
of ROCm config recovers the missing memory. See
[troubleshooting.md#uma-bug](troubleshooting.md#uma-bug).

## 2. ROCm — 7.14.0 (recommended default)

**ROCm 7.14.0** is AMD's official gfx1151 distribution and this project's
recommended default. The historical benchmark evidence was recorded on **ROCm
7.2.1** (the fully-validated reference, kept as supplementary); the 7.14
GGUF/llama.cpp matrix reproduces it within noise. See
[`docs/results/README.md`](results/README.md) for the status and evidence
boundary.

```bash
bash scripts/install-rocm-7.14.sh              # installs ~/rocm-7.14.0 (official gfx1151)
~/rocm-7.14.0/bin/rocminfo | grep -i gfx1151   # must list gfx1151
# System /opt/rocm (7.2.1) is the historical reference; both coexist side-by-side.
```

Manual install details (size/SHA256 verification, staged extraction):
[§ROCm 7.14.0 manual install](#rocm-7140-install) below. The system `/opt/rocm`
(7.2.1) needs no special install; see [docs/results/README.md](results/README.md)
for the historical-reference boundary.

## 3. UMA carve-out / VRAM pool

Strix Halo is unified-memory: the GPU draws from system DRAM. Ensure the BIOS
memory-iGPU carve-out (often labelled "UMA Frame Buffer Size" or "iGPU memory")
leaves the bulk of the 94 GiB visible to the runtime. `00-check-env.sh` asserts
the pool ROCm reports is **≥ 60 GiB**. This is a hard requirement for the
validated BF16 path, so the check fails below it; the UMA bug is the usual
cause. The smaller GGUF path may work on lower-memory systems, but that is not
yet a validated configuration.

## 4. Python 3.12 + uv

The TheRock gfx1151 torch wheels **fail to import on Python 3.13**; pin **3.12**.
The project is `uv`-managed.

```bash
python3.12 --version          # 3.12.*
command -v uv || curl -LsSf https://astral.sh/uv/install.sh | sh
```

## 5. Verify

```bash
uv sync                       # create .venv, pull TheRock gfx1151 torch
bash scripts/00-check-env.sh  # asserts ROCm 7.14.x/7.2.x · kernel ≥6.16.9 · gfx1151 · ≥60 GiB pool
uv run pytest tests/test_env_torch.py tests/test_env.py -v -m gpu
```

Once that is green, continue with the vLLM build (`scripts/01-build-vllm.sh`).

<a id="rocm-7140-install"></a>
## ROCm 7.14.0 — manual install details

ROCm 7.14.0 was published on **2026-07-15** and provides an official AMD gfx1151
distribution. AMD's release notes list several gfx1151 Ryzen AI MAX PRO SKUs,
but not the exact Ryzen AI MAX+ PRO 395 used here. The repository's 395 result
is independent project validation, not an AMD SKU-support claim.

Install the recorded tarball side-by-side. The commands verify size and SHA256,
retry transient HTTP failures, refuse to overlay an existing prefix, and stage
extraction before the final rename:

```bash
ROCM714_PREFIX="${ROCM714_PREFIX:-$HOME/rocm-7.14.0}"
ROCM714_ARCHIVE="${TMPDIR:-/tmp}/therock-dist-linux-gfx1151-7.14.0.tar.gz"
ROCM714_URL="https://repo.amd.com/rocm/tarball-multi-arch/therock-dist-linux-gfx1151-7.14.0.tar.gz"
ROCM714_SHA256="2567d5e34e470db104a62a02c36aa770cb0430175e48c1c46df0eefc05e1d77c"
ROCM714_SIZE=1713449440

if [[ -e "$ROCM714_PREFIX" ]]; then
  echo "Refusing to overlay existing prefix: $ROCM714_PREFIX" >&2
  exit 1
fi

curl --fail --location --retry 5 --retry-all-errors --connect-timeout 20 \
  --output "$ROCM714_ARCHIVE" "$ROCM714_URL"
test "$(stat -c %s "$ROCM714_ARCHIVE")" -eq "$ROCM714_SIZE"
printf '%s  %s\n' "$ROCM714_SHA256" "$ROCM714_ARCHIVE" | sha256sum -c -

ROCM714_PARENT="$(dirname "$ROCM714_PREFIX")"
mkdir -p "$ROCM714_PARENT"
ROCM714_STAGE="$(mktemp -d "$ROCM714_PARENT/.rocm-7.14.0.XXXXXX")"
trap 'rm -rf -- "$ROCM714_STAGE"' EXIT
tar -xf "$ROCM714_ARCHIVE" -C "$ROCM714_STAGE"
test -x "$ROCM714_STAGE/bin/hipcc"
mv "$ROCM714_STAGE" "$ROCM714_PREFIX"
trap - EXIT

export PATH="$ROCM714_PREFIX/bin:$PATH"
export LD_LIBRARY_PATH="$ROCM714_PREFIX/lib:${LD_LIBRARY_PATH:-}"
hipcc --version
```

The archive identity and observed HIP/LLVM versions are authoritative in
[`configs/rocm-7.14-gguf-validation.json`](../configs/rocm-7.14-gguf-validation.json).

**Validation status (2026-08-13):** the reduced **GGUF/llama.cpp track is
project-validated for 17 of 21 planned cells**; all four `np=16` cells were
deferred. Mean TPOT deltas were −0.4% at `np=1` and −1.7% at `np=4`. All 17
cells recorded a lower VmPeak mapped-address-space envelope (mean −2.8%).
DFlash acceptance rates were similar. The operator observed no incident during
the six-hour run, but raw system logs were not retained, so the run is not a
standalone stability qualification. See the [scoped result and
protocol](results/rocm-7.14/README.md) and [per-cell summary
evidence](results/matrix-714/).

The **BF16/vLLM track was evaluated and is not pursued**: a rocBLAS BF16-GEMM
proxy showed no 7.14 compute gain over 7.2.1, so a vLLM/7.14 rebuild would not
improve UX (c=1 is bandwidth-bound; c≥4 GEMM is unchanged). vLLM/BF16 stays on
the validated 7.2.1 reference — do not infer the GGUF result to the vLLM stack.

Open Strix Halo unified-memory issues
([#6370](https://github.com/ROCm/ROCm/issues/6370),
[#6165](https://github.com/ROCm/ROCm/issues/6165)) remain relevant to future
stress validation.

[uma]: https://github.com/ROCm/ROCm/issues/5444
