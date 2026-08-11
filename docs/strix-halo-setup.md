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

## 2. ROCm 7.2.1

gfx1151 officially enters AMD's compatibility matrix at **ROCm 7.14.0**, but it is
community-verified to work on 7.0–7.2.x. This project targets **7.2.1** (no host
upgrade needed) and matches the only known-good consumer-RDNA precedent for this
model (gfx1100 + ROCm 7.2.0).

```bash
cat /opt/rocm/.info/version     # 7.2.x
rocminfo | grep -i gfx1151      # must list gfx1151
```

Want the official path instead? See
[§Alternative: ROCm 7.14.0](#alternative-rocm-7140) below.

## 3. UMA carve-out / VRAM pool

Strix Halo is unified-memory: the GPU draws from system DRAM. Ensure the BIOS
memory-iGPU carve-out (often labelled "UMA Frame Buffer Size" or "iGPU memory")
leaves the bulk of the 94 GiB visible to the runtime. `00-check-env.sh` asserts
the pool ROCm reports is **≥ 60 GB** (warns below that; the UMA bug is the usual
cause).

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
bash scripts/00-check-env.sh  # asserts ROCm 7.2.x · kernel ≥6.16.9 · gfx1151 · ≥60 GB pool
uv run pytest tests/test_env_torch.py tests/test_env.py -v -m gpu
```

Once that is green, continue with the vLLM build (`scripts/01-build-vllm.sh`).

## Alternative: ROCm 7.14.0

If you want AMD's officially-supported ROCm for gfx1151, install **7.14.0**
instead of 7.2.1 and re-point the TheRock index to the matching wheel line. The
adaptations in this project (BF16, `FLASH_ATTN`, source-built vLLM, AITER off)
are ROCm-version-independent; only the torch wheel pin in `pyproject.toml`
changes. Not tested here — recorded as a forward path.

[uma]: https://github.com/ROCm/ROCm/issues/5444
