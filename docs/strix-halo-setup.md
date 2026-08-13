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

## 2. ROCm tracks

The published benchmark is historical evidence from **ROCm 7.2.1**. ROCm
**7.14.0** is the forward official gfx1151 validation target, kept as a separate
track so new results cannot overwrite the reference matrix. See
[`docs/results/README.md`](results/README.md) for the status and evidence
boundary.

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
bash scripts/00-check-env.sh  # asserts ROCm 7.2.x · kernel ≥6.16.9 · gfx1151 · ≥60 GiB pool
uv run pytest tests/test_env_torch.py tests/test_env.py -v -m gpu
```

Once that is green, continue with the vLLM build (`scripts/01-build-vllm.sh`).

## Alternative: ROCm 7.14.0

If you want AMD's officially-supported ROCm for gfx1151, use **7.14.0** (released
2026-07-16; first ROCm with official gfx1151 APU support). Note: 7.14.0 ships via
**TheRock** (the new modular build system) — the legacy `repo.radeon.com/rocm/apt/`
repo tops out at **7.2.4**, so `apt` will not find 7.13/7.14 there. Install via the
TheRock release/prefix and re-point the torch wheel pin in `pyproject.toml` to the
matching line. The adaptation hypotheses (BF16, `TRITON_ATTN`, source-built vLLM, AITER
off) must be revalidated as a complete stack; do not assume only the wheel pin
changes. Partial cells may exist, but the ROCm 7.14 track is incomplete and has
no published conclusion. Follow
[`docs/results/rocm-7.14/README.md`](results/rocm-7.14/README.md). Caveat:
official support does not imply every workload is bug-free; open Strix Halo
unified-memory issues
([#6370](https://github.com/ROCm/ROCm/issues/6370), [#6165](https://github.com/ROCm/ROCm/issues/6165))
persist and can affect stability under sustained load.

[uma]: https://github.com/ROCm/ROCm/issues/5444
