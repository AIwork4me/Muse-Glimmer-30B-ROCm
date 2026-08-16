# W7900 (gfx1100) independent reproduction cells — ROCm 7.2.1

Raw per-cell JSONs from an **independent reproduction** of the W7900 numbers on
a **second Radeon PRO W7900 host**, run by the maintainer while reviewing
PR #6 (2026-08-16). Every cell is the unmodified output of the committed
harness (`scripts/gguf-bench-cell.sh`); each cell's `manifest` field
self-labels `rocm_version`, kernel, llama.cpp build, date and the full
`rocm-smi` card identity (Device 0x744b, `gfx1100`, SKU D7070910).

## Environment

| Item | Value |
|---|---|
| GPU | AMD Radeon PRO W7900, `gfx1100`, 48 GiB GDDR6 (dedicated) |
| ROCm | **7.2.1** (system stack at `/opt/rocm`) |
| Kernel | 6.8.0-79-generic (Ubuntu 24.04) |
| llama.cpp | upstream `ggml-org/llama.cpp` @ `0b1bad14ff204627636aeb1de22ddcd5acb859d4` — the repo's validated pin — built locally: `GGML_HIP=ON, AMDGPU_TARGETS=gfx1100, Release` |
| GGUFs | `meta-models/Muse-Glimmer-30B-GGUF` via `scripts/w7900-repro/00_prepare.sh` (HF mirror endpoint; exact byte size + GGUF magic verified for all four artifacts) |
| Path | Method 2 bare metal (`run_host.sh` / `gguf-bench-cell.sh`), same `study1.conf` / `study2.conf` prompts, sampling and seeds as the original run |

## Coverage (9 cells)

Study 1 (greedy): `17gb × {baseline, DFlash}` — the two cells whose numbers
were quoted in PR #6's README table without committed evidence; they are now
backed by these cells.

Study 2 (temp 1.0): `17gb × {c=1, c=1+DFlash, c=4, c=4+DFlash, c=16, c=32}` and
`dynamic × {c=1}`. The remaining five `dynamic` cells were not run: the 7.2.1
pass was stopped early to prioritize the full **ROCm 7.14.0** matrix (the
project's recommended default), which is committed separately under
[`../cells-rocm-7.14.0/`](../cells-rocm-7.14.0/). This is an honest partial,
not a cherry-pick: the run order is the driver's fixed sequence and every
completed cell is committed.

## Deltas vs the PR #6 submission (ROCm 7.2.4 image)

| cell | PR #6 (7.2.4) | this run (7.2.1) | delta |
|---|---:|---:|---:|
| study1 17gb baseline | 33.19 | 33.31 | +0.4% |
| study1 17gb DFlash | 63.98 | 62.03 (accept 0.241) | −3.0% |
| study2 17gb c=1 | 33.72 | 33.77 | +0.2% |
| study2 17gb c=1 DFlash | 56.35 | 54.69 | −2.9% |
| study2 17gb c=4 | 66.46 | 65.77 | −1.0% |
| study2 17gb c=4 DFlash | 77.44 | 79.17 | +2.2% |
| study2 17gb c=16 | 218.56 | 221.43 | +1.3% |
| study2 17gb c=32 | 293.29 | 298.08 | +1.6% |
| study2 dynamic c=1 | 30.61 | 30.60 | −0.0% |

All cells agree within ±3% across two hosts and two ROCm point releases —
the PR #6 numbers are corroborated. Integrity: `SHA256SUMS` covers every cell.
