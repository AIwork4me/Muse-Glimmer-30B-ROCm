# W7900 (gfx1100) full matrix — ROCm 7.14.0 (project-recommended default)

Raw per-cell JSONs from the **full 12-cell Study-2 matrix + both Study-1
cells** run on a Radeon PRO W7900 under **ROCm 7.14.0** (2026-08-16),
aligning the W7900 track with the project's recommended default ROCm release.
Every cell is unmodified harness output (`scripts/gguf-bench-cell.sh`, exact
committed prompts/sampling/seeds); each cell's `manifest` self-labels
`rocm_version: "7.14.0"`, kernel, llama.cpp build, date and full `rocm-smi`
card identity.

## Environment

| Item | Value |
|---|---|
| GPU | AMD Radeon PRO W7900, `gfx1100`, 48 GiB GDDR6 (dedicated) |
| ROCm | **7.14.0** — official AMD tarball **`therock-dist-linux-gfx110X-all-7.14.0.tar.gz`** at a private prefix (side-by-side; system stack untouched) |
| Kernel | 6.8.0-79-generic (Ubuntu 24.04) |
| llama.cpp | upstream `ggml-org/llama.cpp` @ `0b1bad14ff204627636aeb1de22ddcd5acb859d4` (the validated pin), built locally: `GGML_HIP=ON, AMDGPU_TARGETS=gfx1100, Release` |
| GGUFs | identical artifacts to the 7.2.1 pass (byte-size + GGUF-magic verified) |
| Path | `_repro_driver.sh` resume + direct `gguf-bench-cell.sh` cells; single GPU, no other GPU processes during measurement |

## Headline (agg tok/s; `baseline / DFlash` where both measured)

| weight | c=1 | c=4 | c=16 | c=32 |
|---|---:|---:|---:|---:|
| 17GB | 33.72 / 54.27 | 65.69 / 78.88 | 220.62 | 276.16 |
| dynamic | 30.54 / 54.63 | 64.63 / 72.26 | 202.68 | 280.49 |

Study 1 (greedy, 17GB): baseline **33.21**, DFlash **61.45** (draft
acceptance 0.241). All 14/14 planned cells completed.

Within ±2% of the 7.2.1/7.2.4 values on every overlapping cell except
`c=32` (−7%), consistent with the ±6–9% per-cell 7.14-vs-7.2.1 spread the
project measured on gfx1151.

## Findings (negative results are results)

1. **The gfx1151 tarball cannot serve a W7900 beyond single-stream.** The
   repo's pinned `therock-dist-linux-gfx1151-7.14.0.tar.gz` ships rocBLAS
   Tensile data for gfx1151 only. Batched (multi-slot) decode calls rocBLAS
   and dies: `rocBLAS error: Cannot read .../TensileLibrary.dat ... for GPU
   arch : gfx1100` → `llama-server` core-dumps; c=1 cells appear healthy.
   W7900 hosts must use the **`gfx110X-all`** tarball (or distro packages
   covering gfx1100). Observed 2026-08-16; the four c=1 cells first measured
   under the gfx1151 tarball were discarded and re-measured under gfx110X.
2. **`c=32` completes cleanly in an isolated run.** A first in-driver
   attempt appeared pathological (>56 min, no cell), but a clean isolated
   retry completed with a normal 5-rep measurement (wall 339 s). The stall
   was an artifact of a polluted process tree during that session (an orphan
   `llama-server` sharing the GPU/port), not a 7.14 code path. Lesson
   recorded: verify `pgrep llama-server` and `rocm-smi` VRAM are clean
   before/after every cell.
3. **Environment caveat.** This host's kernel (6.8) is older than the
   project's 7.14-validated gfx1151 host (≥6.17). One mid-matrix
   `dynamic c=1` cell died server-side mid-stream during a polluted session
   and was re-measured cleanly (30.54, matching 7.2.1's 30.60). All
   committed cells come from clean, verified-idle GPU states.

Integrity: `SHA256SUMS` covers every cell.
