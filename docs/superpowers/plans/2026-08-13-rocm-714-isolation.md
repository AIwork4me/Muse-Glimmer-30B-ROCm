# Plan — Part 2: ROCm 7.14.0 isolation comparison (7.2.1 vs 7.14.0)

- **Date:** 2026-08-13
- **Status:** Draft (awaiting go-ahead)
- **Spec source of truth:** `docs/superpowers/specs/2026-08-12-llamacpp-dflash-benchmark-design.md` §9 (P-D)
- **Baseline (immutable):** the 21-cell ROCm **7.2.1** matrix committed on `master` at `docs/results/matrix/`
- **Controlled variable:** **ROCm version only.** Same llama.cpp source (`0b1bad1`), same flags, same weights, same prompt set, same seeds, same harness.

---

## 0. Goal

Produce an honest, cell-by-cell **ROCm 7.2.1 (apt) vs ROCm 7.14.0 (TheRock, official stable)** comparison on the
llama.cpp path for Muse-Glimmer-30B on gfx1151 (Strix Halo), without ever endangering the proven 7.2.1 stack or its
committed matrix. 7.14.0 is the first ROCm with **official** gfx1151 APU support (released 2026-07-16; adds Ryzen AI
MAX+ PRO 495/490/485 = gfx1151). Output: a second immutable matrix under `docs/results/matrix-714/` + a comparison
section in the docs.

## 1. Artifact (verified)

Official **stable** ROCm 7.14.0, self-contained for gfx1151:

```
https://repo.amd.com/rocm/tarball/therock-dist-linux-gfx1151-7.14.0.tar.gz
  size 1,713,449,440 B (1.60 GiB) · built 2026-07-15 · content-type binary/octet-stream
```

Not a nightly alpha (the per-family *nightly* line is frozen at `7.14.0a20260612`; the stable 7.14.0 lives on the
stable index). TheRock tarballs are relocatable ("just raw files", `/opt/rocm`-style layout), so no install step, no
apt, no conflict with 7.2.1.

## 2. Isolation principle (non-destructive + revertible)

- 7.14.0 extracted to **`~/rocm-7.14.0`** (user-owned, no sudo; `rm -rf` to revert). `/opt/rocm`-7.2.1 apt install
  untouched. (Alternative if convention preferred: `/opt/rocm-7.14.0` via sudo — functionally identical.)
- llama.cpp rebuilt into **`third_party/llama.cpp/build-714/`**; the 7.2.1 `build/` left intact.
- 7.14 matrix written to **`docs/results/matrix-714/`**; the 7.2.1 `docs/results/matrix/` cells stay byte-identical.
- 7.14.0 selected at run time only via `PATH`/`LD_LIBRARY_PATH` env — nothing system-wide changes.
- **Revert = `rm -rf ~/rocm-7.14.0 third_party/llama.cpp/build-714 docs/results/matrix-714`; unset env.**

## 3. Steps

### S0 — Pre-flight safety
- Confirm `git status` clean on `master` (done: clean).
- Snapshot the immutable baseline: `cp -r docs/results/matrix docs/results/.matrix-721-baseline-snapshot` (local
  guard only; not committed) so any accidental overwrite is recoverable.
- Confirm `third_party/llama.cpp` at `0b1bad1` (done).

### S1 — Install 7.14.0 to `~/rocm-7.14.0`
```
curl -fL -o /tmp/rocm714.tar.gz \
  https://repo.amd.com/rocm/tarball/therock-dist-linux-gfx1151-7.14.0.tar.gz
mkdir -p ~/rocm-7.14.0 && tar xf /tmp/rocm714.tar.gz -C ~/rocm-7.14.0 --strip-components=1
```
**Gate (must pass before S3):**
- `~/rocm-7.14.0/bin/rocminfo | grep gfx1151` → lists gfx1151.
- `~/rocm-7.14.0/bin/hipcc --version` → reports 7.14.0 (HIP version).
- `~/rocm-7.14.0/.info/version` → 7.14.0 (if present).
- `ldd ~/rocm-7.14.0/bin/rocminfo` + `readelf -d ~/rocm-7.14.0/lib/libhipblas.so.*` → no hardcoded `/opt/rocm`
  RPATH leaking (relocatable). All libs resolve under `~/rocm-7.14.0/lib`.
- `~/rocm-7.14.0/bin/rocm-smi --showproductname` → sees the Radeon 8060S.
- If any fail → stop, document (the tarball may need an `LD_LIBRARY_PATH`/RPATH shim).

### S2 — Harness prep (backward-compatible env overrides + explicit ROCm version)
Small, safe edits to `scripts/gguf-bench-cell.sh` (defaults unchanged = 7.2.1 behavior; CI tests unaffected):
- Server binary: `LLAMA="${LLAMA_BIN:-$HERE/third_party/llama.cpp/build/bin/llama-server}"`.
- Output dir: `MATRIX_DIR="${MATRIX_OUTDIR:-docs/results/matrix}"; OUT="$MATRIX_DIR/cell-…"; mkdir -p "$MATRIX_DIR"`.
- Manifest: add explicit `rocm_version` (parsed from the active `hipcc --version` / `.info/version`) so each cell is
  self-labeling. (7.2.1 cells are NOT backfilled — they stay immutable; their `matrix/` dir + docs identify them.)
- Add a thin `scripts/run-gguf-matrix-714.sh` wrapper that exports the 7.14 env + the two overrides and delegates to
  `run-gguf-matrix.sh`, so the 7.14 run is one reproducible command.

### S3 — Rebuild llama.cpp against 7.14.0
```
export ROCM_PREFIX=$HOME/rocm-7.14.0
cmake -S third_party/llama.cpp -B third_party/llama.cpp/build-714 \
  -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx1151 -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_HIP_COMPILER=$ROCM_PREFIX/bin/hipcc \
  -Dhip_DIR=$ROCM_PREFIX/lib/cmake/hip -DROCM_PATH=$ROCM_PREFIX
env PATH=$ROCM_PREFIX/bin:$PATH cmake --build third_party/llama.cpp/build-714 -j
```
**Gate:** `ldd build-714/bin/llama-server` → resolves hipblas/rocblas under `~/rocm-7.14.0/lib` (not `/opt/rocm`);
server boots on :8080 and answers `17 × 23 → 391` correctly.

### S4 — Re-run the matrix under 7.14.0 (Study 1/2/3)
`bash scripts/run-gguf-matrix-714.sh all` → writes `docs/results/matrix-714/cell-*.json`, identical protocol to 7.2.1
(same cells, configs, reps, warmup, seeds, randomized order), only ROCm differs.

**Scope (decision point — see §5):** run **19 of 21 cells** — all of Study 1 (4) + Study 3 (5) + Study 2's 10
non-pathological cells. **Skip the 2 known-pathological `c=16 + DFlash` cells** (`study2-17gb-np16-df1`,
`study2-dynamic-np16-df1`) and reference the existing 7.2.1 pathology evidence (METHODOLOGY §6) for those.

**Stability watch (the APU risks — #6370 GTT billing, #6165 hard freeze under sustained load):**
- Background `dmesg -wT` logger for the whole run; grep for `amdgpu`, `GTT`, `fault`, ring resets.
- Between cells: 30–60 s cooldown + `rocm-smi --showtemp` sanity (abort threshold if temp climbs out of the 58–63 °C
  band seen on 7.2.1).
- Abort + report if: amdgpu ring/GTT reset, ROCm init failure, or a cell >3× its 7.2.1 wall time (freeze precursor).
- The matrix driver already isolates failures per-cell (one crash can't cascade); each cell is an independent server.

### S5 — Cell-by-cell comparison (7.2.1 vs 7.14.0)
New `scripts/compare_rocm.py` (or extend `render_matrix.py`) joining `matrix/` ↔ `matrix-714/` on
`(study, weight, np, dflash, vision)`. Per cell: `agg_tok/s` (median, Δ%, min/max), TTFT p50/p90, TPOT, VmPeak,
acceptance (DFlash cells). Render a markdown comparison table → `docs/results/matrix-714/comparison.md`.

### S6 — Document + commit
- `docs/results/METHODOLOGY.md`: add the 7.14.0 manifest row + a "7.2.1 vs 7.14.0" subsection (methodology box:
  identical source/flags/weights/seeds; only ROCm differs; official stable 7.14.0 gfx1151 tarball provenance).
- `docs/results/benchmark.md`: new **7.2.1 vs 7.14.0** section with the comparison table + honest caveats.
- `handoff.md` §8 + `README.md`: close out the 7.14.0 "pending Part 2" item with the result.
- Commit `docs/results/matrix-714/` (cell JSONs + comparison) as immutable artifacts alongside the 7.2.1 matrix.

## 4. Risks & mitigations

| Risk | Mitigation |
|---|---|
| Hard freeze (#6165) under sustained c=16 load on newer ROCm | Skip known-pathological c=16+DFlash; watch dmesg/temp; abort-on-anomaly; per-cell isolation |
| GTT billing gap (#6370) skewing memory | Trust VmPeak (METHODOLOGY §5) for both arms; record VmHWM/rocm-smi for transparency |
| Tarball not relocatable (hardcoded `/opt/rocm`) | S1 gate checks `readelf`/`ldd`; shim with `LD_LIBRARY_PATH`/`patchelf` if needed |
| 7.14.0 userspace ↔ kernel 6.17 / amdgpu driver mismatch | S1 gate (`rocminfo` sees gfx1151, `rocm-smi` works); tarball is userspace-only, rides existing driver |
| Accidental overwrite of 7.2.1 matrix | Separate `matrix-714/` out dir + env override + S0 local snapshot; never run 7.14 cells into `matrix/` |
| llama.cpp picks up 7.2.1 at build/run | Explicit `ROCM_PATH`/`hip_DIR`/`PATH` for configure+build; `ldd` gate confirms linkage |

## 5. Scope — DECIDED (2026-08-13): 17/21, reduced first pass

**Decision: skip ALL `c=16` cells** (both baselines and DFlash) for the first pass — lowest freeze risk. Run
**17 cells**: Study 1 (4) + Study 3 (5) + Study 2 at `np ∈ {1,4}` (8). The 4 skipped cells are
`study2-{17gb,dynamic}-np16-df{0,1}`. **Add c=16 later only if this pass is stable.** The 2 `c=16 + DFlash` cells
remain documented-pathological (METHODOLOGY §6); the c=16 *baselines* are deferred, not abandoned.

Implemented in the harness via an `EXCLUDE_NPS` filter (default empty = current 7.2.1 behavior); the 7.14 wrapper
sets `EXCLUDE_NPS=16`.

## 6. Honest-reporting caveats (written into docs)
- Comparison is **ROCm 7.2.1 (apt, legacy) vs ROCm 7.14.0 (TheRock, official stable)** on the llama.cpp path; vLLM/torch
  are out of scope (the spec targets llama.cpp).
- 7.14.0 = first *official* gfx1151 ROCm; 7.2.1 was community-verified. "Official ≠ bug-free": open APU UMA issues
  persist and may affect stability/numbers — any anomaly is reported, not hidden.
- Absolute tok/s differences conflate ROCm version + compiler/runtime maturity; the cell-by-cell Δ is the finding.
