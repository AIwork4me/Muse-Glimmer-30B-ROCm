# Plan — Part 2: ROCm 7.14.0 isolation comparison (7.2.1 vs 7.14.0)

- **Date:** 2026-08-13
- **Status:** Complete — scoped GGUF/llama.cpp validation (17/21 cells)
- **Spec source of truth:** `docs/superpowers/specs/2026-08-12-llamacpp-dflash-benchmark-design.md` §9 (P-D)
- **Baseline (immutable):** the 21-cell ROCm **7.2.1** matrix at `docs/results/matrix/`
- **Intended controlled variable:** ROCm runtime; recorded invariants are llama.cpp
  commit, model artifacts, flags, prompts, seeds and harness.

## Outcome and deviations from the draft

- Evidence base: `97882c40347329b7d7b471bf1a586f7481e18494`; evidence
  commit: `1677fb42d57c6c06c2004f6e99150b6037dc4db3`.
- Completed 17 of 21 planned cells. All four `np=16` cells were deferred.
- The archive came from `tarball-multi-arch/`; extraction used the archive
  root directly, without `--strip-components`.
- The operator observed no incident during the six-hour run, but raw
  dmesg/amdgpu logs were not retained. No standalone stability claim is made.
- The BF16/vLLM track was evaluated and not pursued (rocBLAS BF16-GEMM proxy: 7.14 ≈ 7.2.1 on gfx1151 compute; no UX benefit).
- Authoritative identities and scope:
  `configs/rocm-7.14-gguf-validation.json`.

---

## 0. Goal

Produce a cell-by-cell comparison on the llama.cpp path while preserving the
historical ROCm 7.2.1 matrix. ROCm 7.14.0 provides an official AMD gfx1151
distribution; AMD's release notes do not list the exact Ryzen AI MAX+ PRO 395
used here. Output is a separate, scoped evidence matrix and comparison.

## 1. Artifact (verified)

Official ROCm 7.14.0 gfx1151 tarball, published 2026-07-15:

```text
https://repo.amd.com/rocm/tarball-multi-arch/therock-dist-linux-gfx1151-7.14.0.tar.gz
size: 1,713,449,440 bytes
sha256: 2567d5e34e470db104a62a02c36aa770cb0430175e48c1c46df0eefc05e1d77c
```

The archive was extracted to a side-by-side prefix. The system ROCm 7.2.1
installation remained intact.

## 2. Isolation principle (non-destructive + revertible)

- 7.14.0 extracted to **`~/rocm-7.14.0`**; the system ROCm 7.2.1 install stayed
  untouched.
- llama.cpp rebuilt into **`third_party/llama.cpp/build-714/`**; the 7.2.1 `build/` left intact.
- 7.14 matrix written to **`docs/results/matrix-714/`**; the 7.2.1 `docs/results/matrix/` cells stay byte-identical.
- 7.14.0 selected at run time only via `PATH`/`LD_LIBRARY_PATH` env — nothing system-wide changes.
- Local runtime/build artifacts may be removed after target verification.
  Committed evidence is changed only through reviewed Git history.

## 3. Steps

### S0 — Pre-flight safety
- Confirm `git status` clean on `master` (done: clean).
- Snapshot the immutable baseline: `cp -r docs/results/matrix docs/results/.matrix-721-baseline-snapshot` (local
  guard only; not committed) so any accidental overwrite is recoverable.
- Confirm `third_party/llama.cpp` at `0b1bad1` (done).

### S1 — Install 7.14.0 to `~/rocm-7.14.0`
The executed run used the `tarball-multi-arch/` archive and extracted its root
without `--strip-components`. The verified reproduction commands now live in
[`docs/strix-halo-setup.md`](../../strix-halo-setup.md#alternative-rocm-7140).
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
`bash scripts/run-gguf-matrix-714.sh all` wrote `docs/results/matrix-714/cell-*.json`.
Recorded invariants include configs, repetitions, warmup, seeds and cell flags;
the intended experimental variable was the ROCm runtime.

**Actual scope:** 17 of 21 planned cells — Study 1 (4), Study 3 (5), and
Study 2 at `np ∈ {1,4}` (8). All four `np=16` cells were deferred.

**Stability watch (the APU risks — #6370 GTT billing, #6165 hard freeze under sustained load):**
- Background `dmesg -wT` logger for the whole run; grep for `amdgpu`, `GTT`, `fault`, ring resets.
- Between cells: 30–60 s cooldown + `rocm-smi --showtemp` sanity (abort threshold if temp climbs out of the 58–63 °C
  band seen on 7.2.1).
- Abort + report if: amdgpu ring/GTT reset, ROCm init failure, or a cell >3× its 7.2.1 wall time (freeze precursor).
- The matrix driver already isolates failures per-cell (one crash can't cascade); each cell is an independent server.

### S5 — Cell-by-cell comparison (7.2.1 vs 7.14.0)
Implemented `scripts/compare_rocm.py`, joining on
`(study, weight, np, dflash, vision)`. It reports one-sided cells, renders
latency in milliseconds, leads with TPOT and retains aggregate tok/s with its
length-confound warning.

### S6 — Document + commit
- Committed the 17 per-cell summaries and derived comparison separately from
  the historical matrix.
- Documented scoped status, provenance, metric limitations and pending BF16 work.

## 4. Risks & mitigations

| Risk | Mitigation |
|---|---|
| Hard freeze (#6165) under sustained c=16 load on newer ROCm | Skip known-pathological c=16+DFlash; watch dmesg/temp; abort-on-anomaly; per-cell isolation |
| GTT billing gap (#6370) skewing memory | Treat VmPeak as a mapped-address-space envelope; retain VmHWM/rocm-smi for context |
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
- The comparison is scoped to llama.cpp/GGUF; vLLM/PyTorch was evaluated and not pursued (no 7.14 compute gain).
- ROCm 7.14.0 is an official AMD gfx1151 distribution, but the release notes do
  not list the exact 395 SKU used for this independent project validation.
- Aggregate tok/s is length-confounded for sampled cells; TPOT is primary but
  does not establish a causal kernel mechanism or an equivalence bound.
- Raw response, server and system logs were not retained; stability evidence is
