# ROCm 7.14 scoped validation track

This repository project-validated a reduced **GGUF/llama.cpp** matrix using
AMD's official ROCm 7.14.0 gfx1151 tarball on Ryzen AI MAX+ PRO 395 / Radeon
8060S. AMD's ROCm 7.14 release notes describe gfx1151 support but do not list
this exact 395 SKU; this is project evidence, not an AMD SKU-support claim.
The vLLM/BF16 track remains pending.

The historical ROCm 7.2.1 evidence remains in `../matrix/`. The 17 ROCm 7.14
cells remain in `../matrix-714/`; all four `np=16` cells were intentionally
deferred. The [scoped validation manifest](../../../configs/rocm-7.14-gguf-validation.json)
records the archive, runtime, engine, model and evidence identities.

## Result summary — GGUF matrix (2026-08-13)

Recorded invariants include llama.cpp `0b1bad1`, flags, model artifacts, prompt
set and seeds; the intended experimental variable is the ROCm runtime. Mean TPOT
deltas were −0.4% at `np=1` and −1.7% at `np=4`. Those cell-level means are
descriptive and do not establish a general equivalence bound or speedup.

All 17 ROCm 7.14 cells had a lower VmPeak mapped-address-space envelope (mean
−2.8%). The operator observed no system incident during the six-hour run, but
raw dmesg/amdgpu logs were not retained. This result is therefore not a
standalone stability qualification.

### Why TPOT is the primary cross-version metric

In sampled Study 2/3 cells, generated-token counts differ across versions.
Aggregate tok/s includes that count and is therefore length-confounded. TPOT
normalizes decode time by generated tokens and is less length-confounded, but it
still includes scheduling, workload and measurement variation. No profiler
counters or independent run-level repeats were retained.

Study 1 used greedy decoding. Its records have equal total token counts and
finish-reason distributions across versions. Raw response text and token hashes
were not retained, so the repository does not claim identical token streams.

| Observed metric | Result |
|---|---|
| **TPOT `np=1`** (11 cells) | mean **−0.4%**, range −6.4%…+9.0% |
| **TPOT `np=4`** (6 cells) | mean **−1.7%**, range −5.4%…+0.2% |
| **Greedy Study 1 baselines** | +0.5% / +0.3% TPOT |
| **Greedy Study 1 DFlash** | −5.8% / −6.4% TPOT; causal attribution needs repeats/profiling |
| **VmPeak envelope** | mean −2.8%; lower in all 17 cells (range −1.8%…−7.0%) |
| **DFlash acceptance** | similar; largest observed difference 1.21 percentage points |
| **Run observation** | no incident observed in six hours; raw system logs not retained |

### Table A — Study 1 (greedy, `temp=0`): TPOT

This is the least length-confounded subset. It is descriptive, not an
equivalence or causal performance proof.

| weight | mode | 7.2.1 TPOT (s) | 7.14 TPOT (s) | Δ |
|---|---|---|---|---|
| 17gb | baseline | 0.0942 | 0.0947 | +0.5% |
| 17gb | DFlash | 0.0482 | 0.0454 | **−5.8%** |
| dynamic | baseline | 0.1082 | 0.1085 | +0.3% |
| dynamic | DFlash | 0.0486 | 0.0455 | **−6.4%** |

The two DFlash cells had ~6% lower TPOT on 7.14; independent repeats and
profiling are needed before attributing that observation to the runtime.

### Table B — c=4: TPOT (less length-confounded) vs aggregate tok/s

| cell | 7.2.1 TPOT | 7.14 TPOT | TPOT Δ | agg tok/s Δ | tokens 7.14/7.2.1 |
|---|---|---|---|---|---|
| 17gb base | 0.1778 | 0.1750 | −1.6% | **+40.6%** | **1.36×** |
| 17gb DFlash | 0.1066 | 0.1039 | −2.5% | +18.7% | 1.20× |
| dynamic base | 0.1809 | 0.1711 | −5.4% | +0.5% | 0.94× |
| dynamic DFlash | 0.1212 | 0.1214 | +0.2% | +10.2% | 1.09× |
| 17gb base vision | 0.1768 | 0.1752 | −0.9% | −0.6% | 0.94× |
| dynamic base vision | 0.1758 | 0.1757 | −0.1% | +1.1% | 0.97× |

**Reading this table:** the +40.6% aggregate result coincides with 1.36× as many
generated tokens, while TPOT changed −1.6%. It is not evidence of a 40% decode
speedup. TPOT reduces this length confound but does not isolate a kernel-level
cause (see [METHODOLOGY.md §12](../METHODOLOGY.md)).

### Table C — observed DFlash acceptance

| cell | 7.2.1 accept | 7.14 accept |
|---|---|---|
| study1 17gb c1 | 23.3% | 23.3% |
| study1 dynamic c1 | 23.7% | 23.6% |
| study2 17gb c1 | 20.7% | 19.4% |
| study2 17gb c4 | 18.4% | 18.6% |
| study2 dynamic c1 | 19.4% | 19.4% |
| study2 dynamic c4 | 19.2% | 19.2% |
| study3 17gb c1 | 18.8% | 18.8% |

The recorded drafter, model, draft limit and seeds are the same. Acceptance
rates are similar; the largest observed difference is 1.21 percentage points.
This evidence does not by itself prove which runtime component caused a TPOT
change.

### Table D — VmPeak mapped-address-space envelope (mean −2.8%)

All 17 cells recorded a lower VmPeak on 7.14 (range −1.8%…−7.0%). VmPeak is
virtual address-space size, not resident physical memory; this is an observed
mapped-memory envelope difference.

Full per-cell table (all metrics, both arms): [`../matrix-714/comparison.md`](../matrix-714/comparison.md).
Rendered 7.14 matrix: [`../matrix-714/matrix.md`](../matrix-714/matrix.md).
Per-cell summary evidence: [`../matrix-714/cell-*.json`](../matrix-714/) (each carries
`manifest.rocm_version = "7.14.0"`).

### c=16 — deferred (per scope), not re-run

The 4 c=16 cells are 7.2.1-only by design (this first pass excludes c=16 to
limit sustained-load freeze risk). On 7.2.1 the c=16 **baselines** were healthy
(17gb 34.5, dynamic 31.0 tok/s) and the c=16 **DFlash** cells were pathological
([warning](../benchmark.md#c16-dflash-do-not-use)). All four remain deferred;
the reduced run does not qualify the omitted high-concurrency scope.

---

## Protocol

## Invariants

Keep these fixed unless the experiment explicitly studies them:

- llama.cpp commit from `configs/validated-stack.json`
- model revision, byte size and SHA256 from `configs/artifact-manifest.json`
- prompt set and fixed image
- cell flags, seeds, repetitions and warmup
- hardware, BIOS memory allocation and kernel
- no unrelated GPU load

Record every intentional difference.

## Phase 0 — safety and provenance

- [x] Preserve `docs/results/matrix/`; its Git tree matches base commit `97882c4`.
- [x] Record repository base `97882c40347329b7d7b471bf1a586f7481e18494`
  and llama.cpp `0b1bad14ff204627636aeb1de22ddcd5acb859d4`.
- [x] Verify the GGUF artifact set from `configs/artifact-manifest.json`.
- [x] Record kernel `6.17.0-1032-oem`, hardware/gfx identity and the side-by-side
  ROCm 7.14.0 prefix.
- [x] Confirm no benchmark server already bound to 8080/8090.
- [x] Keep the 7.14 install side-by-side; the ROCm 7.2.1 system stack stayed
  untouched.

## Phase 1 — environment initialization

- [x] Select the 7.14 prefix for one process only (`PATH`/`LD_LIBRARY_PATH`).
- [x] Record the official archive size (1,713,449,440 bytes) and full SHA256
  `2567d5e34e470db104a62a02c36aa770cb0430175e48c1c46df0eefc05e1d77c`.
- [x] Record HIP `7.14.60850-0000000` and LLVM build identity in the scoped
  validation manifest.

```bash
export ROCM_PREFIX="$HOME/rocm-7.14.0"
export PATH="$ROCM_PREFIX/bin:$PATH"
export LD_LIBRARY_PATH="$ROCM_PREFIX/lib:${LD_LIBRARY_PATH:-}"
hipcc --version
```

## Phase 2 — llama.cpp build and smoke

- [x] Build pinned source `0b1bad1` into `build-714/` (`-DGGML_HIP=ON -DAMDGPU_TARGETS=gfx1151`, `ROCM_PATH`/`hip_DIR` → 7.14).
- [x] Binary links to 7.14 under the 7.14 env (no `/opt/rocm-7.2.1` leak).
- [x] Text model loads and answers `17 × 23 → 391` (greedy smoke).
- [x] DFlash log shows non-zero drafted/accepted tokens (see cell `/metrics` probe).
- [x] Operator observed no incident during smoke tests; standalone system logs
  were not retained.

## Phase 3 — GGUF matrix

- [x] `run-gguf-matrix-714.sh --dry-run all` → 17 cells (c=16 excluded).
- [x] `run-gguf-matrix-714.sh all` → 17/17 cells written to `matrix-714/` + rendered.
- [x] `compare_rocm.py` reviewed for one-sided/missing cells (c=16 cells noted as 7.2.1-only).
- [x] Reviewed TPOT, TTFT, aggregate tok/s, VmPeak, DFlash acceptance, vision,
  finish reasons and recorded temperature; stability evidence is operator-level.
- [x] Every new cell carries `manifest.rocm_version = "7.14.0"`.

## Phase 4 — BF16/vLLM validation

Separate from the GGUF matrix; remains pending until a matching 7.14/TheRock
Python stack is installed without disturbing the historical one.

- [ ] Python/runtime environment captured.
- [ ] BF16 artifacts hash-verified.
- [ ] vLLM source commit and patches recorded.
- [ ] Model initialization completes.
- [ ] `TRITON_ATTN` serve path completes text and vision requests.
- [ ] Reasoning and ATEM tool-call parsers validated.
- [ ] TTFT, TPOT and aggregate throughput captured.
- [ ] Memory methodology includes VmPeak, RSS/HWM and stronger counters where available.
- [ ] Long-context sanity test completed with the exact tested length recorded.
- [ ] Sustained stability window and request count recorded.
- [ ] DFlash status reported separately; no inference from llama.cpp to vLLM.

## Publication gate

A ROCm 7.14 result may be labeled **project-validated within an explicit scope**
only when:

- completed and deferred cells are explicit: ✓ (17/21; four `np=16` deferred)
- artifact, runtime and source identities are recorded: ✓
- per-cell summaries, exact flags and SHA256 inventory are committed: ✓
- absent raw response/server/system logs are disclosed: ✓
- one-sided cells and metric confounds are reviewed: ✓
- the ROCm 7.2.1 history remains unchanged: ✓

**GGUF/llama.cpp status: project-validated within the recorded 17-cell scope.**
The vLLM/BF16 track and all four `np=16` cells remain pending/deferred.
