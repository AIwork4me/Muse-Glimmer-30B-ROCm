# ROCm 7.14 scoped validation track

This repository project-validated a reduced **GGUF/llama.cpp** matrix using
AMD's official ROCm 7.14.0 gfx1151 tarball on Ryzen AI MAX+ PRO 395 / Radeon
8060S. AMD's [ROCm 7.14 release notes](https://rocm.docs.amd.com/en/docs-7.14.0/about/release-notes.html)
list that exact platform as `gfx1151`; AMD's [GPU specifications](https://rocm.docs.amd.com/en/latest/reference/gpu-arch-specs.html)
also map the 395 / 8060S to RDNA 3.5 and `gfx1151`. Muse-Glimmer workload
validation and benchmark evidence here are independent project results.
**Optional / not prioritized for v0.1; ROCm 7.14 Muse-Glimmer vLLM validation
pending.** Historical vLLM/BF16 remains validated on the 7.2.1 reference.
Current rocBLAS BF16-GEMM proxy results did not justify prioritizing a 7.14
rebuild; they do not establish zero value for a future cohesive 7.14 stack.

The historical ROCm 7.2.1 evidence remains in `../matrix/`. The 19 ROCm 7.14
cells remain in `../matrix-714/`; of the four `np=16` cells, **both baselines
were measured on 2026-08-15 with the corrected SSE-framing benchmark client**
(see [`#c16`](#c16--both-baselines-measured-2026-08-15-dflash-cells-stay-deferred)),
while the two DFlash cells remain deferred (pathological scope). The
[scoped validation manifest](../../../configs/rocm-7.14-gguf-validation.json)
records the archive, runtime, engine, model and evidence identities.

## Result summary — GGUF matrix (2026-08-13)

Recorded invariants include llama.cpp `0b1bad1`, flags, model artifacts, prompt
set and seeds; the intended experimental variable is the ROCm runtime. Mean TPOT
deltas were −0.4% at `np=1` and −1.7% at `np=4`. Those cell-level means are
descriptive and do not establish a general equivalence bound or speedup.

All 17 ROCm 7.14 cells of the original pass had a lower VmPeak
mapped-address-space envelope (mean −2.8%). The operator observed no system
incident during the six-hour run, but raw dmesg/amdgpu logs were not retained.
This result is therefore not a
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
| **VmPeak envelope** | mean −2.8%; lower in all 17 original cells (range −1.8%…−7.0%); added `np=16` baselines: 17gb +16.1%, dynamic −1.6% |
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

### Table D — VmPeak mapped-address-space envelope (original 17 cells, mean −2.8%)

All 17 cells of the original pass recorded a lower VmPeak on 7.14 (range
−1.8%…−7.0%). VmPeak is virtual address-space size, not resident physical
memory; this is an observed mapped-memory envelope difference. The 2026-08-15
`np=16` baselines sit outside it: 17gb is **+16.1%** higher than its 7.2.1
twin, dynamic is −1.6% lower; both are excluded from this table's envelope
statements.

Full per-cell table (all metrics, both arms): [`../matrix-714/comparison.md`](../matrix-714/comparison.md).
Rendered 7.14 matrix: [`../matrix-714/matrix.md`](../matrix-714/matrix.md).
Per-cell summary evidence: [`../matrix-714/cell-*.json`](../matrix-714/) (each carries
`manifest.rocm_version = "7.14.0"`).

### c=16 — both baselines measured (2026-08-15); DFlash cells stay deferred

Both `study2 np=16` **baseline** cells were run to completion on 2026-08-15
(`cell-study2-{17gb,dynamic}-np16-df0-vis0.json`, `manifest.rocm_version =
"7.14.0"`) using the benchmark client **after** its SSE parsing was fixed to
split the stream by newline framing (raw-chunk parsing silently dropped
coalesced events at high concurrency; the pre-fix attempt's corrupt record
was quarantined and never published). With identical flags, weights, prompts
and seeds, both cells are healthy and closely track the 7.2.1 reference:

| metric | 17gb 7.2.1 | 17gb 7.14 | Δ | dynamic 7.2.1 | dynamic 7.14 | Δ |
|---|---|---|---|---|---|---|
| aggregate tok/s (median rep) | 34.47 | **36.97** | +7.3% | 31.05 | **32.01** | +3.1% |
| TTFT p50 / p90 (s) | 2.12 / 3.23 | 2.25 / 3.08 | +6.0% / −4.5% | 2.41 / 3.37 | 2.58 / 3.33 | +7.1% / −1.0% |
| TPOT median (s) | 0.171 | 0.179 | +4.9% | 0.237 | 0.298 | +26.1% |
| total generated tokens | 29,801 | 32,529 | +9.2% | 31,229 | 34,813 | +11.5% |
| VmPeak (GiB) | 30.27 | 35.14 | **+16.1%** | 38.41 | 37.79 | −1.6% |

These cells do not generalize to "c=16 is faster on 7.14" (no repeats, no
profiling); they establish that **c=16 baseline serving is functional and in
the same throughput band as 7.2.1**. The 17gb cell's VmPeak is *higher* than
7.2.1's — the "lower VmPeak in all cells" statement below remains scoped to
the original 17 cells.

**Provenance disclosure (dynamic cell).** During the dynamic cell's
warmup/first repetition (12:03–12:07), an unrelated `llama-bench` sweep
briefly shared the GPU. The affected repetition is visible as the min
(18.2 tok/s) against the reported median (32.0 tok/s); the median over five
repetitions is robust to it. Disclosed for provenance completeness.

**The two `np=16` DFlash cells remain deferred — the pathological
combination stands.** A bounded 16×48-token probe
(`probe-study2-17gb-np16-df1-max48-reps1.json`, `MAX_TOKENS=48`, `REPS=1`)
completed at 16.9 tok/s aggregate with 19.3% draft acceptance, but the
full-fidelity attempt (`MAX_TOKENS=512`, `REPS=5`) decayed from ~74 to ~35
prompt-tokens/min as generations lengthened and was aborted by the operator
at ~2 h with roughly 45% of the workload done — matching the 7.2.1 finding
that c=16 + DFlash degrades with sustained long-generation load
([warning](../benchmark.md#c16-dflash-do-not-use)). **Do not combine DFlash
with `-np 16` on either runtime.**

### Single-user flash-attn micro-sweep (2026-08-15, descriptive)

`llama-bench` (`pp512`/`tg128`, `r=5`, exclusive GPU, same build and
runtime; raw records: [`llama-bench-fa-sweep-17gb.json`](../matrix-714/llama-bench-fa-sweep-17gb.json),
[`...-dynamic.json`](../matrix-714/llama-bench-fa-sweep-dynamic.json)):

| weight | `-fa off` tg128 | `-fa on` tg128 | Δ | best pp512 |
|---|---|---|---|---|
| 17gb | 10.59–10.66 t/s | **10.87–10.95 t/s** | +2.3…+2.9% | 315.9 t/s (`-fa on`, `-ub 1024`) |
| dynamic | 9.26–9.34 t/s | **9.47–9.50 t/s** | +1.7% | 309.2 t/s (`-fa on`, `-ub 512`) |

`-ub` is insensitive for decode (±0.3 t/s across 256/512/1024 on both
weights); flash-attn is the actionable knob. These sweeps are **descriptive
single-run evidence** (no repeats, baseline decoding only — `llama-bench`
cannot exercise DFlash in this build), so they do not change the validated
matrix flags; they support `--flash-attn on` as a reasonable single-user
baseline knob on gfx1151. Interaction with DFlash is untested.

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
- [x] 2026-08-15: both `study2 np=16` **baseline** cells measured with the
      fixed (newline-framing) benchmark client → 19 cells total; the pre-fix
      attempt's corrupt record was quarantined, never published. The
      full-fidelity 17gb `np=16` **DFlash** attempt was aborted at ~2 h with
      decaying pace (bounded probe + record in the c=16 section); both
      DFlash cells stay deferred.
- [x] `compare_rocm.py` reviewed for one-sided/missing cells (c=16 cells noted as 7.2.1-only).
- [x] Reviewed TPOT, TTFT, aggregate tok/s, VmPeak, DFlash acceptance, vision,
  finish reasons and recorded temperature; stability evidence is operator-level.
- [x] Every new cell carries `manifest.rocm_version = "7.14.0"`.

## Phase 4 — BF16/vLLM: proxy evaluated; validation pending

The full ROCm 7.14 Muse-Glimmer vLLM rebuild and validation was not prioritized
for v0.1. A minimal proxy did not show a consistent BF16-compute improvement
across the sampled GEMM shapes. The proxy informed prioritization; it is not a
statistical equivalence test or a validation of a future cohesive stack:

- vLLM BF16 at **c=1 is bandwidth-bound** (4.2 tok/s ≈ the 56 GB-BF16/token
  bandwidth ceiling) — unchanged by ROCm version.
- vLLM BF16 at **c≥4 is GEMM(compute)-bound**, so we measured rocBLAS BF16-GEMM
  throughput on both ROCm versions (`scripts/bench_rocblas_gemm.cpp`, exercising
  the actual gfx1151 GEMM kernels each `librocblas` ships):

| GEMM (bf16) | ROCm 7.2.1 | ROCm 7.14.0 | Δ |
|---|---|---|---|
| 4096³ | 3.5 TFLOPS | 3.9 TFLOPS | +11% |
| 8192×8192×4096 | 3.6 TFLOPS | 3.6 TFLOPS | 0% |
| 12288³ | 3.1 TFLOPS | 2.9 TFLOPS | −6% |
| **mean** | **3.4** | **3.5** | **≈ +2% (noise)** |

The sampled proxy showed no consistent direction: one shape improved, one was
flat, and one regressed. Those current results did not justify prioritizing the
ROCm 7.14 vLLM rebuild for v0.1. They do not prove that a future cohesive ROCm
7.14 vLLM stack has zero user-perceivable value. Historical vLLM/BF16 validation
stays on the 7.2.1 reference; the default single-user path is Q4/llama.cpp.

Reproduce the proxy:

```bash
for V in /opt/rocm-7.2.1 "$HOME/rocm-7.14.0"; do
  PATH="$V/bin:$PATH" hipcc scripts/bench_rocblas_gemm.cpp -I"$V/include" -L"$V/lib" -lrocblas -O3 -o /tmp/gemm
  LD_LIBRARY_PATH="$V/lib" /tmp/gemm
done
```

## Publication gate

A ROCm 7.14 result may be labeled **project-validated within an explicit scope**
only when:

- completed and deferred cells are explicit: ✓ (19/21; both `np=16`
  baselines measured 2026-08-15, two `np=16` DFlash cells deferred)
- artifact, runtime and source identities are recorded: ✓
- per-cell summaries, exact flags and SHA256 inventory are committed: ✓
- absent raw response/server/system logs are disclosed: ✓
- one-sided cells and metric confounds are reviewed: ✓
- the ROCm 7.2.1 history remains unchanged: ✓

**GGUF/llama.cpp status: project-validated within the recorded 19-cell scope.**
**Optional / not prioritized for v0.1; ROCm 7.14 Muse-Glimmer vLLM validation
pending.** Of the four `np=16` GGUF cells, both baselines are measured
(healthy, [`#c16`](#c16--both-baselines-measured-2026-08-15-dflash-cells-stay-deferred));
the two DFlash cells remain deferred (pathological on 7.2.1; the 7.14
full-fidelity attempt aborted with decaying pace).
