# ROCm 7.14 validation track

ROCm 7.14 is the forward/current official gfx1151 validation target. This
directory is the protocol plus the GGUF-matrix **result** (run 2026-08-13).

The historical ROCm 7.2.1 evidence remains in `../matrix/`. The 7.14 GGUF cells
are written to `../matrix-714/` so the two tracks cannot overwrite one another.

## Result summary — GGUF matrix (2026-08-13)

**Official stable ROCm 7.14.0 (`therock-dist-linux-gfx1151-7.14.0.tar.gz`,
built 2026-07-15), installed side-by-side at `~/rocm-7.14.0`, same llama.cpp
source `0b1bad1` / flags / weights / prompt set / seeds — only ROCm differs.
17 cells (Study 1/2/3, c=16 deferred per scope). Run 11:41–17:43, zero
`dmesg`/amdgpu errors (no #6370 GTT, no #6165 freeze) across 6 h of sustained
load.**

**Headline: 7.14.0 ≈ 7.2.1 on real per-token decode throughput.** The benefit of
the first *official* gfx1151 ROCm is **support + stability + a small memory
reduction**, not a speedup — single-stream decode is bandwidth-bound, and a
bandwidth-bound kernel cannot speed up by changing ROCm.

### The metric that matters: TPOT, not aggregate tok/s

At `temp=1.0` (Study 2/3), **aggregate tok/s is length-confounded**: tiny
cross-ROCm numerical differences flip sampled tokens, generations diverge, and
the run that happens to emit longer sequences scores higher `agg_tok/s`
(`Σtokens ÷ wall`) without decoding faster. **TPOT (per-token decode cost) is
the clean cross-version throughput metric** (each token's cost is a
hardware/kernel property, independent of which token it is). Study 1 (`temp=0`,
greedy) is clean by construction — identical token streams (tokens ratio 1.00×).

| Metric (clean) | Result |
|---|---|
| **TPOT c=1** (11 cells) | mean **−0.4%** → identical |
| **TPOT c=4** (6 cells) | mean **−1.7%** (range −5.4%…+0.2%) → 7.14 within noise, marginally faster |
| **Greedy Study 1 baselines** | Δ ≤ +0.5% (identical decode) |
| **Greedy Study 1 DFlash decode** | **−5.8% / −6.4%** per-token (small real win on the spec-decode verify path) |
| **Memory (VmPeak)** | 7.14 **−2.8% mean, every cell** (range −1.8%…−7.0%) |
| **DFlash acceptance** | **identical** (Δ ≤ 1.2 pp) → confirms deltas are runtime-side, not drafter-side |
| **Stability** | **0 dmesg/amdgpu errors in 6 h** sustained — official 7.14 ran the full reduced matrix without incident |

### Table A — Study 1 (greedy, `temp=0`): clean per-token decode (TPOT)

The most defensible comparison: deterministic, identical token streams.

| weight | mode | 7.2.1 TPOT (s) | 7.14 TPOT (s) | Δ |
|---|---|---|---|---|
| 17gb | baseline | 0.0942 | 0.0947 | +0.5% |
| 17gb | DFlash | 0.0482 | 0.0454 | **−5.8%** |
| dynamic | baseline | 0.1082 | 0.1085 | +0.3% |
| dynamic | DFlash | 0.0486 | 0.0455 | **−6.4%** |

Baselines unchanged; DFlash verify-path decode ~6% faster/token on 7.14.

### Table B — c=4: TPOT (clean) vs aggregate (confounded)

| cell | 7.2.1 TPOT | 7.14 TPOT | TPOT Δ | agg tok/s Δ | tokens 7.14/7.2.1 |
|---|---|---|---|---|---|
| 17gb base | 0.1778 | 0.1750 | −1.6% | **+40.6%** | **1.36×** |
| 17gb DFlash | 0.1066 | 0.1039 | −2.5% | +18.7% | 1.20× |
| dynamic base | 0.1809 | 0.1711 | −5.4% | +0.5% | 0.94× |
| dynamic DFlash | 0.1212 | 0.1214 | +0.2% | +10.2% | 1.09× |
| 17gb base vision | 0.1768 | 0.1752 | −0.9% | −0.6% | 0.94× |
| dynamic base vision | 0.1758 | 0.1757 | −0.1% | +1.1% | 0.97× |

**Reading this table:** TPOT (per-token cost) is flat-to-marginally-faster on
7.14. The large `agg tok/s` numbers (e.g. **+40.6%** on 17gb c=4 base) track the
**tokens ratio** (1.36×), i.e. 7.14 emitted 36% more tokens from sampling
divergence — *not* a 40% decode speedup. This is the central methodological
caveat of the comparison (see [METHODOLOGY.md §12](../METHODOLOGY.md)).

### Table C — DFlash acceptance (identical ⇒ deltas are runtime-side)

| cell | 7.2.1 accept | 7.14 accept |
|---|---|---|
| study1 17gb c1 | 23.3% | 23.3% |
| study1 dynamic c1 | 23.7% | 23.6% |
| study2 17gb c1 | 20.7% | 19.4% |
| study2 17gb c4 | 18.4% | 18.6% |
| study2 dynamic c1 | 19.4% | 19.4% |
| study2 dynamic c4 | 19.2% | 19.2% |
| study3 17gb c1 | 18.8% | 18.8% |

Same drafter + same model + same `--spec-draft-n-max 16` + same seeds ⇒ the
accepted-token rate is unchanged (Δ ≤ 1.2 pp, mostly < 0.3 pp). The throughput
deltas above are therefore ROCm kernel/runtime effects, not spec-decode
behavior changes.

### Table D — memory (VmPeak GiB): 7.14 consistently lower (mean −2.8%)

Every one of the 17 cells uses less mapped memory on 7.14 (range −1.8%…−7.0%;
largest cut on the heaviest cells, e.g. 17gb c=4 DFlash −7.0%). A consistent,
real secondary improvement.

Full per-cell table (all metrics, both arms): [`../matrix-714/comparison.md`](../matrix-714/comparison.md).
Rendered 7.14 matrix: [`../matrix-714/matrix.md`](../matrix-714/matrix.md). Raw
cells: [`../matrix-714/cell-*.json`](../matrix-714/) (each carries
`manifest.rocm_version = "7.14.0"`).

### c=16 — deferred (per scope), not re-run

The 4 c=16 cells are 7.2.1-only by design (this first pass excludes c=16 to
limit sustained-load freeze risk). On 7.2.1 the c=16 **baselines** were healthy
(17gb 34.5, dynamic 31.0 tok/s) and the c=16 **DFlash** cells were pathological
([warning](../benchmark.md#c16-dflash-do-not-use)). They remain open for a
follow-up pass once this reduced pass is confirmed stable — which it now is.

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

- [x] Preserve `docs/results/matrix/` unchanged (byte-identical snapshot at `.matrix-721-baseline-snapshot/`).
- [x] Record `git status --short` and the repository commit (HEAD `97882c4`; llama.cpp `0b1bad1`).
- [x] Verify model artifacts (weights/drafter/mmproj on disk, unchanged).
- [x] Record kernel (`6.17.0-1032-oem`), GPU/gfx (`gfx1151`), ROCm prefix (`~/rocm-7.14.0`, `.info/version = 7.14.0`).
- [x] Confirm no benchmark server already bound to 8080/8090.
- [x] Keep the 7.14 install side-by-side; `/opt/rocm` (7.2.1) untouched. Tarball relocatable ($ORIGIN RPATH, no /opt/rocm leak).

## Phase 1 — environment initialization

- [x] Select the 7.14 prefix for one process only (`PATH`/`LD_LIBRARY_PATH`).
- [x] Capture the exact package source: `therock-dist-linux-gfx1151-7.14.0.tar.gz` (1.60 GiB, sha256 `2567d5e3…`, from `repo.amd.com/rocm/tarball-multi-arch/`). `hipcc --version` → HIP 7.14.60850.

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
- [x] No system instability during smoke tests.

## Phase 3 — GGUF matrix

- [x] `run-gguf-matrix-714.sh --dry-run all` → 17 cells (c=16 excluded).
- [x] `run-gguf-matrix-714.sh all` → 17/17 cells written to `matrix-714/` + rendered.
- [x] `compare_rocm.py` reviewed for one-sided/missing cells (c=16 cells noted as 7.2.1-only).
- [x] Reviewed TPOT, TTFT, aggregate tok/s, VmPeak, DFlash acceptance, vision, finish reasons, temperature, stability — not only median throughput. (See Result summary above.)
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

A ROCm 7.14 result becomes "validated" only when:

- required cells/tests are complete or explicitly recorded as negative findings; ✓ for the GGUF matrix (17/17; c=16 explicitly deferred).
- artifacts and source revisions match the manifests; ✓
- raw outputs and exact commands are committed; ✓ (`matrix-714/`)
- the comparison is reviewed for one-sided/missing cells; ✓
- README wording is updated without altering the ROCm 7.2.1 history. ✓ (2026-08-13)

**GGUF-track status: validated (reduced matrix, c=16 deferred).** The vLLM/BF16
track (Phase 4) remains pending.
