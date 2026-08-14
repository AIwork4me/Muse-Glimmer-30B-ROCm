# METHODOLOGY — llama.cpp DFlash benchmark on gfx1151 (Strix Halo)

> This is the **science layer**: every number published in
> [`benchmark.md`](benchmark.md) and every cell in
> [`matrix/`](matrix/) points here for its definition. Spec source of
> truth: `docs/superpowers/specs/2026-08-12-llamacpp-dflash-benchmark-design.md`
> (§6 methodology, §4 inputs). Raw per-cell records: `matrix/cell-*.json`.

The benchmark is a **3-study matrix** measured by a custom streaming harness
(`scripts/bench_client.py` + `scripts/gguf-bench-cell.sh`) driving a per-cell
`llama-server` process. Each cell is an **independent server** started with its
exact flags, warmed up, measured, and torn down — llama.cpp startup is ~1 s so
per-cell restarts are cheap and isolate failures.

---

## 1. Hardware / software manifest

The machine-readable source of truth is
[`configs/validated-stack.json`](../../configs/validated-stack.json). Exact
model revisions, sizes, and SHA256 digests are in
[`configs/artifact-manifest.json`](../../configs/artifact-manifest.json). The
table below describes the historical ROCm 7.2.1 run; it is not a statement that
every listed version is the current upstream release.

| Item | Value |
|---|---|
| GPU | AMD Radeon 8060S, **gfx1151** (RDNA 3.5), 40 CUs |
| Memory | 94 GiB unified LPDDR5X (~215 GB/s unified fabric) |
| ROCm | 7.2.1 (community-verified for gfx1151) |
| Kernel | `6.17.0-1032-oem` (project reference floor ≥ 6.16.9 for the observed UMA issue; not a universal ROCm requirement) |
| llama.cpp | **v1, commit `0b1bad14ff204627636aeb1de22ddcd5acb859d4`**, HIP build, `-DAMDGPU_TARGETS=gfx1151` |
| Server binary | `third_party/llama.cpp/build/bin/llama-server` |
| Weights | `meta-models/Muse-Glimmer-30B-GGUF` — see [`matrix/gguf-manifest.md`](matrix/gguf-manifest.md) |

| Weight file | Size | Role |
|---|---|---|
| `muse-glimmer-30B-kquant-17gb.gguf` | 15.6 GiB | 4-bit K-quant, Meta's 24 GB-class target (the row Meta published) |
| `muse-glimmer-30B-kquant-dynamic.gguf` | 18.3 GiB | 4-bit dynamic K-quant, 32 GB-class (novel — Meta has no DFlash number for it) |
| `dflash-kquant.gguf` | 1.5 GiB | DFlash drafter (5 layers, block_size 16, SWA 2048, 32q/8kv GQA) |
| `mmproj-kquant.gguf` | 1.3 GiB | Vision projector for the multimodal path (Study 3) |

> **Fetch note (xet-backed artifacts).** `dflash-kquant.gguf` and
> `mmproj-kquant.gguf` are Xet-backed. The official Hugging Face endpoint is the
> project default; `HF_ENDPOINT` may select a mirror when regional access
> requires it. Some mirrors do **not** proxy Xet's CAS, so fetch these two from
> `huggingface.co` with `HF_HUB_DISABLE_XET=1` if a mirror fails. See
> [troubleshooting.md#dflash-mmproj-xet](../troubleshooting.md#dflash-mmproj-xet).

---

## 2. The 3 studies

### Study 1 — Meta-aligned DFlash anchor (greedy, batch 1)

**Purpose:** provide the closest reproducible comparison to Meta's published
RTX 5090 3.1× and M5 Max 1.8× rows. It aligns the disclosed controls—llama.cpp
for the RTX row, K-Quant-17GB plus quantized drafter, greedy decoding, and batch
1—but Meta's exact prompt corpus and full harness are not public.

| Axis | Value |
|---|---|
| weight | `{17gb, dynamic}` |
| DFlash | `{off, on}` (4 cells total) |
| decoding | **greedy** (`--temp 0`, `--seed 0`) |
| concurrency | **batch 1** (`-np 1`) |
| prompt set | 6-prompt diverse set (code / math / factual / creative / reasoning / instruction), averaged |
| max_tokens | 256 (pure decode budget) |
| per-slot context | 8192 (`-c 8192`) — far above max output, no truncation |
| reps | **3** per cell (output is deterministic at greedy → timing noise only) |

Reported per cell: agg tok/s (median + min/max), TTFT p50/p90, TPOT median,
VmPeak, **draft acceptance** (DFlash cells only), finish_reason distribution,
**byte-equivalence vs baseline** (the spec-decode exactness check).

**Exact serve flags (single source of truth):**
```
-m models/<weight>.gguf -ngl 999 -np 1 -c 8192 --jinja --temp 0 --seed 0 \
  --metrics --port 8080 --host 127.0.0.1
# DFlash cells append:
-md models/dflash-kquant.gguf -ngld 99 --spec-type draft-dflash --spec-draft-n-max 16
```

### Study 2 — Throughput under load (temp 1.0)

**Purpose:** realistic serving throughput at concurrency. **Labeled
Original to this repository** — Meta has no batch>1 DFlash data; this is
original throughput-under-load research.

| Axis | Value |
|---|---|
| weight | `{17gb, dynamic}` |
| concurrency `-np` | `{1, 4, 16}` |
| DFlash | `{off, on}` (12 cells, of which 2 c=16 DFlash are pathological non-completions) |
| decoding | **sampling** `--temp 1.0 --top-p 0.95 --top-k 64 --seed 42` (Meta's serving config) |
| `reasoning_strength` | `high` (the model default; pinned + reported) |
| max_tokens | 512 |
| per-slot context | 8192 ⇒ `-c = np × 8192` (no truncation trap) |
| reps | **5** per cell (output varies under sampling → more variance); c=16 baselines 1–5 reps |

Reported per cell: **aggregate tok/s** (`Σ tokens ÷ max per-request wall`),
TTFT p50/p90, TPOT median, VmPeak, finish_reason distribution, draft acceptance
(DFlash cells).

**Exact serve flags:**
```
-m models/<weight>.gguf -ngl 999 -np <1|4|16> -c <np×8192> --jinja \
  --temp 1.0 --seed 42 --metrics --port 8080 --host 127.0.0.1 \
  --top-p 0.95 --top-k 64
# DFlash cells append:
-md models/dflash-kquant.gguf -ngld 99 --spec-type draft-dflash --spec-draft-n-max 16
```

### Study 3 — Vision axis

**Purpose:** exercise multimodal loading and image-conditioned generation via
`--mmproj`, and measure the mapped-memory delta vs text-only (a comparison
with Meta's published ~+2 GB vision delta).

| Axis | Value |
|---|---|
| cells | `{17gb, dynamic} × {vision} × {c=1, c=4}` (4 baselines) + `{17gb} × {vision + DFlash} × {c=1}` (1 cell) |
| decoding | sampling (same as Study 2) |
| image | `scripts/prompt-sets/test-image.png` (fixed, committed reproducibility artifact) |
| reps | 3 |

Reported per cell: same metrics + **VmPeak delta vs the text-only cell at the
same weight/np**.

**Exact serve flags:** Study 2 flags, replacing `-c <np×8192>` per cell, and
appending `--mmproj models/mmproj-kquant.gguf`. The DFlash vision cell appends
both `--mmproj …` and the `-md … --spec-type draft-dflash --spec-draft-n-max 16`
block.

> No c=16 vision cell (heavy, not meaningful — and c=16 + DFlash is pathological
> in any case; see §6 below and the [warning](#c16-dflash-pathology) in
> this doc).

---

## 3. Prompt set

Single source of truth: [`scripts/prompt-sets/muse-glimmer-diverse.json`](../../scripts/prompt-sets/muse-glimmer-diverse.json)
(`id: muse-glimmer-diverse`, `version: 1`) — 6 short, well-defined prompts
spanning code, math, factual QA, creative, reasoning, instruction. Matching
Meta's "diverse prompt set" qualitatively (Meta does not publish their exact
prompts). Each prompt is short so decode dominates wall time. Study 3 cells
prepend the fixed test image to the user message.

---

## 4. Metric definitions (spec §6.1, verbatim semantics)

| Metric | Definition | How captured |
|---|---|---|
| **decode tok/s (Study 1)** | `Σ generated tokens ÷ max per-request wall (t1−t0)` — the same formula used for Study 2/3 (decode-dominated at np=1; the `llama-bench` tg128 cross-check supports this interpretation) | counted from the SSE stream per-request; cross-checked vs server `usage.completion_tokens`. Includes reasoning tokens — by definition the total decode rate of the model. |
| **aggregate tok/s (Study 2/3)** | `Σ generated tokens across all concurrent requests ÷ max per-request wall` | per-request stream counts, summed; matches the v1 harness so cross-run comparisons hold |
| **TTFT** | request send → first streamed chunk; **p50 and p90** reported | `time.perf_counter()` around the SSE stream; loopback HTTP overhead negligible |
| **TPOT** | `(first→last token wall) ÷ (tokens after first)`; **per-request median** reported | inter-chunk timestamps from the stream — the core DFlash metric (it lowers per-token decode cost) |
| **peak memory** | process **VmPeak** from `/proc/<pid>/status` (see §5 — VmHWM and rocm-smi both undercount on this APU) | polled during the cell by `scripts/capture_proc.py` |
| **DFlash acceptance** | `acceptance_rate = accepted_draft_tokens ÷ draft_tokens`, plus `avg_accepted_per_step` | **primary: parsed from the `llama-server` stderr `draft acceptance=…` log line** (carries mean length); **secondary: `/metrics` counters** (`llamacpp:spec_decode_num_{draft_tokens,accepted_tokens,drafts}_total`), which populate when `--spec-type draft-dflash` is engaged. The two rates match exactly; only `avg_accepted_per_step` differs (different denominator). |
| **byte-equivalence (Study 1)** | DFlash canonical response message byte-identical to baseline for each greedy prompt | `scripts/check_dflash_equiv.sh` — runs two servers over the six-prompt corpus plus arithmetic smoke and compares messages. Greedy spec-decode is exact, so a mismatch would be a correctness finding. |
| **finish_reason distribution** | count of `stop` vs `length` across reps | server response field — Meta's truncation trap: high `length` ratio ⇒ wrong output / context budget. Reported so a reader can see the truncation risk. |
| **power/temp** | GPU temp (°C) during run | `rocm-smi --showtemp` (these ARE valid on Strix Halo, unlike VRAM). Power key not exposed for this APU; `power_w` left null. |

### Per-cell JSON schema

The descriptive Draft 2020-12 definition is
[`schemas/benchmark-cell-v1.schema.json`](../../schemas/benchmark-cell-v1.schema.json).
It covers both completed and pathological/non-completing historical cells.

Each cell record (`matrix/cell-*.json`) carries, verbatim:

```json
{
  "study": "study1|study2|study3",
  "weight": "17gb|dynamic",
  "dflash": false,
  "vision": false,
  "np": 1,
  "metrics": { "agg_tok_s": 10.484, "agg_tok_s_min": 10.479, "agg_tok_s_max": 10.509,
               "reps": 3, "ttft_p50": 0.460, "ttft_p90": 0.470, "tpot_median": 0.0942,
               "total_tokens": 1536, "wall_s": 146.507, "finish_reason_dist": {"length": 18} },
  "mem":    { "VmPeak_gib": 23.91, "VmHWM_gib": 1.38, "vram_used_mib": 1084.0, "vram_total_mib": 32768.0 },
  "acceptance": { "accepted_draft_tokens": 0, "draft_tokens": 0, "acceptance_rate": null },
  "acceptance_source": "metrics|log",
  "power_temp": { "power_w": null, "temp_c": 63.0 },
  "manifest": { "flags": "<exact serve flags>",
                "seed": 0, "reps": 3, "build": "0b1bad1",
                "rocm": "<gfx1151 json>", "kernel": "6.17.0-1032-oem", "date": "2026-08-12" }
}
```

Pathological (non-completing) cells carry `pathological: true` plus a
`pathological_evidence` block instead of metrics — see the c=16 DFlash cells.

---

<a id="memory-methodology"></a>

## 5. Memory methodology — the process mapped-memory envelope

This is the most misread column on Strix Halo, so it gets its own section.

| Source | What it measures | On this APU |
|---|---|---|
| **VmPeak** (`/proc/<pid>/status`) | peak virtual address-space size of the `llama-server` process | **most useful process-level mapped-memory envelope here (~24–32 GiB)** — it includes the mmap'd GGUF and GPU-offload mappings, but is not a direct measurement of resident physical memory |
| `VmHWM` / `VmRSS` | peak resident *physical* pages owned by the process | **undercounts (1–10 GiB)** — mmap'd pages are paged in/out by the kernel and many GPU-offloaded pages are not counted as resident |
| `rocm-smi --showmeminfo vram` | the ~32 GiB **dedicated VRAM carve-out** only | **misleading (~1 GiB)** — the carve-out counter only ticks for buffers allocated through the carve-out path; mmap'd + unified-host-visible GPU buffers don't increment it |

**Therefore: every published footprint number in this matrix is labeled
VmPeak.** The renderer (`scripts/render_matrix.py`) emits VmPeak as the
footprint column. VmHWM and rocm-smi VRAM are kept in the cell JSON for
transparency but are not the headline number. VmPeak is useful for relative
deltas in this workload; it must not be interpreted as literal resident DRAM
use.

Cross-checks that VmPeak is sane:

- DFlash cells add the ~1.5 GiB drafter + ~1.5 GiB working set → ~+2.5–3 GiB
  VmPeak vs baseline (matrix: 17gb 23.9 → 26.7; dynamic 26.6 → 29.3). Matches
  Meta's ~+3 GB drafter envelope.
- Vision cells add the mmproj projector → ~+2 GiB VmPeak vs text-only at the
  same weight/np (Study 3: 17gb c=1 24.4 → 26.2; dynamic c=1 27.2 → 29.0).
  Matches Meta's ~+2 GB vision envelope.
- Aggregate concurrency raises VmPeak via KV cache (17gb c=1 24.4 → c=16 30.3;
  dynamic c=1 27.2 → c=16 38.4).

Future runs should add `/proc/<pid>/smaps_rollup` (RSS/PSS), system
`MemAvailable` deltas, cgroup accounting where available, and relevant
GTT/unified-memory counters. Those measurements were not captured for the
historical matrix and are not reconstructed here.

---

<a id="c16-dflash-pathology"></a>

## 6. The c=16 + DFlash pathology

> **Headline user-facing warning. Do NOT combine DFlash with `-np 16` (high
> concurrency). It is pathologically slow — >1000× slower per-request than the
> c=16 baseline.** See [benchmark.md warning block](benchmark.md#c16-dflash-do-not-use)
> for the public-facing write-up and best-practice table, and
> [troubleshooting.md#dflash-c16-pathological](../troubleshooting.md#dflash-c16-pathological).

The short version, for the science record:

- **17gb c=16 DFlash probe** (`cell-study2-17gb-np16-df1-vis0.json`): a 16×48-
  token batch completed **0 of 768 tokens in 27.7 s** while the baseline c=16
  run delivered 34.47 tok/s aggregate — effectively >1000× slower per-request.
  Probe-time acceptance was 0.182 with per-slot acceptance 0.065–0.091.
- **dynamic c=16 DFlash attempt** (`cell-study2-dynamic-np16-df1-vis0.json`):
  the full REPS=5 cell was **aborted after 5 h 16 m** with no completion. At
  abort, the drafter had emitted **3,270,000 draft tokens** of which **6,060
  were accepted** — an acceptance rate of **0.0018 (0.18 %)**, ~1/37× the
  baseline aggregate rate.
- **Observed proximate mechanism:** at `-np 16` the drafter fires for all 16 slots simultaneously,
  generating an enormous draft volume that is almost entirely rejected (>99.8 %
  in the dynamic run) while the full generate+verify compute for *all those
  rejected drafts* is paid in full. The acceptance collapse explains why
  spec-decode costs more than it saves in these cells. The matrix does not
  isolate a deeper causal split among drafter behavior, scheduling, and kernels.
- **c=16 itself is fine** (baseline 17gb 34.5 tok/s, dynamic 31.0 tok/s). The
  pathology is **DFlash-specific**.

Both c=16 DFlash cells are recorded as evidence-based non-completions
(`pathological: true`); they are not missing data, they are *findings*.

---

## 7. Statistical protocol (spec §6.2)

- **Study 1 (greedy):** 3 reps. Output is deterministic at `temp=0`, so reps
  measure timing noise only. Report **median + min/max** (`agg_tok_s`,
  `agg_tok_s_min`, `agg_tok_s_max` in the cell JSON).
- **Study 2 (sampling):** 5 reps. Output varies under `temp=1.0`, so the larger
  rep count controls variance. Report **median + min/max**. (c=16 baselines ran
  1–5 reps due to wall-time budget — see each cell's `reps` field.)
- **Study 3 (sampling):** 3 reps.
- **Warmup:** 2 warmup requests discarded before each measured cell (thermal +
  JIT + cache).
- **Cell order randomized** within a study (`scripts/run-gguf-matrix.sh`
  shuffles) and the order recorded, to kill systematic thermal drift across a
  long matrix run. Temps 58–63 °C observed across the whole matrix — no thermal
  throttling.
- **Fairness controls:** same hardware, same ROCm 7.2.1, same llama.cpp build
  `0b1bad1`, same warmup, same prompt set, no other GPU load. Only the cell
  config varies (weight file, ±`-md`, ±`--mmproj`, `-np`, sampling vs greedy).
- **Raw distributions, not just point estimates:** `agg_tok_s_min/max`,
  TTFT p50/p90, TPOT median, finish_reason distribution all live in each cell.

---

<a id="dflash-enablement"></a>

## 8. DFlash enablement — the silent no-op gotcha

`llama-server`'s `--spec-type` **defaults to `none`**. Passing only
`-md dflash.gguf -ngld 99` loads the draft model but **never drafts** — a
silent 1.0× no-op. The fix is mandatory:

```
-md models/dflash-kquant.gguf -ngld 99 \
  --spec-type draft-dflash --spec-draft-n-max 16
```

`--spec-draft-n-max 16` is the **measured sweet spot** — it equals the DFlash
drafter's block_size. A pre-matrix sweep gave n_max `3 / 8 / 16 / 32` →
**1.14× / 1.51× / 1.60× / 1.60×** speedup; n_max 16 is the elbow (32 is flat,
so 16 wins on memory). The cell `flags` field on every DFlash cell shows the
full block above — by construction, the harness cannot ship a silent-no-op
DFlash run.

> This is the single most important "is DFlash even on?" check. A DFlash cell
> that shows 1.0× with null acceptance has fallen into this trap. The matrix
> records non-zero draft activity and 2.20× / 2.39× Study 1 speedups.

---

## 9. Byte-equivalence (greedy spec-decode exactness)

Speculative decoding under **greedy** sampling is an exact equivalence: the
accepted tokens are by definition the argmax the target would have produced, so
DFlash-on and DFlash-off must emit the identical token sequence. A mismatch is a
correctness finding.

The current [`scripts/check_dflash_equiv.sh`](../../scripts/check_dflash_equiv.sh)
runs baseline and DFlash servers over all six Study 1 prompts, plus the original
arithmetic smoke prompt, and compares canonical response-message bytes. It also
requires non-zero DFlash draft activity so a silent no-op cannot pass.

**Recorded historical result:** the original arithmetic smoke check passed—both
servers emitted `'391'` (17×23 = 391, correct). The check script was the subject
of fix commit `290e596`: an earlier version omitted `--spec-type` and therefore
passed trivially against a no-drafting server. The expanded six-prompt harness
is now available, but its result must not be published until it is run on the
validated GPU stack.

---

## 10. Known limitations & honest caveats

1. **Unified ~215 GB/s fabric vs RTX 5090 GDDR7.** Absolute tok/s will differ
   across hardware. The speedup ratio is the closest methodology-aligned
   comparison to Meta, subject to the unpublished-prompt limitation above. We
   report ratios alongside absolute numbers.
2. **Reasoning model tok/s measures raw decode, not per-turn answer latency.**
   Muse-Glimmer emits chain-of-thought in the `reasoning` channel first; a real
   chat turn is longer than `tokens / tok_s`. The `reasoning` tokens are
   included in tok/s by definition (it is the model's total decode rate).
3. **`finish_reason = length` is common.** With `max_tokens=256` (Study 1) or
   `512` (Study 2/3), and `reasoning_strength=high` (the default, not
   switchable off — `--reasoning off` is a no-op), many runs hit the cap before
   EOS. This is expected and does **not** invalidate tok/s (computed over the
   generated tokens); it does mean the per-cell `finish_reason_dist` is part of
   the record so the truncation risk is visible. `--reasoning_strength low`
   shortens thinking if you need shorter turns.
4. **`rocm-smi` VRAM is not the footprint** on Strix Halo — see §5.
5. **`power_w` is null** — the APU does not expose the relevant rocm-smi key;
   `temp_c` is captured (58–63 °C, no throttling).
6. **This directory remains ROCm 7.2.1 evidence.** ROCm 7.14.0 has a separate,
   scoped 17-cell GGUF matrix; all four `np=16` GGUF cells remain deferred.
   ROCm 7.14 Muse-Glimmer vLLM validation is optional / not prioritized for v0.1
   and pending. See [`rocm-7.14/README.md`](rocm-7.14/README.md).
7. **`llama-bench` cross-check is non-DFlash.** `llama-bench` has no `-md`
   support in this build, so the optional cross-check (`matrix/llama-bench.json`,
   pp512 / tg128) validates only the model-level baseline decode, not spec-decode.
   Result (run 2026-08-13, build `0b1bad1`): 17gb tg128 **10.73 tok/s** vs Study 1
   baseline 10.48 (within 2.4 %); dynamic tg128 **9.28 tok/s** vs 9.14 (within
   1.5 %). The ~2-3 % delta is the expected server/slot-scheduler overhead.
   Prefill (pp512): 17gb **318.96 tok/s**, dynamic **309.03 tok/s**.

---

## 11. Reproducibility

To reproduce a cell:

```bash
cd /path/to/Muse-Glimmer-30B-ROCm
# one cell, e.g. study1 17gb DFlash:
bash scripts/gguf-bench-cell.sh study1 17gb dflash  # emits cell-*.json to matrix/

# a full study (randomized cell order, REPS respected):
bash scripts/run-gguf-matrix.sh study1
bash scripts/run-gguf-matrix.sh study2
bash scripts/run-gguf-matrix.sh study3

# render the matrix markdown from the cell JSONs:
uv run --no-sync python scripts/render_matrix.py
```

The exact serve flags are reproduced in each cell's `manifest.flags` field — the
cell JSON is the authoritative record; the rendered tables in `matrix.md` and
`benchmark.md` derive from it.

---

## 12. Cross-ROCm comparison (7.2.1 vs 7.14.0) — why TPOT, not aggregate tok/s

The Part 2 comparison uses the same recorded llama.cpp commit, flags, model
artifacts, prompt set and seeds; the intended experimental variable is the ROCm
runtime ([rocm-7.14/README.md](rocm-7.14/README.md)). The 7.14 arm is scoped to
17 of 21 planned cells.

**The aggregate-throughput trap.** `aggregate tok/s = Σ generated tokens ÷ max
per-request wall` (§4). The Study 1 records have equal total token counts and
finish-reason distributions across versions. Raw response text and token hashes
were not retained, so the repository does not claim identical token streams for
this cross-ROCm run. In sampled Study 2/3 cells, generated-token counts differ.
That observation is consistent with small numerical differences changing the
sampling path, but the retained summaries do not isolate the mechanism.

Because aggregate tok/s includes generated-token count, a run emitting more
tokens can report a higher value without a comparable change in per-token
latency. The largest example is +40.6% aggregate tok/s for the 17gb `np=4`
baseline, where the 7.14 arm emitted 1.36× as many tokens while TPOT changed
−1.6%.

**The less length-confounded metric.** **TPOT** (`(first→last token wall) ÷
tokens after first`, per-request median) normalizes decode time by generated
tokens. It remains subject to scheduling, workload and measurement variation;
no run-level repeat distribution or profiler counters were retained. DFlash
acceptance rates were similar, with the largest observed difference 1.21
percentage points. Acceptance alone does not prove which runtime component
caused a throughput delta.

| Observed TPOT Δ (7.14 vs 7.2.1) | `np=1` (11 cells) | `np=4` (6 cells) |
|---|---|---|
| Per-cell percentage mean and range | **−0.4%** (−6.4%…+9.0%) | **−1.7%** (−5.4%…+0.2%) |

These observed means do not establish a general speedup or performance
equivalence bound. The aggregate-tok/s column remains in
[matrix-714/comparison.md](matrix-714/comparison.md) for transparency and
compatibility with the original reporting, but TPOT is the primary
cross-version metric for the sampled cells.
