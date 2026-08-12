# llama.cpp DFlash + Full Benchmark Matrix — Design Spec

- **Date:** 2026-08-12
- **Status:** Approved (brainstorming phase) → awaiting implementation plan
- **Owner:** maintainer (AMD ROCm inference)
- **Approach chosen:** **A** — server-per-cell matrix harness on the proven ROCm 7.2.1
  stack, phased (DFlash + load + vision first; ROCm 7.14.0 explored isolated and
  revertible afterwards)
- **Builds on:** `2026-08-11-muse-glimmer-30b-rocm-design.md` (v1, complete). This spec
  is the follow-on that closes v1's three deferred items (DFlash, vision, ROCm 7.14.0)
  and produces the comprehensive llama.cpp benchmark matrix.

---

## 1. Overview & problem statement

v1 shipped two inference paths for Muse-Glimmer-30B on gfx1151 (Strix Halo), validated
on hardware, and benchmarked head-to-head — but deliberately deferred three things:
**DFlash speculative decoding** (vLLM path hits a registry bug), the **llama.cpp vision
path**, and **ROCm 7.14.0** (gfx1151's "official" ROCm). v1 also benchmarked only a
single weight (kquant-17gb) at a single sampling config.

The follow-on work:

1. **Enable and measure DFlash** — natively supported in llama.cpp (unlike vLLM), so
   llama.cpp is the correct home for it.
2. **Validate the llama.cpp vision path** (`--mmproj`) and add it to the matrix.
3. **Explore ROCm 7.14.0** in isolation, without endangering the proven 7.2.1 stack.
4. **Produce the comprehensive llama.cpp benchmark matrix** the maintainer's org needs:
   multiple weight sizes × concurrency {1, 4, 16} × {baseline, +DFlash}, with full
   speed and memory data — **apples-to-apples with Meta's published numbers and
   reproducible by any third party.**

The dominant deliverable is #4. #1 and #3 feed it; #2 (vision) is a parallel axis.
Scientific rigor and public defensibility are first-class requirements: every number
must have a defensible provenance, and the methodology must match Meta's so our gfx1151
row sits honestly next to Meta's published RTX 5090 / Apple Silicon rows.

## 2. Goals & success criteria

**A reader can:**

1. Reproduce **Study 1** — the Meta-comparable DFlash anchor (greedy, batch 1, diverse
   prompt set) — and obtain a gfx1151 speedup that is methodologically comparable to
   Meta's published 3.1× (RTX 5090, llama.cpp) and 1.8× (M5 Max).
2. Read the **full matrix** (Study 2): both official kquants × {c1, c4, c16} × {baseline,
   +DFlash}, each with aggregate tok/s, TTFT (p50/p90), TPOT, peak memory, and — for
   DFlash cells — draft acceptance and finish_reason distribution.
3. See the **vision axis** (Study 3): image-input throughput + the memory delta vs
   text-only, cross-checked against Meta's published ~+2 GB.
4. Read a **7.2.1-vs-7.14.0** comparison (or an honest "not feasible on this host"
   finding) that never put the 7.2.1 deliverable at risk.
5. Trust every number because `docs/results/METHODOLOGY.md` documents the exact hardware,
   software, flags, prompt set, definitions, and statistical treatment.

**Non-goals:** re-running the vLLM path (the org asked to focus on llama.cpp);
re-measuring quantization quality (Meta already published degradation: dynamic 0.2%,
17gb 1.0%); training/fine-tuning; vLLM-side DFlash (registry-blocked, stays documented);
custom/non-Meta quants.

## 3. Scope & decisions (locked during brainstorming)

| Decision | Choice |
|---|---|
| Sequencing vs risk | **Deliver the matrix on proven ROCm 7.2.1 first; explore 7.14.0 isolated + revertible afterwards** |
| Weight matrix breadth | **2 official Meta kquants only** (kquant-17gb, kquant-dynamic) — apples-to-apples with Meta |
| Vision depth | **Add vision to the matrix** (Study 3) |
| Memory metric | **Process RSS (VmHWM) + llama.cpp startup mem log**; rocm-smi carve-out documented as misleading and not used as the headline |
| Speed metrics | **tok/s + TTFT + TPOT breakdown** (per-request, p50/p90) |
| Harness approach | **A — server-per-cell parameterized harness**, `llama-bench` only as a non-DFlash model-level cross-check (it has no `-md` support) |

## 4. Authoritative inputs (verified from Meta's official docs)

Sourced from `meta-models/Muse-Glimmer-30B-GGUF` README and `meta-models/Muse-Glimmer-30B`
model card (fetched 2026-08-12 via the project's hf-mirror).

### 4.1 Weight inventory (the complete GGUF repo)

| File | Size | Role |
|---|---|---|
| `muse-glimmer-30B-kquant-17gb.gguf` | 16.76 GiB | Text, default, fits 24 GB |
| `muse-glimmer-30B-kquant-dynamic.gguf` | 19.65 GiB | Text, higher quality, 32 GB |
| `dflash-kquant.gguf` | 1.63 GiB | DFlash speculative drafter |
| `mmproj-kquant.gguf` | 1.40 GiB | Perception encoder (vision) |

No BF16/Q8/Q2 GGUF is published ("use the base repo for full precision"). Both text
builds are text-only on their own; the two companions are additive.

### 4.2 Meta's published DFlash numbers (the comparability anchor)

> Footnote, verbatim: *"Average across a diverse prompt set. Measurements done with
> **batch size 1 and greedy decoding**. M4/M5 measurements were done using ExecuTorch,
> and RTX using llama.cpp."*

| GPU | Baseline no-spec (tok/s) | Avg with DFlash (tok/s) | Speedup |
|---|---|---|---|
| **Nvidia RTX 5090** | 74.9 | 233.4 | **3.1×** *(llama.cpp)* |
| Apple M4 Max | 23.7 | 37.8 | 1.5× *(ExecuTorch)* |
| Apple M5 Max | 26.6 | 50.2 | 1.8× *(ExecuTorch)* |

All on **K-Quant-17GB + quantized DFlash drafter**, greedy, batch 1. The RTX 5090 row is
**itself a llama.cpp measurement** — so our gfx1151 llama.cpp number is directly
comparable **if we match the methodology**. This drives Study 1's design.

### 4.3 DFlash drafter architecture (for interpreting results)

| Component | Setting |
|---|---|
| Draft layers | 5 |
| Block size | 16 tokens per forward pass |
| Attention | Sliding-window, 2048, all layers |
| Attention heads | 32 query / 8 KV (GQA) |
| Sequence length | 131,072 |
| Hidden-feature layers | 5, uniform over target {1, 13, 25, 37, 49} of 52 |

### 4.4 Quantization degradation (quality is already answered by Meta)

| | Full Precision | K-Quant-Dynamic | K-Quant-17GB |
|---|---|---|---|
| % Degradation* | — | **0.2%** | **1.0%** |
| Target hardware | 64 GB | 32 GB | 24 GB |

\* Average accuracy across 15 common benchmarks. → The 17gb-vs-dynamic question is
**speed/memory only** on our side; quality is cited from Meta, not re-measured.

### 4.5 Meta's memory envelope (cross-check target)

| Build | text only | + vision | + vision + drafter |
|---|---|---|---|
| `17gb` | ~17 GB | ~19 GB | ~20 GB |
| `dynamic` | ~20 GB | ~22 GB | ~23 GB |

### 4.6 Generation model (the rigor-critical facts)

- **Best-practice sampling:** temperature **1.0**, top_p **0.95**, top_k **64**.
- **`reasoning_strength`** controls thinking length: `low`/`medium`/`high`/`xhigh`,
  **default `high`**. Thinking **cannot be switched off** (`--reasoning off` is a no-op).
- **Silent context-truncation trap:** `llama-server` divides `-c` across `-np` slots, so
  one request gets `-c / -np`. If a generation exceeds its slot context it produces **no
  error and no answer** — *"that silently reads as a wrong answer rather than a failure."*
  → per-slot context must comfortably exceed max output; `finish_reason` must be reported.
- Muse-Glimmer reasons at length; TTFT/TPOT measure raw decode, not per-turn answer latency.
- Spec-decode at greedy is **exact** (byte-identical output) → equivalence is verifiable.

## 5. The matrix (revised Section 1) — two labeled studies + vision

Each cell is an independent `llama-server` process with its exact flags; llama.cpp
startup is ~1 s, so per-cell restarts are cheap.

### Study 1 — Meta-comparable DFlash anchor (greedy, batch 1)

Reproduces Meta's methodology so the gfx1151 row sits next to their RTX 5090 3.1× / M5
Max 1.8× rows. **The headline comparable number.**

```
weight   ∈ {kquant-17gb, kquant-dynamic}     # Meta published only 17gb; dynamic is novel
DFlash   ∈ {off, on}                          # => 4 cells
greedy (temp 0), batch size 1 (c = 1)
diverse prompt set (6 prompts), averaged      # matches Meta's "diverse prompt set"
```

Per cell: avg tok/s (over the prompt set), **speedup ratio**, **draft acceptance rate**,
**byte-equivalence vs baseline** (same prompt, greedy → identical token sequence), peak
memory (VmHWM + llama.cpp mem log), finish_reason.

Serve flags (single source of truth): `-m <weight> [-md models/dflash-kquant.gguf -ngld
99] -ngl 999 -np 1 -c 8192 --jinja --temp 0 --seed 0`.

Client params: endpoint `/v1/chat/completions`, `max_tokens` **256** (a pure-decode
budget — tok/s is valid regardless of whether reasoning finished), fixed seed.
Byte-equivalence = the DFlash-on output is token-identical to the DFlash-off output for
the same prompt (greedy → exact; a mismatch is a finding).

### Study 2 — Throughput under load (realistic serving)

```
weight   ∈ {kquant-17gb, kquant-dynamic}
conc     ∈ {1, 4, 16}
DFlash   ∈ {off, on}                          # => 2 × 3 × 2 = 12 cells
temp 1.0, top_p 0.95, top_k 64 (Meta serving config), --seed pinned (reproducible)
reasoning_strength = high (the default; pinned + reported), max_tokens 512
per-slot context 8192  =>  -c = np × 8192     # safely above max output (no truncation trap)
```

Per cell: aggregate tok/s, TTFT (p50/p90), TPOT (median), peak memory, finish_reason
distribution; DFlash cells also report draft acceptance.

**Labeled, in the docs, as NOT directly comparable to Meta** (Meta has no batch>1 DFlash
data) — this is original throughput-under-load research.

### Study 3 — Vision axis (lighter)

```
{17gb, dynamic} × {vision} × {c=1, c=4}        # 4 baseline cells
+ {17gb} × {vision + DFlash} × {c=1}           # 1 cell (does DFlash help vision decode?)
```

Per cell: same metrics + **memory delta vs text-only** (cross-check Meta's ~+2 GB vision,
~+3.6 GB with drafter). No c=16 vision (heavy, not meaningful). Images via
`/v1/chat/completions` with `--mmproj`; a fixed test image is committed for reproducibility.

### Optional: `llama-bench` cross-check

One `llama-bench` run per weight (no HTTP, pp512/tg128, JSON output) to cross-check the
c=1 server tok/s and to document prefill (pp) vs decode (tg) separately. **Does not apply
to DFlash** (llama-bench has no `-md` support in this build).

## 6. Measurement methodology (Section 2)

### 6.1 Metric definitions

| Metric | Definition | Capture | Pitfall defused |
|---|---|---|---|
| decode tok/s | total tokens generated ÷ generation-phase wall (after first token); all tokens, no reasoning/content split | count from stream; cross-check server `usage`; document any diff | "does your count include reasoning?" → yes, by definition; split reported |
| aggregate tok/s (load) | Σ generated tokens ÷ max per-request wall | per-request stream counts, summed | matches existing harness; comparable to prior runs |
| TTFT | request send → first streamed chunk; report **p50/p90** | `perf_counter()` around SSE stream; loopback, HTTP overhead negligible | "mean hides queueing tail" → distribution reported |
| TPOT | (first→last token wall) ÷ (tokens after first); per-request median | inter-chunk timestamps from stream | the core DFlash metric (it lowers per-token decode cost) |
| peak memory | process **VmHWM** + llama.cpp startup-log component breakdown (weights / KV / compute) | poll `/proc/<pid>/status`; parse server stderr | unified memory: GPU buffers ARE RAM, so RSS is the real footprint; component log splits it authoritatively. Cross-check vs Meta's ~17/20 GB |
| DFlash acceptance | avg accepted draft tokens per verify step + acceptance rate | llama.cpp spec-decode stats (`-v` / metrics) | **without this the speedup is uninterpretable**; "2.1× at 78% acceptance" is the defensible claim |
| byte-equivalence (Study 1 only) | DFlash output token-identical to baseline for the same prompt | diff fixed-seed greedy outputs across modes | verifies Meta's "identical output quality" on our setup; greedy spec-decode is exact, so a mismatch is a finding |
| finish_reason distribution | count of `stop` vs `length` across N runs | server response field | Meta's truncation trap: high `length` ratio ⇒ wrong output/context budget |
| power/temp | GPU power (W) + temp (°C) during run | `rocm-smi --showpower --showtemp` (these ARE valid on Strix Halo, unlike VRAM) | "did the APU throttle?" → show it didn't (or admit it did) |

### 6.2 Statistical protocol

- **Greedy anchor (Study 1):** 3 reps (output deterministic; timing noise only), report
  **median + min/max**.
- **Load test (Study 2):** **5 reps** (output varies under sampling → more variance),
  report **median + IQR**.
- **2 warmup requests** discarded before each measured cell (thermal + JIT + cache).
- **Cell order randomized** and the order recorded, to kill systematic thermal drift
  across a long matrix run.
- Raw distributions reported, not just point estimates.

### 6.3 Fairness controls (apples-to-apples within the matrix)

Same hardware, same ROCm, same llama.cpp commit/build, same warmup, same thread count
(pinned to physical cores, reported), same prompt set, no other GPU load (documented).
Only the cell config varies (weight file, ±`-md`, ±`--mmproj`, `-np`). The 7.2.1-vs-7.14.0
comparison reuses identical source/flags/weights/workload — only ROCm differs.

### 6.4 Honest caveats (written into docs, not hidden)

1. gfx1151 unified ~215 GB/s vs RTX 5090 GDDR7 — absolute tok/s will differ; only the
   **speedup ratio + methodology** are directly comparable to Meta.
2. If llama.cpp does not emit draft-acceptance stats, we instrument it (scoped risk,
   flagged in P0; Study 1 is blocked until this is resolved).
3. Reasoning-model tok/s measures raw decode, not end-to-end per-turn answer latency.
4. rocm-smi VRAM is not the footprint on Strix Halo (documents only the ~32 GB carve-out).

## 7. Components & interfaces (Section 3)

| File | Responsibility | Interface |
|---|---|---|
| `scripts/gguf-bench-cell.sh` | Run **one** cell: launch `llama-server` with exact flags, health-check, warmup, drive client, capture mem/power/temp, tear down. Emits one JSON record. Fail-fast. | args: weight, ±DFlash, ±vision, np, per_slot_ctx, study params |
| `scripts/bench_client.py` *(upgrade)* | Streaming client: per-request TTFT/TPOT, aggregate tok/s, finish_reason, prompt-set loop, both endpoints, p50/p90 | args: base, np, max_tokens, temp, top_p, top_k, seed, reasoning_strength, prompt-set-id, endpoint |
| `scripts/run-gguf-matrix.sh` | Driver: enumerate Study 1/2/3 cells, randomize order, call cell script, collect JSON, render markdown | writes `docs/results/matrix/` |
| `scripts/prompt-sets/muse-glimmer-diverse.json` | Fixed, published 6-prompt set + version id | reproducibility artifact |
| `scripts/prompt-sets/test-image.png` | Fixed test image for Study 3 | reproducibility artifact |
| `configs/gguf-bench/*.conf` | Per-study parameter sets as single source of truth (like `serve-args.conf`) | lintable, CI-safe |
| `tests/test_gguf_bench.py` | Config validation: per-study flag logic correct (greedy→temp 0, DFlash→`-md`, load→temp 1.0), prompt set loads, JSON schema | CI-safe only |

The existing `scripts/benchmark.sh` / `gguf-quickstart.sh` are reused unchanged for the
vLLM/quick-start paths; the new harness is additive.

## 8. Phasing & validation gates

- **P0 — Harness.** Build the 6 components + tests. Validate end-to-end on one cell
  (17gb / greedy / baseline / c=1). **Gate:** that cell emits well-formed JSON + sane
  tok/s + memory/power/acceptance captured.
- **P-A — Study 1 (Meta-comparable anchor), 7.2.1.** 4 cells + byte-equivalence +
  acceptance. **Gate:** gfx1151 row sits next to Meta's table with a clear methodology
  note; equivalence passes.
- **P-B — Study 2 (load), 7.2.1.** 12 cells. **Gate:** 12 clean records, finish_reason
  predominantly `stop`.
- **P-C — Study 3 (vision), 7.2.1.** ~5 cells. **Gate:** vision loads, answers, memory
  delta matches Meta's ~+2 GB.
- **P-D — 7.14.0 isolated.** **Gate:** 7.2.1 data committed/immutable in git; 7.14.0 is
  installable + revertible, OR honestly documented as infeasible.
- **P-E — Docs.** Render tables + analysis; update `benchmark.md`, `adaptation.md`,
  `troubleshooting.md`, `handoff.md`; write `METHODOLOGY.md`.

## 9. ROCm 7.14.0 isolation (P-D) — designed for safety + rigor

The 7.2.1 deliverable is **committed and locked in git before P-D begins**, so any
upgrade fallout cannot touch the core data. Then:

1. **Feasibility first.** ROCm is system-level (`/opt/rocm`); a worktree cannot isolate
   it. P-D step 1 is an honest check: *can 7.14.0 actually be installed/obtained on this
   host without removing 7.2.1?* If no → document as infeasible and stop (7.2.1 data safe).
2. **If feasible: side-by-side install.** Install 7.14.0 to `/opt/rocm-7.14.0` (or
   equivalent), select via env (`LD_LIBRARY_PATH`/`RPATH`), **7.2.1 stays default**.
   Rebuild llama.cpp against 7.14.0 in a separate build dir (same source/flags). Fully
   revertible by pointing back at 7.2.1.
3. **Re-run the identical matrix** (same source, flags, weights, workload, prompt set,
   seed) — only ROCm differs. Cell-vs-cell comparison: 7.2.1 vs 7.14.0 tok/s / TTFT / RSS.

## 10. Documentation & reproducibility (the public "science" layer)

- **`docs/results/METHODOLOGY.md`** — the document critics read first: hardware + software
  versions, exact serve flags, prompt set, every metric's definition (§6 table),
  statistical treatment (reps/median/warmup/randomization), known limitations. Every
  number points here.
- **`docs/results/benchmark.md`** — new sections: Study 1 (gfx1151 row next to Meta's
  RTX 5090 3.1× / M5 Max 1.8×), Study 2, Study 3, 7.2.1-vs-7.14.0 — each with its
  methodology box + caveats.
- **`docs/results/matrix/`** — raw per-cell JSON + `matrix.json` + rendered markdown
  (published artifacts, re-analyzable by anyone).
- **`docs/adaptation.md`** — DFlash row updated from "deferred" to "validated on
  llama.cpp" with measured speedup + acceptance.
- **`docs/troubleshooting.md`** — add: harmless DFlash draft-mem warning, silent
  context-truncation trap, `reasoning_strength`, byte-equivalence finding.
- **`handoff.md`** — status + future-work closed (DFlash, vision) + 7.14.0 status.

## 11. Error handling & risks

- Server fails to start (the DFlash `[spec] failed to measure draft model memory` warning
  is harmless; a real crash isn't) → cell marked ERROR, logged, never silently dropped.
- OOM at np=16 + DFlash + dynamic → detect, report, skip cell with an explanation.
- Request timeout/hang → client timeout, cell marked failed.
- Each cell is an independent process, so one failure cannot cascade.
- **Top risk:** draft-acceptance not exposed by llama.cpp → P0 resolves this (instrument
  if needed); Study 1 is blocked on it.
- **7.14.0 risk:** host destabilization → mitigated by immutable 7.2.1 data + side-by-side
  install + feasibility-first.

## 12. Testing & CI strategy

- `tests/test_gguf_bench.py` (CI-safe, no GPU): per-study flag-logic validation, prompt-set
  validity, cell-output JSON schema, markdown-render sanity.
- Existing layered tests (gpu/server markers) unchanged; new server tests added under the
  same `server` marker (auto-skip when no server listening).
- CI (`.github/workflows/ci.yml`) runs the no-GPU subset + shellcheck. Hardware validation
  is the local `verify` target + the published `docs/results/matrix/` artifacts.

## 13. Open items to resolve during implementation

1. **Confirm llama.cpp emits draft-acceptance stats** (via `-v` or a metrics endpoint);
   if not, decide instrumentation scope. (Blocks Study 1.)
2. **Confirm `usage.completion_tokens`** includes reasoning tokens on `/v1/chat/completions`
   with `--jinja`; define tok/s from the stream regardless and cross-check.
3. **Finalize the 6-prompt diverse set** (code / math / factual QA / creative / reasoning /
   instruction) with a version id in `muse-glimmer-diverse.json`. (max_tokens already
   pinned: 256 for Study 1, 512 for Study 2.)
4. **Confirm `--seed` determinism** at greedy for the byte-equivalence test.
5. **Capture exact llama.cpp startup-log memory lines** to parse the component breakdown
   programmatically.
6. **7.14.0 feasibility probe** (P-D): is 7.14.0 obtainable/installable side-by-side on
   this host?

## 14. References

- GGUF repo (authoritative): https://huggingface.co/meta-models/Muse-Glimmer-30B-GGUF
- Base model card: https://huggingface.co/meta-models/Muse-Glimmer-30B
- DFlash paper: https://arxiv.org/abs/2602.06036
- llama.cpp Muse-Glimmer merge: https://github.com/ggml-org/llama.cpp/pull/26841 (b10353+)
- v1 spec: `docs/superpowers/specs/2026-08-11-muse-glimmer-30b-rocm-design.md`
- v1 handoff: `handoff.md`
- Existing harness: `scripts/benchmark.sh`, `scripts/bench_client.py`, `scripts/gguf-quickstart.sh`
