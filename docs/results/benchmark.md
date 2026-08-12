# Benchmark — Muse-Glimmer-30B on gfx1151 (Strix Halo)

> Reproduce: start a server (`scripts/03-serve-vllm.sh` or
> `scripts/gguf-quickstart.sh`), then `BASE=http://127.0.0.1:<port> bash
> scripts/benchmark.sh`. Raw JSON lands in `docs/results/*.json` (gitignored
> runtime artifacts); the numbers below are the validated reference runs. Same
> harness, same prompt, same 512 output tokens, same concurrencies for both
> engines — an apples-to-apples GPU-vs-GPU comparison.

## Head-to-head: vLLM (BF16) vs llama.cpp (Q4 K-quant)

| Concurrency | vLLM — BF16 (tok/s) | llama.cpp — Q4 kquant (tok/s) | Q4 speedup |
|---|---|---|---|
| 1 | 4.2 | 10.5 | 2.5× |
| 4 | 14.0 | 21.3 | 1.5× |
| 16 | 40.8 | 102.0 | 2.5× |

Both engines run on the **GPU (HIP/gfx1151)**. **llama.cpp Q4 is faster at every
concurrency** because 4-bit weights move ~4× less memory per decoded token, and
single-stream / batched decode on this APU is **memory-bandwidth-bound** (the
~215 GB/s unified fabric is the ceiling for both). vLLM pays BF16's 2-byte/param
traffic; llama.cpp pays ~0.5-byte/param.

**Reading the numbers**

- **Single-stream chat speed**: vLLM ~4.2 tok/s, llama.cpp ~10.5 tok/s — both
  fluid for interactive use; llama.cpp is snappier thanks to the quant.
- **Concurrency scaling**: both scale with batching. vLLM's continuous batching
  scales **automatically** (no tuning). llama.cpp scales to c=16 **only when you
  raise the slot count** (`-np 16`): the default `gguf-quickstart.sh` ships with
  **4 slots and plateaus at ~22 tok/s** (c=4 ≈ c=16); with `-np 16 -c 16384` it
  reaches the 102 tok/s above. This tuning difference is itself part of the
  contrast — vLLM's scheduler is the better out-of-box concurrent server.

### Inference quality (both correct)

Muse-Glimmer is a reasoning model (chain-of-thought in a `reasoning` channel,
then the answer in `content`). Both engines surface this and answer correctly:

| Prompt | vLLM (BF16) | llama.cpp (Q4 kquant) |
|---|---|---|
| "Say hello in one word." | reasons → `Hello` (`tests/test_smoke.py`) | reasons → `Hello` |
| "What is 17 × 24? Just the number." | (BF16 reference) | `408` ✓ |
| "Use the get_weather tool for Tokyo." | ATEM tool-call parses → `tool_calls` (`tests/test_parsers.py`) | n/a (no native tool parser) |

The Q4 kquant stays coherent and numerically correct on these checks — Meta's
calibrated quant is high quality (the model README reports "minimum to no
degradation on agentic tasks" for the K-quant).

### Feature matrix

| | vLLM (BF16) | llama.cpp (Q4 kquant) |
|---|---|---|
| Precision | BF16 (full) | ~4-bit K-quant |
| Weights on disk | 55.5 GiB | 15.6 GiB |
| Attention | `TRITON_ATTN` | llama.cpp HIP kernels |
| muse_glimmer reasoning parser | ✅ native (`message.reasoning`) | ⚠️ via chat template (`--jinja`); lands in message body |
| ATEM tool-call parser | ✅ native (`message.tool_calls`) | ❌ (raw text only) |
| Vision / multimodal | ✅ native (ViT-G/14) | ⚠️ separate `mmproj-kquant.gguf`, `--mmproj` |
| Context | 128K | per-slot (e.g. 1024 at `-np 16`); raise `-c` to grow |
| Concurrent batching | automatic (continuous) | manual via `-np <slots>` |
| Install | source-build vLLM (~1 h HIP compile) | cmake build llama.cpp (~10 min) |
| Startup | ~8 min (loads 56 GiB + captures graphs) | ~1 s (mmap, lazy GPU fill) |

### When to use which

- **vLLM** — the full-feature path: native reasoning + tool-call parsing, vision,
  128K context, automatic concurrency. Use when you need the agentic/multimodal
  features or a production-grade concurrent server, and can afford the BF16
  precision + the source build.
- **llama.cpp** — the fast, light path: ~3.5× smaller weights, ~2.5× faster
  decode, ~1 s startup, trivial install. Use for interactive text chat, when the
  quant's quality is acceptable, or to be up-and-running without compiling vLLM.

---

## Environment manifest

| Item | Value |
|---|---|
| GPU | AMD Radeon 8060S, **gfx1151** (RDNA 3.5), 40 CUs |
| Memory | 94 GiB unified LPDDR5X (~215 GB/s); vLLM sees an 80 GiB pool |
| ROCm | 7.2.1 |
| Kernel | 6.17.0-1020-oem |
| PyTorch | 2.10.0+rocm7.13.0a20260513 (TheRock gfx1151) |
| vLLM | 0.1.dev1+g606a12cd7 (source-built, PR #51655) |
| llama.cpp | v1 (0b1bad1), HIP build, `-DAMDGPU_TARGETS=gfx1151` |
| GGUF | meta-models/Muse-Glimmer-30B-GGUF `muse-glimmer-30B-kquant-17gb.gguf` (15.6 GiB) |

**vLLM memory at startup** (vLLM's own accounting): 56.49 GiB weights + 1.89 GiB
peak activation + 0.72 GiB CUDA graphs, with **13.62 GiB KV cache** in the 80 GiB
unified pool.

## Caveats

- **Reasoning model**: `/v1/completions` throughput measures raw decode speed; a
  real chat turn emits chain-of-thought first, so end-to-end per-turn latency is
  higher than `512 tok / tok_s`.
- **`rocm-smi` VRAM is not the real footprint on Strix Halo**: it reports only
  the ~32 GiB dedicated carve-out (and ~1 GiB "used" — misleading). On the APU's
  unified memory, llama.cpp's GPU buffers don't increment the carve-out counter;
  trust vLLM's startup accounting above. (We confirmed llama.cpp is GPU-bound, not
  CPU: aggregate CPU during generation was ~8% across 32 cores.)
- **llama.cpp slot tuning**: the default 4 slots cap concurrent throughput at
  ~22 tok/s; pass `-np <slots>` (and enough `-c` per slot) to scale.

---

# Part 2 — DFlash (speculative decoding) + full benchmark matrix

> Methodology, raw per-cell JSON, prompt set, and metric definitions:
> [`METHODOLOGY.md`](METHODOLOGY.md). Rendered full matrix:
> [`matrix/matrix.md`](matrix/matrix.md). Spec:
> `docs/superpowers/specs/2026-08-12-llamacpp-dflash-benchmark-design.md`.
> All numbers below are pulled verbatim from the cell JSONs.

This part adds the v1-deferred items — **DFlash speculative decoding** and
**vision via `--mmproj`** — measured on the same gfx1151 APU, same llama.cpp
build (`0b1bad1`), same ROCm 7.2.1. Each cell is an independent `llama-server`
process with its exact flags recorded in the cell JSON.

> **DFlash enablement gotcha — read before any DFlash run.** `llama-server`'s
> `--spec-type` **defaults to `none`**, so `-md dflash.gguf -ngld 99` *alone*
> loads the draft model but **never drafts** (silent 1.0× no-op). Always pass
> `--spec-type draft-dflash --spec-draft-n-max 16`. `n_max=16` is the measured
> sweet spot (it equals the DFlash block_size); n_max `3 / 8 / 16 / 32` →
> 1.14× / 1.51× / 1.60× / 1.60×. See
> [METHODOLOGY.md §8](METHODOLOGY.md#8-dflash-enablement-the-silent-no-op-gotcha)
> and [troubleshooting.md#dflash-silent-noop](../troubleshooting.md#dflash-silent-noop).

## ⚠️ c=16 + DFlash: do not use

> **Do NOT combine DFlash with `-np 16` (high concurrency). It is pathologically
> slow — >1000× slower per-request than the c=16 baseline. c=16 itself is fine;
> the pathology is DFlash-specific.**

**Evidence (from the cell JSONs):**

- **17gb c=16 DFlash** (`matrix/cell-study2-17gb-np16-df1-vis0.json`): a
  16×48-token probe batch completed **0 of 768 tokens in 27.7 s**, while the
  baseline c=16 run on the same weight delivered **34.47 tok/s** aggregate —
  effectively >1000× slower per-request. Probe-time draft acceptance was 0.182
  with per-slot acceptance 0.065–0.091.
- **dynamic c=16 DFlash** (`matrix/cell-study2-dynamic-np16-df1-vis0.json`): the
  full REPS=5 cell was **aborted after 5 h 16 m** without completing. At abort,
  the drafter had emitted **3,270,000 draft tokens** of which **6,060 were
  accepted** — acceptance rate **0.0018 (0.18 %)**, ~1/37× the baseline
  aggregate rate.

**Verified root cause:** at `-np 16` the drafter fires for all 16 slots
simultaneously, generating an enormous draft volume (millions of tokens) that is
almost entirely rejected (>99.8 % in the dynamic run), while the full
generate+verify compute for all those rejected drafts is paid in full. The draft
model's predictions diverge badly from the target under batched concurrent load,
so spec-decode goes into reverse — it costs more than it saves.

**c=16 itself is fine** (baseline 17gb 34.5 tok/s, dynamic 31.0 tok/s — see
Study 2 below). The pathology is DFlash-specific. Both c=16 DFlash cells are
recorded as evidence-based non-completions (`pathological: true`); they are not
missing data, they are *findings*.

**Best-practice table (user-facing):**

| Use case | Config | Expected |
|---|---|---|
| Single-stream interactive chat (c=1) | DFlash ON, `--spec-draft-n-max 16` | **~2.2× faster**, identical output (greedy), ~+3 GiB VmPeak |
| Light concurrent (c ≤ 4) | DFlash ON | ~1.3–1.75×, mind the +3 GiB drafter footprint |
| **High throughput (c ≥ 8, esp c=16)** | **DFlash OFF (baseline)** | c16 ~31–34 tok/s; **DFlash here is pathological** |

> Always: when DFlash is on, pass `--spec-type draft-dflash --spec-draft-n-max 16`
> (else it silently does nothing). See also
> [README.md — Best practices / Pitfalls](../../README.md#best-practices-pitfalls-read-this-before-using-dflash-or-c16).

---

## Study 1 — DFlash anchor (greedy, batch 1) — Meta-comparable

6-prompt diverse set, `temp=0 seed=0`, `-np 1 -c 8192`, `max_tokens=256`,
**3 reps** (median + min/max). Headline result placing gfx1151 against Meta's
published DFlash numbers.

| weight | mode | agg tok/s | TTFT p50 (s) | TPOT med (s) | footprint VmPeak (GiB) | Speedup | draft acceptance |
|---|---|---|---|---|---|---|---|
| 17gb | baseline | 10.48 | 0.460 | 0.0942 | 23.91 | 1.00× | — |
| 17gb | DFlash | 23.03 | 0.475 | 0.0482 | 26.73 | **2.20×** | 0.233 (3932/16896) |
| dynamic | baseline | 9.14 | 0.486 | 0.1082 | 26.57 | 1.00× | — |
| dynamic | DFlash | 21.82 | 0.507 | 0.0486 | 29.28 | **2.39×** | 0.237 (3955/16656) |

- Per-rep variance is tight (17gb DFlash 22.91–23.40; dynamic DFlash 21.66–22.25)
  — the speedup is not a one-rep artifact.
- All reps finish on `length` (max_tokens=256 cap; `reasoning_strength=high`
  default). This is expected and does not affect the tok/s computation.
- VmPeak: the drafter adds ~+2.5–3 GiB. DFlash dynamic is ~+3 GiB vs 17gb.
- **Byte-equivalence PASS** — both baseline and DFlash emit `'391'` for
  `17 × 23` (`scripts/check_dflash_equiv.sh`). Greedy spec-decode is exact.

### gfx1151 next to Meta's published DFlash rows

Meta's footnote: *"Average across a diverse prompt set. Measurements done with
**batch size 1 and greedy decoding**. M4/M5 measurements were done using
ExecuTorch, and RTX using llama.cpp."* All on K-Quant-17GB + quantized drafter.

| GPU | Baseline (tok/s) | DFlash (tok/s) | Speedup | Engine |
|---|---|---|---|---|
| Nvidia RTX 5090 | 74.9 | 233.4 | **3.1×** | llama.cpp |
| Apple M4 Max | 23.7 | 37.8 | 1.5× | ExecuTorch |
| Apple M5 Max | 26.6 | 50.2 | 1.8× | ExecuTorch |
| **gfx1151 (this repo, 17gb)** | **10.48** | **23.03** | **2.20×** | **llama.cpp** |
| **gfx1151 (this repo, dynamic)** | **9.14** | **21.82** | **2.39×** | **llama.cpp** |

Our 17gb row matches Meta's RTX 5090 methodology exactly (greedy, batch 1, same
K-quant + quantized drafter, llama.cpp). gfx1151 lands **between the M5 Max
(1.8×) and the RTX 5090 (3.1×)** — credible for a 50 TOPS NPU/iGPU-class part.
The dynamic row is **novel** (Meta did not publish a dynamic DFlash number); its
higher speedup reflects DFlash recovering more of the slower dynamic baseline.

> **Comparability caveat.** gfx1151's ~215 GB/s unified fabric is much narrower
> than the RTX 5090's GDDR7, so absolute tok/s differs (10.5 vs 74.9 baseline).
> The **speedup ratio** + methodology are the directly comparable quantities.

## Study 2 — Throughput under load (temp 1.0) — NOT Meta-comparable

`temp=1.0 top_p=0.95 top_k=64 seed=42`, `reasoning_strength=high`, `max_tokens=512`,
per-slot context 8192 (no truncation), **5 reps** (median + min/max). c=16
baselines ran 1–5 reps due to wall-time budget. **This is original
throughput-under-load research** — Meta has no batch>1 DFlash data, so do not
compare the speedup ratios here to Meta's table.

| weight | np | mode | agg tok/s | TTFT p90 (s) | TPOT med (s) | VmPeak (GiB) | acceptance |
|---|---|---|---|---|---|---|---|
| 17gb | 1 | baseline | 10.52 | 0.555 | 0.0942 | 24.41 | — |
| 17gb | 1 | DFlash | 22.26 | 0.567 | 0.0435 | 27.27 | 0.207 |
| 17gb | 4 | baseline | 15.60 | 1.144 | 0.1778 | 27.24 | — |
| 17gb | 4 | DFlash | 27.30 | 1.247 | 0.1066 | 33.44 | 0.184 |
| 17gb | 16 | baseline | 34.47 | 3.225 | 0.1706 | 30.27 | — |
| 17gb | 16 | DFlash | ⚠ **PATHOLOGICAL — did not complete** (see [warning](#c16-dflash-do-not-use)) | — | — | — | — |
| dynamic | 1 | baseline | 9.09 | 0.604 | 0.1091 | 27.16 | — |
| dynamic | 1 | DFlash | 19.89 | 0.608 | 0.0521 | 30.04 | 0.194 |
| dynamic | 4 | baseline | 20.90 | 1.274 | 0.1809 | 29.18 | — |
| dynamic | 4 | DFlash | 28.22 | 1.343 | 0.1212 | 34.88 | 0.192 |
| dynamic | 16 | baseline | 31.05 | 3.366 | 0.2366 | 38.41 | — |
| dynamic | 16 | DFlash | ⚠ **PATHOLOGICAL — did not complete** (see [warning](#c16-dflash-do-not-use)) | — | — | — | — |

**Reading the table:**

- **DFlash speedup shrinks as concurrency rises.** At c=1 it is still ~2.1–2.4×
  (matching Study 1). At c=4 it falls to ~1.75× (17gb) and ~1.35× (dynamic) on
  aggregate throughput. At c=16, DFlash is **pathological** — see the warning
  block above and [`METHODOLOGY.md §6`](METHODOLOGY.md#6-the-c16-dflash-pathology).
- **c=16 baselines are the throughput winners** (17gb 34.5, dynamic 31.0 tok/s)
  — and you keep them by leaving DFlash **off**.
- **Aggregate vs per-request.** Aggregate tok/s (`Σ tokens ÷ max per-request
  wall`) is the right column for "how many tokens/s can this server push"; the
  per-request TPOT rises with concurrency (queueing). TTFT p90 rises too.

## Study 3 — Vision axis (temp 1.0, `--mmproj`, test image)

`--mmproj models/mmproj-kquant.gguf` + the fixed `scripts/prompt-sets/test-image.png`,
same sampling as Study 2, **3 reps**. Confirms the multimodal path loads +
answers, and measures its memory delta vs text-only.

| weight | np | mode | agg tok/s | TTFT p90 (s) | TPOT med (s) | VRAM (MiB) | VmPeak (GiB) | acceptance |
|---|---|---|---|---|---|---|---|---|
| 17gb | 1 | baseline | 10.50 | 0.762 | 0.0940 | 1093 | 26.24 | — |
| 17gb | 1 | DFlash | 20.46 | 0.830 | 0.0471 | 1094 | 28.99 | 0.188 |
| 17gb | 4 | baseline | 21.12 | 2.038 | 0.1768 | 1093 | 29.19 | — |
| dynamic | 1 | baseline | 9.08 | 0.823 | 0.1086 | 1093 | 29.03 | — |
| dynamic | 4 | baseline | 20.00 | 2.145 | 0.1758 | 1093 | 31.89 | — |

**Reading the table:**

- **Vision loads + answers via `--mmproj`.** All five cells produce coherent
  image-grounded answers (the same generate path as text, with the projected
  image patch tokens prepended).
- **mmproj adds ~+2–3 GiB VmPeak** vs the text-only cell at the same weight/np
  (Study 2 17gb c=1 text-only VmPeak 24.41 → Study 3 17gb c=1 vision 26.24;
  dynamic c=1 27.16 → 29.03). Cross-checks Meta's published ~+2 GB vision delta.
- **DFlash composes with vision** (17gb c=1 DFlash + vision = 20.46 tok/s ≈ the
  text-only DFlash rate, 0.188 acceptance). The drafter ignores the image and
  drafts the text continuation; the speedup survives.
- **VRAM ≈ 1093 MiB across all cells** — the carve-out counter only ticks for
  carve-out-path allocations; the mmap'd GGUF and unified-host-visible GPU
  buffers don't increment it. **Trust VmPeak, not rocm-smi** — see
  [`METHODOLOGY.md §5`](METHODOLOGY.md#5-the-memory-methodology-trust-vmpeak-not-rocm-smi-or-vmhwm).

## Memory table — the full footprint picture (VmPeak)

Pulled from the cell JSONs. On Strix Halo, **VmPeak is the real footprint**
(the mmap'd GGUF + GPU-offloaded buffers all live in the process address space);
`VmHWM` undercounts (1–10 GiB) and `rocm-smi` VRAM shows only the ~32 GiB
carve-out (~1 GiB used — misleading).

| Cell | VmPeak (GiB) | VmHWM (GiB) | VRAM (MiB) | Notes |
|---|---|---|---|---|
| 17gb baseline c=1 (Study 1, greedy) | 23.91 | 1.38 | 1084 | text-only floor |
| 17gb DFlash c=1 (Study 1) | 26.73 | 2.04 | 1085 | +2.82 GiB drafter |
| dynamic baseline c=1 (Study 1) | 26.57 | 1.33 | 1084 | dynamic quant floor |
| dynamic DFlash c=1 (Study 1) | 29.28 | 1.95 | 1084 | +2.71 GiB drafter |
| 17gb baseline c=16 (Study 2) | 30.27 | 4.08 | 1089 | KV cache raises VmPeak with np |
| dynamic baseline c=16 (Study 2) | 38.41 | 9.51 | 1089 | largest baseline footprint |
| 17gb DFlash c=4 (Study 2) | 33.44 | 7.71 | 1088 | drafter + c=4 KV |
| 17gb baseline c=1 vision (Study 3) | 26.24 | 1.86 | 1093 | +1.83 GiB mmproj vs text c=1 |
| dynamic baseline c=1 vision (Study 3) | 29.03 | 1.89 | 1093 | +1.87 GiB mmproj vs text c=1 |

**Cross-check vs Meta's published memory envelope:**

| Build | Meta text-only | Meta +vision | Meta +vision+drafter | Our VmPeak (text / +vision / +drafter) |
|---|---|---|---|---|
| 17gb | ~17 GB | ~19 GB | ~20 GB | ~24 / ~26 / ~27 GiB |
| dynamic | ~20 GB | ~22 GB | ~23 GB | ~27 / ~29 / ~29 GiB |

Our numbers run ~7 GiB higher than Meta's envelope, which is expected: Meta
reports model+KV working set, while VmPeak includes the full mmap'd GGUF
mapping (16–20 GiB file) plus the working set. The **deltas match closely**
(drafter +2.7–2.8 GiB vs Meta ~+3 GB; vision +1.8–1.9 GiB vs Meta ~+2 GB), which
is the meaningful cross-check.

## Optional `llama-bench` cross-check

`matrix/llama-bench.json` holds the model-level `pp512` (prefill) and `tg128`
(decode) numbers from `llama-bench` for each weight. **This is a non-DFlash
cross-check** — `llama-bench` has no `-md` support in this build — so it
validates the baseline c=1 decode tok/s against an independent binary (no HTTP,
no slot scheduler).

| weight | pp512 (tok/s) | tg128 (tok/s) | Study 1 baseline c=1 (tok/s) | decode cross-check |
|---|---|---|---|---|
| 17gb | 318.96 | **10.73** | 10.48 | within 2.4 % ✓ |
| dynamic | 309.03 | **9.28** | 9.14 | within 1.5 % ✓ |

`llama-bench` uses `n_batch=2048`, `n_ubatch=512`, 16 threads, type_k/v=f16 — a
slightly different code path from `llama-server -np 1`, so a ~2-3 % delta is
expected and confirms the c=1 baseline decode rate is genuine (not a server/slot
artifact). Reproduce with:

```bash
B=third_party/llama.cpp/build/bin
for W in models/muse-glimmer-30B-kquant-17gb.gguf models/muse-glimmer-30B-kquant-dynamic.gguf; do
  "$B/llama-bench" -m "$W" -ngl 99 -p 512 -n 128 -o json >> docs/results/matrix/llama-bench.json
done
```
