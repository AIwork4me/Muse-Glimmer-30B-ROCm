# Muse-Glimmer-30B-ROCm

[![ci](https://github.com/USER/muse-glimmer-30b-rocm/actions/workflows/ci.yml/badge.svg)](.github/workflows/ci.yml)
![BF16](https://img.shields.io/badge/precision-BF16-blue)
![gfx1151](https://img.shields.io/badge/GPU-gfx1151%20RDNA%203.5-red)
![ROCm](https://img.shields.io/badge/ROCm-7.2.1-orange)

Run **Meta's Muse-Glimmer-30B** (dense 29.6B vision-language model, Apache 2.0)
on **AMD gfx1151** — Ryzen AI MAX+ PRO 395 / Radeon 8060S "Strix Halo", RDNA 3.5 —
via **vLLM on ROCm 7.2.1**, in BF16. Adapted from Meta's MI300X recipe; every
delta is documented in [`docs/adaptation.md`](docs/adaptation.md).

Two paths: a **full-feature vLLM** server (vision + the `muse_glimmer` reasoning
& ATEM tool-call parsers) and a **GGUF quick-start** (chat in minutes, no
compile).

> **CI has no gfx1151.** GitHub Actions runs only the no-GPU lint/banned-flag
> tests (`-m "not gpu and not server"`). The GPU + server tests are marked and
> run locally on the Strix Halo box.

---

## TL;DR — vLLM path (full feature)

```bash
git clone <this-repo> && cd muse-glimmer-30b-rocm
uv sync                          # TheRock gfx1151 torch + deps (Python 3.12)
bash scripts/00-check-env.sh     # ROCm 7.2.1 · kernel ≥6.16.9 · gfx1151 · ≥60 GB pool
bash scripts/01-build-vllm.sh    # source-build vLLM (PR #51655) for gfx1151  (long)
bash scripts/02-fetch-model.sh   # download ~55 GiB BF16 weights
bash scripts/03-serve-vllm.sh    # OpenAI server on http://127.0.0.1:8000
```

```bash
uv run --no-sync pytest tests/test_smoke.py tests/test_parsers.py -v -m server
```

> **`--no-sync` is load-bearing** on every `uv run`: vLLM is source-installed
> editable and is not in `uv.lock`; a bare `uv run` re-syncs and wipes it. See
> [troubleshooting.md#no-sync](docs/troubleshooting.md#no-sync).

### Model download on a slow / region-locked link

`02-fetch-model.sh` defaults to **`https://hf-mirror.com`** (`HF_ENDPOINT`) and
pulls the two weight shards with a built-in **parallel range downloader**
(`scripts/hf_parallel_get.py`, 24 connections) — on a ~0.2 MiB/s-per-connection
link this is ~16× faster than the stock single-stream `hf download`. Override
with `HF_ENDPOINT=…` / `NCONNS=…`; set `USE_HF_DOWNLOAD=1` for the plain tool.
Full rationale: [troubleshooting.md#model-fetch-slow](docs/troubleshooting.md#model-fetch-slow).

## TL;DR — GGUF path (no-compile chat)

```bash
bash scripts/gguf-quickstart.sh  # builds llama.cpp HIP (gfx1151) + fetches Q4_K_M GGUF
                                 # -> llama-server on http://127.0.0.1:8080
```

Text-focused (llama.cpp's VLM support lags upstream); for vision + agentic
tool-calls use the vLLM path. See [`docs/adaptation.md`](docs/adaptation.md).

---

## Best practices / Pitfalls — read this before using DFlash or c=16

The DFlash + full benchmark matrix
([`docs/results/benchmark.md`](docs/results/benchmark.md),
[`docs/results/METHODOLOGY.md`](docs/results/METHODOLOGY.md)) uncovered two
operational gotchas. Follow this table and you will not hit them.

### Best-practice config table

| Use case | Config | Expected |
|---|---|---|
| Single-stream interactive chat (c=1) | **DFlash ON**, `--spec-type draft-dflash --spec-draft-n-max 16` | **~2.2× faster** (10.5 → 23.0 tok/s on 17gb), identical output under greedy, ~+3 GiB VmPeak |
| Light concurrent (c ≤ 4) | **DFlash ON** | ~1.3–1.75× aggregate; mind the +3 GiB drafter footprint |
| **High throughput (c ≥ 8, esp c=16)** | **DFlash OFF** (baseline) | c16 ~31–34 tok/s; **DFlash here is pathological** |

> When DFlash is on, you MUST pass `--spec-type draft-dflash --spec-draft-n-max 16`
> (plus `-md models/dflash-kquant.gguf -ngld 99`). **`--spec-type` defaults to
> `none`**, so `-md …` alone loads the draft but never drafts — a silent 1.0×
> no-op. `n_max=16` is the measured sweet spot (= DFlash block_size).

### ⚠ Do NOT combine DFlash with `-np 16`

It is **pathologically slow** — verified, not a fluke:

- **17gb c=16 DFlash:** a 16×48-token probe batch completed **0 of 768 tokens in
  28 s** while the baseline c=16 run delivered 34.5 tok/s — effectively >1000×
  slower per-request.
- **dynamic c=16 DFlash:** the full benchmark cell was **aborted after 5 h 16 m**
  with no completion; draft acceptance collapsed to **0.18 %** (6,060 accepted /
  3,270,000 draft tokens), ~1/37× the baseline aggregate rate.
- **Root cause:** at c=16 the drafter fires for all 16 slots at once, generating
  millions of tokens that are >99.8 % rejected, while the full generate+verify
  cost for those rejected drafts is paid in full. Spec-decode goes into reverse.
- **c=16 itself is fine** (baseline 17gb 34.5, dynamic 31.0 tok/s). The pathology
  is DFlash-specific.

Full evidence + write-up:
[`docs/results/benchmark.md` — c=16 + DFlash: do not use](docs/results/benchmark.md#c16--dflash-do-not-use).
Troubleshooting entries: [dflash-c16-pathological](docs/troubleshooting.md#dflash-c16-pathological),
[dflash-silent-noop](docs/troubleshooting.md#dflash-silent-noop),
[memory-footprint-apu](docs/troubleshooting.md#memory-footprint-apu).

### Headline DFlash result (gfx1151 vs Meta's anchors)

Same methodology as Meta's published table (greedy, batch 1, K-Quant-17GB +
quantized drafter, llama.cpp):

| GPU | Baseline (tok/s) | DFlash (tok/s) | Speedup |
|---|---|---|---|
| Nvidia RTX 5090 (Meta) | 74.9 | 233.4 | 3.1× |
| Apple M5 Max (Meta) | 26.6 | 50.2 | 1.8× |
| **gfx1151, 17gb (this repo)** | **10.48** | **23.03** | **2.20×** |
| **gfx1151, dynamic (this repo)** | **9.14** | **21.82** | **2.39×** |

gfx1151 sits **between the M5 Max and the RTX 5090** — credible for a 50 TOPS
NPU/iGPU-class part. Byte-equivalence PASS (greedy spec-decode is exact: both
baseline and DFlash emit `391` for `17 × 23`). See
[`docs/results/benchmark.md` — Study 1](docs/results/benchmark.md#study-1--dflash-anchor-greedy-batch-1--meta-comparable).

---

## What changed from Meta's MI300X recipe

| | MI300X | **gfx1151** |
|---|---|---|
| Install | docker nightly | **source build** (`PYTORCH_ROCM_ARCH=gfx1151`) |
| Precision | bf16 / fp8 | **bf16 only** (RDNA 3.5 has no FP8) |
| Attention | `ROCM_AITER_FA` | **`TRITON_ATTN`** (AITER is CDNA3+/RDNA4-only; FLASH_ATTN also fails — no gfx1151 codegen) |
| TP | 4 | **1** (single iGPU) |
| Chunked prefill | on | **off** (hangs on RDNA) |

Full table with the *why* + memory math:
[`docs/adaptation.md`](docs/adaptation.md).

## Results — vLLM (BF16) vs llama.cpp (Q4 K-quant)

Muse-Glimmer-30B on gfx1151 (Strix Halo), GPU (HIP), `/v1/completions`, 512 output tokens/request:

| Concurrency | vLLM BF16 (tok/s) | llama.cpp Q4 (tok/s) | Q4 speedup |
|---|---|---|---|
| 1 | 4.2 | **10.5** | 2.5× |
| 4 | 14.0 | **21.3** | 1.5× |
| 16 | 40.8 | **102.0** | 2.5× |

llama.cpp Q4 is faster at every concurrency (~4× less memory traffic per token —
decode is bandwidth-bound on this APU). vLLM's edge is features, not speed:
native `muse_glimmer` reasoning + ATEM tool-call parsing, vision/multimodal,
128K context, and **automatic** continuous batching (llama.cpp needs `-np <slots>`
tuning to scale — its default 4 slots plateau at ~22 tok/s). Both produce correct
inference (e.g. "17 × 24 → 408"). Full comparison, quality checks, feature
matrix, and when-to-use-which: [`docs/results/benchmark.md`](docs/results/benchmark.md).

Reproduce: `BASE=http://127.0.0.1:<port> bash scripts/benchmark.sh` (8000 = vLLM, 8080 = llama.cpp).

## Docs

- [**adaptation.md**](docs/adaptation.md) — the MI300X → gfx1151 delta, explained.
- [strix-halo-setup.md](docs/strix-halo-setup.md) — host prerequisites.
- [troubleshooting.md](docs/troubleshooting.md) — symptom → cause → fix.

## Status

vLLM (BF16, `TRITON_ATTN`) ✅ · llama.cpp GGUF (Q4 kquant) ✅ · live smoke + `muse_glimmer`
parser tests ✅ · model fetch ✅ (parallel) · vLLM-vs-llama.cpp benchmark ✅ · CI-safe tests ✅.
