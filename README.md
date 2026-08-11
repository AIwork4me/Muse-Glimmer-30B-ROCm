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

## What changed from Meta's MI300X recipe

| | MI300X | **gfx1151** |
|---|---|---|
| Install | docker nightly | **source build** (`PYTORCH_ROCM_ARCH=gfx1151`) |
| Precision | bf16 / fp8 | **bf16 only** (RDNA 3.5 has no FP8) |
| Attention | `ROCM_AITER_FA` | **`FLASH_ATTN`** (AITER is CDNA3+/RDNA4-only) |
| TP | 4 | **1** (single iGPU) |
| Chunked prefill | on | **off** (hangs on RDNA) |

Full table with the *why* + memory math:
[`docs/adaptation.md`](docs/adaptation.md).

## Results

| Engine | Precision | Concurrency | tok/s | peak VRAM |
|---|---|---|---|---|
| _to be filled from_ `docs/results/` | | | | |

Run `bash scripts/benchmark.sh` against a live server to populate this.

## Docs

- [**adaptation.md**](docs/adaptation.md) — the MI300X → gfx1151 delta, explained.
- [strix-halo-setup.md](docs/strix-halo-setup.md) — host prerequisites.
- [troubleshooting.md](docs/troubleshooting.md) — symptom → cause → fix.

## Status

vLLM source build ✅ · serve config ✅ · CI-safe tests ✅ · model fetch ✅ (parallel) ·
live smoke/parser/benchmark — pending a model download + server run on the box.
