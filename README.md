# Muse-Glimmer-30B-ROCm

[![CI](https://github.com/AIwork4me/Muse-Glimmer-30B-ROCm/actions/workflows/ci.yml/badge.svg)](https://github.com/AIwork4me/Muse-Glimmer-30B-ROCm/actions/workflows/ci.yml)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
![ROCm](https://img.shields.io/badge/ROCm-7.2.1-orange)
![GPU](https://img.shields.io/badge/GPU-gfx1151%20RDNA%203.5-red)
![Python](https://img.shields.io/badge/Python-3.12-yellow)

> Run **Meta's Muse-Glimmer-30B** (dense 29.6B vision-language model, Apache-2.0)
> on **AMD RDNA** consumer/APU silicon — validated today on the **Ryzen AI MAX+ 395
> ("Strix Halo", gfx1151)**, with **Radeon discrete GPUs** (e.g. W7900) on the roadmap.

An RDNA-focused inference + benchmarking project. Two serving paths, a full
speed/memory matrix, **DFlash speculative decoding working on llama.cpp**, and
the CDNA→RDNA adaptation documented delta-by-delta. **Not** for MI-series
datacenter cards — Meta's MI300X recipe is the reference we adapt *from*, not a
target.

---

## Contents

- [Features](#features)
- [Supported hardware](#supported-hardware)
- [Requirements](#requirements)
- [Quick start](#quick-start)
- [Headline results](#headline-results)
- [Best practices & pitfalls — read before DFlash or c=16](#best-practices--pitfalls--read-before-dflash-or-c16)
- [What changed from Meta's MI300X recipe](#what-changed-from-metas-mi300x-recipe)
- [Docs](#docs)
- [Status](#status) · [License](#license) · [Acknowledgements](#acknowledgements) · [Contributing](#contributing)

---

## Features

- **Two serving paths** — a full-feature **vLLM** server (BF16: vision, native
  `muse_glimmer` reasoning + ATEM tool-call parsers, 128K context, continuous
  batching) and a fast/light **llama.cpp GGUF** path (chat in minutes, no compile).
- **DFlash speculative decoding on llama.cpp** — **2.2–2.4× faster** decode with
  byte-identical output (greedy). Sits between Apple M5 Max (1.8×) and RTX 5090
  (3.1×) in Meta's own methodology.
- **Full benchmark matrix** — 2 weight sizes × {c1, c4, c16} × {baseline, DFlash}
  + a vision axis, with tok/s, TTFT, TPOT, and real memory footprint per cell.
- **The CDNA → RDNA adaptation, documented** — every delta from Meta's MI300X
  recipe (attention backend, precision, install, KV, spec-decode) with the *why*.
- **Pitfalls you'd otherwise hit the hard way** — DFlash @ c=16 is pathological;
  `--spec-type` defaults to a silent no-op; APU memory accounting is misleading.
  All verified, with fixes + a best-practice table.
- **Reproducible** — pinned stack, a `METHODOLOGY.md`, and CI-safe tests
  (GitHub Actions runs the no-GPU checks; GPU/server tests run locally).

## Supported hardware

| Status | GPU | Notes |
|---|---|---|
| ✅ Validated | **gfx1151 — Ryzen AI MAX+ 395 / Radeon 8060S "Strix Halo"** (RDNA 3.5) | All numbers in this repo are from this part. |
| 🚧 Roadmap | **Radeon RDNA dGPUs** (e.g. W7900 / gfx1100-class) | Same RDNA family; planned follow-on. |
| ❌ Out of scope | **MI-series** (CDNA datacenter, e.g. MI300X/MI355X) | Use Meta's upstream MI300X recipe directly; this project is RDNA-only. |

## Requirements

| Item | Value |
|---|---|
| APU/GPU | gfx1151 (Ryzen AI MAX+ 395 / Radeon 8060S) |
| ROCm | **7.2.1** (community-verified for gfx1151; 7.14.0 is the official alternative — see `docs/strix-halo-setup.md`) |
| Linux kernel | ≥ 6.16.9 (have 6.17; fixes a UMA "only 15.5 GB visible" bug) |
| Python | 3.12 (gfx1151 wheels fail on 3.13) |
| Free memory | ~20 GiB (GGUF path) · ~60 GiB (BF16 vLLM path) — unified LPDDR5X |
| Toolchain | `uv`, `cmake`, a HIP/ROCm toolchain |
| Model | Meta's `muse-glimmer-30B-kquant-17gb.gguf` (GGUF) or BF16 weights (vLLM) — Apache-2.0, **not gated, no token** |

`bash scripts/00-check-env.sh` asserts all of the above before you start.

---

## Quick start

### Fast path — GGUF, chat in minutes (no vLLM compile)

```bash
git clone https://github.com/AIwork4me/Muse-Glimmer-30B-ROCm.git
cd Muse-Glimmer-30B-ROCm
bash scripts/gguf-quickstart.sh   # builds llama.cpp (HIP/gfx1151) + fetches the Q4 GGUF
                                  # → llama-server on http://127.0.0.1:8080
```

Add DFlash (2.2× faster, identical output) by serving with:

```bash
third_party/llama.cpp/build/bin/llama-server \
  -m models/muse-glimmer-30B-kquant-17gb.gguf \
  -md models/dflash-kquant.gguf -ngld 99 \
  --spec-type draft-dflash --spec-draft-n-max 16 \   # ← required; see Pitfalls
  -ngl 999 -c 32768 --port 8080 --jinja
```

Vision: add `--mmproj models/mmproj-kquant.gguf`. Concurrency: add `-np <slots>`
(default 4 slots plateau ~22 tok/s; see Pitfalls before raising it).

### Full-feature path — vLLM (BF16, ~1 h source compile)

```bash
export PATH="$HOME/.local/bin:$PATH"     # uv lives here
uv sync                                  # TheRock gfx1151 torch + deps (Python 3.12)
bash scripts/00-check-env.sh             # ROCm 7.2.1 · kernel ≥6.16.9 · gfx1151 · ≥60 GiB
bash scripts/01-build-vllm.sh            # source-build vLLM (PR #51655) for gfx1151
bash scripts/02-fetch-model.sh           # ~55 GiB BF16 weights (parallel downloader)
bash scripts/03-serve-vllm.sh            # OpenAI server on http://127.0.0.1:8000
```

> **`uv run --no-sync` is load-bearing** on every `uv run`: vLLM is editable-installed
> and not in `uv.lock`; a bare `uv run` re-syncs and wipes it. See
> [`troubleshooting.md`](docs/troubleshooting.md).

> **Slow/region-locked link?** `02-fetch-model.sh` uses `HF_ENDPOINT=https://hf-mirror.com`
> + a 24-connection parallel range downloader (`scripts/hf_parallel_get.py`).

---

## Headline results

### DFlash speculative decoding (greedy, batch-1 — Meta-comparable)

Same methodology as Meta's published table (llama.cpp, K-Quant-17GB + quantized drafter):

| GPU | Baseline (tok/s) | DFlash (tok/s) | Speedup |
|---|---|---|---|
| Nvidia RTX 5090 *(Meta)* | 74.9 | 233.4 | 3.1× |
| Apple M5 Max *(Meta)* | 26.6 | 50.2 | 1.8× |
| **gfx1151, 17gb *(this repo)*** | **10.48** | **23.03** | **2.20×** |
| **gfx1151, dynamic *(this repo)*** | **9.14** | **21.82** | **2.39×** |

Byte-equivalence **PASS** — greedy spec-decode is exact (baseline & DFlash both
emit `391` for `17 × 23`).

### vLLM (BF16) vs llama.cpp (Q4 K-quant) — throughput under load

`/v1/completions`, 512 output tokens/request, GPU (HIP):

| Concurrency | vLLM BF16 (tok/s) | llama.cpp Q4 (tok/s) | Q4 speedup |
|---|---|---|---|
| 1 | 4.2 | **10.5** | 2.5× |
| 4 | 14.0 | **21.3** | 1.5× |
| 16 | 40.8 | **102.0** | 2.5× |

llama.cpp Q4 is faster at every concurrency (~4× less memory traffic — decode is
bandwidth-bound on this APU). **vLLM's edge is features**, not speed: native
`muse_glimmer` reasoning + ATEM tool-call parsing, vision/multimodal, 128K context,
automatic continuous batching. DFlash's speedup **shrinks with concurrency**
(c1 ~2.2× → c4 ~1.35–1.75×) and **breaks at c=16** — see below.

Full matrix + methodology: [`docs/results/benchmark.md`](docs/results/benchmark.md),
[`docs/results/METHODOLOGY.md`](docs/results/METHODOLOGY.md).

---

## Best practices & pitfalls — read before DFlash or c=16

### Best-practice config table

| Use case | Config | Expected |
|---|---|---|
| Single-stream chat (c=1) | **DFlash ON** — `--spec-type draft-dflash --spec-draft-n-max 16` | **~2.2× faster**, identical output, ~+3 GiB |
| Light concurrent (c ≤ 4) | **DFlash ON** | ~1.3–1.75× aggregate; mind the +3 GiB drafter footprint |
| **High throughput (c ≥ 8, esp c=16)** | **DFlash OFF** (baseline) | c16 ~31–34 tok/s; DFlash here is **pathological** |

> **DFlash is a silent no-op without `--spec-type`.** `llama-server --spec-type`
> defaults to `none`, so `-md dflash.gguf` alone loads the drafter but never
> drafts (1.0×). Always pass `--spec-type draft-dflash --spec-draft-n-max 16`
> (`n_max=16` is the measured sweet spot = DFlash block_size).

### ⚠ Do NOT combine DFlash with `-np 16`

It is **pathologically slow** — verified, not a fluke:

- **17gb c=16 DFlash:** a 16×48-token batch completed **0 of 768 tokens in 28 s**
  while baseline c=16 ran at 34.5 tok/s — effectively >1000× slower per-request.
- **dynamic c=16 DFlash:** the benchmark cell was **aborted after 5 h 16 m**;
  draft acceptance collapsed to **0.18 %** (6,060 accepted / 3,270,000 drafted).
- **Root cause:** at c=16 the drafter fires for all 16 slots at once, generating
  millions of tokens that are >99.8 % rejected, while the full generate+verify
  cost for those rejected drafts is paid in full — spec-decode goes into reverse.
- **c=16 itself is fine** (baseline 17gb 34.5, dynamic 31.0 tok/s). The pathology
  is DFlash-specific.

Full evidence: [`benchmark.md` — c=16 + DFlash: do not use](docs/results/benchmark.md#c16-dflash-do-not-use).

### Memory on Strix Halo — trust VmPeak, not `rocm-smi`

The real footprint is the process **VmPeak** (~24–32 GiB). `rocm-smi` VRAM reports
only the ~32 GiB dedicated carve-out, and `/proc` VmHWM undercounts (the GGUF is
mmap'd + GPU-offloaded). See
[`troubleshooting.md` — memory-footprint-apu](docs/troubleshooting.md#memory-footprint-apu).

---

## What changed from Meta's MI300X recipe

Meta's reference targets CDNA (MI300X). Retargeting to RDNA (gfx1151):

| | MI300X (Meta) | **gfx1151 (this repo)** |
|---|---|---|
| Install | docker nightly | **source build** (`PYTORCH_ROCM_ARCH=gfx1151`) — no gfx1151 docker |
| Precision | bf16 / fp8 | **bf16 only** (RDNA 3.5 has no usable FP8) |
| Attention | `ROCM_AITER_FA` | **`TRITON_ATTN`** (AITER is CDNA3+/RDNA4-only; `FLASH_ATTN` also fails — no gfx1151 codegen) |
| Tensor-parallel | 4 | **1** (single iGPU) |
| Spec-decoding | DFlash (vLLM) | **DFlash via llama.cpp** (vLLM path hits an upstream registry bug) |

Full delta table with the *why* + memory math: [`docs/adaptation.md`](docs/adaptation.md).

---

## Docs

- [**adaptation.md**](docs/adaptation.md) — the MI300X → gfx1151 delta, explained (the educational centerpiece).
- [**results/benchmark.md**](docs/results/benchmark.md) — full Study 1/2/3 + the c=16 write-up.
- [**results/METHODOLOGY.md**](docs/results/METHODOLOGY.md) — how every number was measured.
- [strix-halo-setup.md](docs/strix-halo-setup.md) — host prerequisites.
- [troubleshooting.md](docs/troubleshooting.md) — symptom → cause → fix.

## Status

✅ vLLM (BF16, `TRITON_ATTN`) · ✅ llama.cpp GGUF (Q4 kquant) · ✅ DFlash on llama.cpp
(2.2–2.4×, byte-equivalent) · ✅ full benchmark matrix (Studies 1/2/3) · ✅ vision path
loaded + measured · ✅ CI-safe tests. 🚧 Radeon dGPU (W7900) support · ⏸ ROCm 7.14.0
isolated comparison — 7.14.0 (released 2026-07-16, first official gfx1151 APU support)
ships via **TheRock**, not the legacy apt repo (which tops at 7.2.4); side-by-side plan,
non-destructive to the proven 7.2.1 stack. See [`docs/strix-halo-setup.md`](docs/strix-halo-setup.md).

## License

- **This repository's code:** [Apache License 2.0](LICENSE).
- **Model weights:** Meta's Muse-Glimmer-30B, released under
  [Apache-2.0](https://huggingface.co/meta-models/Muse-Glimmer-30B). Download
  implies acceptance of Meta's [usage policy](https://huggingface.co/meta-models/Muse-Glimmer-30B/blob/main/USAGE_POLICY.md).

## Acknowledgements

- **Meta Superintelligence Lab** for Muse-Glimmer-30B (Apache-2.0) + the published GGUFs & DFlash drafter.
- [**llama.cpp**](https://github.com/ggml-org/llama.cpp) (first-class `muse_glimmer` + DFlash support) and [**vLLM**](https://github.com/vllm-project/vllm) (PR #51655).
- AMD **TheRock** for the gfx1151 PyTorch nightlies, and the **ROCm** toolchain.

## Contributing

This is a hardware-specific project (gfx1151 today). GPU/server tests can't run in
CI, so the workflow is: CI runs the no-GPU lint/config/JSON-schema tests
(`uv run --no-sync pytest -m "not gpu and not server"`); hardware validation is the
documented `docs/results/` + a local run. See `docs/troubleshooting.md` for the
non-obvious environment constraints before you build. Open an issue for
Radeon-dGPU (W7900) porting or ROCm 7.14.0 work.
