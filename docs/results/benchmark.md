# Benchmark — Muse-Glimmer-30B on gfx1151 (Strix Halo)

> Reproduce: start the server (`bash scripts/03-serve-vllm.sh`), then
> `BASE=http://127.0.0.1:8000 bash scripts/benchmark.sh`. Raw JSON lands in
> `docs/results/*.json` (gitignored runtime artifacts); the numbers below are the
> validated reference run.

## Environment manifest

| Item | Value |
|---|---|
| GPU | AMD Radeon 8060S, **gfx1151** (RDNA 3.5), 40 CUs |
| Memory | 94 GiB unified LPDDR5X (vLLM sees an 80 GiB pool) |
| ROCm | 7.2.1 |
| Kernel | 6.17.0-1020-oem |
| PyTorch | 2.10.0+rocm7.13.0a20260513 (TheRock gfx1151) |
| vLLM | 0.1.dev1+g606a12cd7 (source-built, PR #51655) |
| Precision | **BF16** |
| Attention | **TRITON_ATTN** (`FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE`) |
| Tensor-parallel | 1 |
| max-model-len | 131072 |
| gpu-memory-utilization | 0.90 |

**Memory at startup** (vLLM's own accounting): 56.49 GiB weights + 1.89 GiB peak
activation + 0.72 GiB CUDA graphs, with **13.62 GiB KV cache** — i.e. the model
fits in the unified pool with headroom.

## Throughput (`/v1/completions`, 512 output tokens/request, temperature 1.0)

| Concurrency | total out tokens | wall (s) | aggregate tok/s |
|---|---|---|---|
| 1 | 512 | 121.0 | **4.23** |
| 4 | 1,735 | 124.2 | **13.97** |
| 16 | 5,439 | 133.3 | **40.81** |

**Reading the numbers.** Single-stream ≈ 4 tok/s is the realistic interactive
chat speed for a dense 30B BF16 model on this iGPU (no AITER acceleration is
available on RDNA 3.5 — see `docs/adaptation.md`). Aggregate throughput scales
≈10× from c=1 to c=16, showing vLLM's continuous batching (with default
chunked-prefill) is healthy on gfx1151 — the historical RDNA chunked-prefill hang
([vllm-project/vllm#5013][chunked]) did **not** reproduce with `TRITON_ATTN`.

## Caveats

- **Muse-Glimmer is a reasoning model.** `/v1/completions` throughput measures raw
  decode speed; a real chat turn first emits a chain-of-thought in the `reasoning`
  channel, then the answer — so end-to-end latency per turn is higher than
  `512 tok / tok_s`. (`/v1/chat/completions` exercises the parser path; see
  `tests/test_smoke.py`, `tests/test_parsers.py`.)
- **`rocm-smi` VRAM is not the real footprint on Strix Halo.** It reports only the
  ~32 GiB dedicated carve-out (and shows ~1 GiB "used" — misleading). Trust vLLM's
  startup accounting above for actual unified-memory usage, not the `vram_peak`
  field the benchmark script snapshots.

[chunked]: https://github.com/vllm-project/vllm/issues/5013
