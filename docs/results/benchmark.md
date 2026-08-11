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
