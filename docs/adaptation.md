# Adaptation: MI300X recipe → gfx1151 (Strix Halo)

> The educational centerpiece of this project. Every row below is a deliberate
> delta from Meta's upstream vLLM recipe ([vllm-project/recipes#776][recipe]),
> which targets CDNA datacenter GPUs (MI300X = gfx942, MI355X = gfx942/950).
> We retarget it to **gfx1151** — AMD Ryzen AI MAX+ PRO 395 / Radeon 8060S
> "Strix Halo", an RDNA 3.5 APU — and explain *why* each change is required.

The model is **Muse-Glimmer-30B** (`meta-models/Muse-Glimmer-30B`), a dense
29.6B vision-language model with a new Meta architecture
(`MuseGlimmerForConditionalGeneration`, `model_type: muse_glimmer`) — not
GPT-OSS, not Llama. Its vLLM support ([vllm-project/vllm#51655][pr]) lives in no
released wheel, which is the root reason for most of the deltas below.

## The delta table

| Aspect | MI300X recipe (PR #776) | **gfx1151 adaptation** | Why (verified) |
|---|---|---|---|
| vLLM | 0.27.0+ / nightly, PR #51655 | same | `muse_glimmer` model code + parsers are in **no released wheel** |
| Install | docker `vllm/vllm-openai-rocm:nightly` | **source build**, PR #51655 pinned commit, `PYTORCH_ROCM_ARCH=gfx1151` + `import amdsmi` shim | AMD: Ryzen APUs are pip/source-only; **no gfx1151 docker**; prebuilt images throw `invalid device function` ([ROCm/ROCm#4909][invdev]) |
| PyTorch | (in image) | **TheRock gfx1151 nightly** wheel, py3.12, numpy<2 | stock ROCm wheels lack gfx1151 codegen |
| Precision | bf16 (72 GB) / fp8_block (40 GB) | **bf16 only** | FP8 matrix units are RDNA4/CDNA3+; vLLM FP8 won't run on gfx1151 |
| Attention backend | `--attention-backend ROCM_AITER_FA` | **`FLASH_ATTN`** (`FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE`), fallback `TRITON_ATTN` | AITER is hard-gated to CDNA3+/RDNA4 via `get_cdna_version()>2` ([vllm-project/vllm#51136][aiter]) |
| Env | `VLLM_ROCM_USE_AITER=1` | **not set** (auto-disables) | same AITER gate |
| Tensor-parallel | `--tensor-parallel-size 4` | **TP=1** | single integrated GPU |
| Chunked prefill | (default on) | **off** | hangs on RDNA ([vllm-project/vllm#5013][chunked]) |
| KV-cache dtype | bf16 | bf16 (no fp8 KV) | fp8 KV is CDNA-only |
| Spec-decoding | DFlash (`Muse-Glimmer-30B-assistant`) | **deferred (v1 off)** | registry bug `DFlashMuseGlimmer…`; needs a patched fork |
| Kernel | — | **≥ 6.16.9** (have 6.17) | fixes the "ROCm sees only ~15.5 GB" UMA bug ([ROCm/ROCm#5444][uma]) |

[recipe]: https://github.com/vllm-project/recipes/pull/776
[pr]: https://github.com/vllm-project/vllm/pull/51655
[invdev]: https://github.com/ROCm/ROCm/issues/4909
[aiter]: https://github.com/vllm-project/vllm/issues/51136
[chunked]: https://github.com/vllm-project/vllm/issues/5013
[uma]: https://github.com/ROCm/ROCm/issues/5444

## Why each row — in prose

**vLLM / Install — source build, not docker.** The model's architecture and its
`muse_glimmer` reasoning/tool-call parsers exist only on the PR #51655 branch,
which has not landed in a release or a nightly docker tag. Worse, every prebuilt
ROCm docker image is codegen'd for CDNA targets; on gfx1151 it throws
`invalid device function` the moment a HIP kernel runs. So we build vLLM from
source (`scripts/01-build-vllm.sh`) with `PYTORCH_ROCM_ARCH=gfx1151`, into the uv
venv, and apply an `import amdsmi`-before-torch shim
(`patches/vllm-amdsmi-import.diff`) that prevents an import-time crash on this
APU. The pin: `xianbaoqian/vllm` @ `606a12cd` (branch `tiezhen/new-model-support`).

**PyTorch — TheRock gfx1151 wheel.** The stock `torch` ROCm wheels are built for
gfx9xx/gfx11-generic and contain no gfx1151 kernels. AMD's **TheRock** project
publishes gfx1151-codegen'd nightlies at
`https://rocm.nightlies.amd.com/v2/gfx1151/`; `pyproject.toml` pins
`torch==2.10.0+rocm7.13.0a20260513` from that index. Python must be **3.12**
(the gfx1151 wheels fail to import on 3.13) and **numpy<2**.

**Precision — BF16 only.** RDNA 3.5 has no usable FP8 path in vLLM. The model's
weights ship BF16 (~59 GB), and that is what we run — see the memory math below.

**Attention — `FLASH_ATTN`, never AITER.** AITER (AMD's tuned attention/decoder
kernels) is gated behind `get_cdna_version() > 2`, i.e. CDNA3+ / RDNA4 only. On
gfx1151 it is silently absent; setting `VLLM_ROCM_USE_AITER=1` only makes vLLM
fall back to broken emulation. We instead set
`FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE` and pass `--attention-backend
FLASH_ATTN` (the Triton-on-AMD path), with `TRITON_ATTN` as a documented
fallback. This single env/flag pair is the heart of the RDNA adaptation.

**Tensor-parallel = 1.** Strix Halo is one integrated GPU sharing unified
memory; there is nothing to shard across.

**Chunked prefill off.** RDNA's prefill path hangs under chunked scheduling; we
leave it off (its absence is asserted by the banned-flag test).

**Speculative decoding deferred.** The DFlash assistant model hits a registry bug
upstream (`DFlashMuseGlimmerAssistant`). It is a v1 non-goal; the closest working
precedent needed a patched fork.

**Kernel ≥ 6.16.9.** Older kernels expose only ~15.5 GB of the unified pool to
ROCm (a KFD/HSA UMA-handling bug). 6.16.9 fixes it; this host runs 6.17.

## Memory math (BF16)

```
weights        ~59.2 GB            (two safetensors shards, 50 + 9.6 GB)
KV @ 128K       ~7   GB            (2(K+V) × 2 kv-heads × 128 × 52 layers × 2 B
                                    ≈ 52 KiB/token × 131072)
activations     ~2–4 GB
─────────────────────────────
total          ~68   GB            fits in 94 GB unified with ~20 GB headroom
                                   for concurrency  →  --gpu-memory-utilization 0.90
```

## Closest precedent (verbatim)

PR #51655 comment by *BlivionIaG* (2026-08-10):

> *"I have dflash working with the authors pr on their fork … got it working on
> two gfx1100 cards. Running on rdna3 (rocm 7.2.0, pytorch 2.12, kernel 6.8.0)."*

This confirms the model runs on consumer RDNA silicon with our stack family
(gfx1100 ≈ RDNA3; we are one generation newer on gfx1151 / RDNA 3.5).
