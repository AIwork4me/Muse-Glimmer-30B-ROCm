# Muse-Glimmer-30B-ROCm — Design Spec

- **Date:** 2026-08-11
- **Status:** Approved (brainstorming phase) → awaiting implementation plan
- **Owner:** maintainer (AMD ROCm inference)
- **Approach chosen:** **A** — vLLM (source-built) on ROCm 7.2.1 + Meta-GGUF quick-start, on gfx1151 (Strix Halo)

---

## 1. Overview & problem statement

Meta released **Muse-Glimmer-30B**, a dense 29.6B vision-language model (Apache 2.0, not gated) tuned for local agentic use. The upstream serving reference is **vLLM recipes PR #776** ("[ROCm] Add muse glimmer 30b mi300x/mi355x recipes"), which targets **CDNA datacenter GPUs** (MI300X = gfx942, MI355X = gfx942/950) and relies on CDNA-only kernels (AITER) and, optionally, FP8 weights.

This project adapts that recipe to run on **consumer/ACP RDNA 3.5 silicon: gfx1151 (AMD Ryzen AI MAX+ PRO 395 / Radeon 8060S "Strix Halo")** with **ROCm 7.2.1**, validates it, and ships it as a reproducible, well-documented open-source project — **`Muse-Glimmer-30B-ROCm`** — so the community can both run the model on AMD integrated graphics and learn *why* each adaptation was made.

The central technical work is a precise set of deltas from the CDNA recipe (drop AITER, swap attention backend, run BF16 instead of FP8, source-build vLLM because no gfx1151 docker exists), each justified by verified evidence.

## 2. Goals & success criteria

**A newcomer can:**
1. `git clone` → `uv sync` → run one build script → get a working **OpenAI-compatible vLLM server** serving Muse-Glimmer-30B in **BF16** on gfx1151;
2. **or** run the **GGUF quick-start** and be chatting in minutes without compiling vLLM;
3. reproduce the **documented benchmark numbers** (tokens/s, TTFT, TPOT, peak VRAM) on the reference Strix Halo box;
4. read `docs/adaptation.md` and understand **every delta** from the MI300X recipe and why.

**Non-goals (v1):** DFlash speculative decoding (upstream registry bug), multi-GPU / TP>1 (single iGPU), FP8 inference (impossible on RDNA 3.5), training/fine-tuning, Windows support. Each is documented as a known limitation with a forward path.

## 3. Target hardware & software (verified)

| Item | Value | Source |
|---|---|---|
| APU | AMD Ryzen AI MAX+ PRO 395 w/ Radeon 8060S | `rocminfo` |
| GPU compute target | **gfx1151** (RDNA 3.5) | `rocminfo` |
| Compute units | 40 | `rocminfo` |
| Memory | **94 GiB unified LPDDR5X** (~215 GB/s measured, 8-channel) | `free -h`, Strix Halo benchmarks |
| CPU threads | 32 (Zen 5) | `nproc` |
| Disk free | ~1.6 TB | `df -h` |
| ROCm | **7.2.1** | `/opt/rocm/.info/version` |
| Kernel | **6.17.0-1020-oem** (≥ 6.16.9 required) | `uname -r` |
| Python | 3.12 (TheRock gfx1151 wheels fail on 3.13) | chosen |
| Env manager | **uv** | user requirement |

**Note on ROCm version:** gfx1151 officially enters AMD's compatibility matrix at ROCm 7.14.0, but community-verified to work on 7.0–7.2.x; our box runs 7.2.1 and matches the only known-good consumer-RDNA precedent for this model (gfx1100 + ROCm 7.2.0). We stay on 7.2.1 (no host upgrade) for v1; 7.14.0 is documented as an alternative in `docs/`.

## 4. The model (verified from `config.json`)

`meta-models/Muse-Glimmer-30B` — Apache 2.0, **not gated, no HF token** required.

- Architecture class: `MuseGlimmerForConditionalGeneration` (`model_type: muse_glimmer`) — **new Meta architecture**, not GPT-OSS, not Llama.
- **Dense 29.6B** (no MoE): ~28B text decoder + ~1.8B ViT-G/14 vision tower.
- Text decoder: 52 layers, hidden 6656, intermediate 19968 (SwiGLU), vocab 202,048, **128K context** (131072).
- Attention: **GQA 32 query / 2 KV heads**, head_dim 128; layer pattern `[sliding, sliding, sliding, full]` repeating (13 full-attention layers), **sliding_window 2048**.
- RoPE θ = 500000; **per-layer `layer_rope_theta`** (full-attention layers use 0).
- **QK-norm with learned scaling**: `qk_scale_factor 3.87`, `output_multiplier 0.19611614`.
- `final_logit_softcapping 20.0`.
- Weights **BF16**, ~59.6 GB on disk (50 + 9.6 GB safetensors).
- Requires `transformers == 5.15.0.dev0`.
- Output: channel-scoped reasoning + **ATEM (XML-style) tool calls** (not JSON) → needs dedicated `muse_glimmer` reasoning & tool-call parsers.
- vLLM support: **PR #51655** (min vLLM 0.27.0 / nightly); model code + parsers are **not in any released wheel**.

## 5. The adaptation — MI300X recipe → gfx1151 (the core of the project)

| Aspect | MI300X recipe (PR #776) | **gfx1151 adaptation** | Why (verified) |
|---|---|---|---|
| vLLM | 0.27.0+/nightly, PR #51655 | same | `muse_glimmer` code/parsers in no released wheel |
| Install | docker `vllm/vllm-openai-rocm:nightly` | **source build**, PR #51655 pinned commit, `PYTORCH_ROCM_ARCH=gfx1151` + `import amdsmi` shim | AMD: Ryzen APUs are pip/source-only; no gfx1151 docker; prebuilt images throw `invalid device function` (ROCm #4909) |
| PyTorch | (in image) | **TheRock gfx1151 nightly** wheel (`rocm.nightlies.amd.com/v2/gfx1151/`, py3.12, numpy<2) | stock ROCm wheels lack gfx1151 codegen |
| Precision | bf16 (72 GB) / fp8_block (40 GB) | **bf16 only** — ~59 GB weights + ~7 GB KV@128K ≈ 68 GB, fits 94 GB | FP8 matrix units are RDNA4/CDNA3+; vLLM FP8 won't run on gfx1151 |
| Attention backend | `--attention-backend ROCM_AITER_FA` | **`FLASH_ATTN`** (`FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE`), fallback `TRITON_ATTN` | AITER hard-gated to CDNA3+/RDNA4 via `get_cdna_version()>2` (vLLM #51136) |
| Env | `VLLM_ROCM_USE_AITER=1` | **not set** (auto-disables) | same |
| Tensor-parallel | `--tensor-parallel-size 4` | **TP=1** | single iGPU |
| Chunked prefill | (default on) | **off** | hangs on RDNA (vLLM #5013) |
| KV-cache dtype | bf16 | bf16 (no fp8 KV) | fp8 KV is CDNA-only |
| Spec-decoding | DFlash (`Muse-Glimmer-30B-assistant`) | **deferred (v1 off)** | registry bug `DFlashMuseGlimmer…`; gfx1100 precedent needed a patched fork |
| Kernel | — | **≥ 6.16.9** (have 6.17) | fixes "ROCm sees only ~15.5 GB" UMA bug (ROCm #5444) |

**Memory math (BF16):** weights ~59.2 GB; KV/token = 2(K+V)×2 kv-heads×128×52 layers×2 B ≈ 52 KiB → KV(128K) ≈ 7 GB; activations/workspace ~2–4 GB ⇒ **~68 GB**, fits 94 GB with ~20 GB headroom for concurrency.

**Closest precedent (verbatim):** PR #51655 comment by *BlivionIaG* (2026-08-10): *"I have dflash working with the authors pr on their fork … got it working on two gfx1100 cards. Running on rdna3 (rocm 7.2.0, pytorch 2.12, kernel 6.8.0)."* — confirms the model runs on consumer RDNA with our stack family.

## 6. Architecture: environment layering

```
uv venv (.venv, Python 3.12)                 ← `uv sync` creates & populates
  ├─ torch == <TheRock gfx1151 pin>          via [tool.uv.index]: rocm.nightlies.amd.com/v2/gfx1151
  ├─ transformers == 5.15.0.dev0              (model-config requirement)
  ├─ numpy < 2, huggingface_hub, compressed-tensors, aiohttp, …
  └─ vllm (editable)                          ← 01-build-vllm.sh installs INTO this venv
       source: git @ <pinned PR#51655 commit> + patches/vllm-amdsmi-import.diff
       build env: PYTORCH_ROCM_ARCH=gfx1151, FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE, MAX_JOBS=16
```

uv owns the venv and all pure-Python deps; vLLM is source-built into it because its HIP compilation flags cannot be expressed as a normal wheel install.

## 7. Repository structure

```
Muse-Glimmer-30B-ROCm/
├── README.md                   # TL;DR two-path quick-start, badges, results table
├── pyproject.toml              # uv project: pins torch(TheRock gfx1151), transformers 5.15.0.dev0, …
├── uv.lock                     # fully locked, reproducible env
├── .python-version             # 3.12
├── docs/
│   ├── adaptation.md           # MI300X-recipe → gfx1151 delta table  ← educational centerpiece
│   ├── strix-halo-setup.md     # kernel≥6.16.9, ROCm 7.2.1, UMA carve-out / VRAM-pool checks
│   ├── troubleshooting.md      # AITER, FP8, chunked-prefill, 15.5 GB bug, amdsmi shim, …
│   └── results/                # benchmark logs + charts + env manifest
├── scripts/
│   ├── 00-check-env.sh         # assert ROCm 7.2.1, kernel, gfx1151, full VRAM pool
│   ├── 01-build-vllm.sh        # source-build vLLm @ pinned PR#51655 commit + amdsmi shim
│   ├── 02-fetch-model.sh       # hf download meta-models/Muse-Glimmer-30B (BF16)
│   ├── 03-serve-vllm.sh        # launch with adapted flags
│   ├── gguf-quickstart.sh      # llama.cpp-HIP + Muse-Glimmer-30B-GGUF easy path
│   └── benchmark.sh            # throughput + correctness validation
├── configs/
│   ├── vllm-gfx1151.env        # FLASH_ATTENTION_TRITON_AMD_ENABLE, etc.
│   └── serve-args.conf         # adapted vLLM serve arguments (single source of truth)
├── tests/
│   ├── test_env.py             # torch sees gfx1151 + correct VRAM pool
│   ├── test_smoke.py           # /v1/models + /v1/chat/completions round-trip
│   └── test_parsers.py         # muse_glimmer reasoning + ATEM tool-call parsing
└── patches/
    └── vllm-amdsmi-import.diff # import-amdsmi-before-torch shim, pinned to a commit
```

## 8. Pipeline 1 — vLLM (faithful, full-feature path)

```
00-check-env.sh    → assert ROCm 7.2.1 · kernel ≥6.16.9 · gfx1151 · full ~94 GB pool visible
01-build-vllm.sh   → git clone vllm@<PR#51655 commit> → apply amdsmi shim → pip install -e . --no-build-isolation
02-fetch-model.sh  → huggingface-cli download meta-models/Muse-Glimmer-30B   (BF16, ~59 GB)
03-serve-vllm.sh   → source configs/vllm-gfx1151.env && vllm serve …
```

**Core serve command** (adaptations are the point):

```bash
# configs/vllm-gfx1151.env
export FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE
export HF_HUB_OFFLINE=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
# VLLM_ROCM_USE_AITER intentionally NOT set — auto-disabled on gfx1151 (CDNA3+ gate)

vllm serve $MODEL_DIR/meta-models/Muse-Glimmer-30B \
  --served-model-name muse-glimmer \
  --tensor-parallel-size 1 \
  --dtype bfloat16 \
  --max-model-len 131072 \
  --attention-backend FLASH_ATTN \
  --enable-auto-tool-choice --tool-call-parser muse_glimmer \
  --reasoning-parser muse_glimmer \
  --generation-config auto \
  --gpu-memory-utilization 0.90
# explicit NON-flags: no --kv-cache-dtype fp8, no --enable-chunked-prefill,
# no --quantization, no --speculative-config
```

**Note:** the env block above is the *starting* configuration; like the tuning flags below, each var (`FLASH_ATTENTION_TRITON_AMD_ENABLE`, `VLLM_WORKER_MULTIPROC_METHOD`, `HF_HUB_OFFLINE`) is verified on hardware and frozen into `configs/vllm-gfx1151.env`.

**Tuning flags resolved by TDD during implementation** (written into `configs/serve-args.conf` once verified on hardware):
`--enforce-eager` (CUDA-graph capture is a common RDNA crash source → first-run eager; drop for throughput once stable), `--max-num-seqs`, fallback `--attention-backend TRITON_ATTN`. The passing minimal flag set becomes the pinned default.

## 9. Pipeline 2 — GGUF quick-start (accessible, no-compile-chat path)

```
gguf-quickstart.sh → build llama.cpp HIP (gfx1151, once) → fetch Muse-Glimmer-30B-GGUF → llama-server
```

```bash
cmake -B build -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx1151 -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
./build/bin/llama-server -m Muse-Glimmer-30B-Q4_K_M.gguf -ngl 999 -c 32768 --port 8080
```

**Documented caveats:**
- **Text-focused.** llama.cpp's VLM/vision support lags; this is the "chat in minutes" path, not multimodal/agentic. vLLM is the full-feature path.
- **No `muse_glimmer` parsers** — llama.cpp uses its own chat template; ATEM tool-calls / channel-scoped reasoning are not parsed.
- **Architecture-support risk:** depends on llama.cpp recognising the `muse_glimmer` GGUF (Meta ships calibrated GGUFs, so likely yes). Verified in implementation, with a fallback note if not.

## 10. Validation & benchmarking

- **Correctness:** `test_smoke.py` (model loads, `/v1/models`, chat round-trip), `test_parsers.py` (channel-scoped reasoning → `message.reasoning`; ATEM tool-calls → `message.tool_calls`), `test_env.py` (torch sees gfx1151 + full ~94 GB pool).
- **Throughput:** TTFT / TPOT / tok/s at concurrency {1, 4, 16} × context {2K-in/512-out, 32K-in/1K-out}; **vLLM BF16 vs GGUF Q4** head-to-head (educational contrast).
- **Memory:** peak VRAM (idle / loaded / under-load via `rocm-smi`), recorded with full env manifest into `docs/results/`.

## 11. Documentation structure

- `README.md` — TL;DR two-path quick-start + results table.
- `docs/adaptation.md` — the delta table (Section 5) with *why*; the educational centerpiece.
- `docs/strix-halo-setup.md` — prerequisites, UMA/VRAM carve-out, verify ~94 GB pool.
- `docs/troubleshooting.md` — every gotcha: symptom → cause → fix.
- `docs/results/` — benchmark logs, charts, env manifest.

## 12. Error handling philosophy

Every script **fails fast with a code-linked message** (e.g. `00-check-env.sh` exits non-zero with *"ROCm sees only 15.5 GB → see troubleshooting.md#uma-bug"*). Each workaround in code (amdsmi shim, no-chunked-prefill) cites its troubleshooting entry explaining why.

## 13. Testing & CI strategy

- Layered tests double as the vLLM-build verification (TDD order: test → build flags → pass).
- **GitHub Actions cannot run gfx1151**, so CI covers only hardware-free checks (shellcheck, pyproject/`uv.lock` validity, doc lint, markdown link-check). Hardware validation is the documented `docs/results/` + a local `verify` target. Stated plainly in the README.

## 14. Known risks & mitigations

| Risk | Mitigation |
|---|---|
| vLLM gfx1151 segfault (vLLM #37151) | pin known-good commit; `--enforce-eager`; document in troubleshooting |
| ViT attention 3.7× regression on gfx1151 (flash-attention #2392) | benchmark vision path separately; document; text path unaffected |
| TheRock nightly wheel drift | pin exact wheel version + hash in `pyproject.toml`; document |
| llama.cpp lacks `muse_glimmer` GGUF support | verify early; fallback = GGUF conversion note / vLLM path |
| UMA pool < expected | `00-check-env.sh` asserts; troubleshooting links kernel ≥6.16.9 fix |
| ROCm 7.2.1 is "community-supported" not "official" for gfx1151 | document; provide 7.14.0 alternative path in `docs/` |

## 15. Open items to resolve during implementation

These are **specific, actionable unknowns** to be nailed down in the build phase (not vague TODOs):

1. **Pin exact vLLM commit.** Resolve PR #51655 head ambiguity (branch `tiezhen/new-model-support` vs `xianbaoqian/vllm` fork). Pin a commit hash + include bug-fix `436be94` lineage in `01-build-vllm.sh`.
2. **Pin exact TheRock gfx1151 wheel** version string + hash in `pyproject.toml` `[tool.uv.sources]`/`[[tool.uv.index]]`.
3. **Confirm the `import amdsmi` shim** line/location against the pinned vLLM tree; materialize as `patches/vllm-amdsmi-import.diff`.
4. **Resolve tuning flags via TDD**: determine whether `--enforce-eager` is required and the best `--attention-backend` (`FLASH_ATTN` vs `TRITON_ATTN`) on this box; freeze the passing set in `configs/serve-args.conf`.
5. **Verify llama.cpp recognizes `muse_glimmer` GGUF**; if not, document the conversion/fallback.
6. **Confirm UMA pool**: verify ROCm sees ≈ the expected carve-out on this kernel; adjust setup doc.

## 16. References

- Upstream recipe PR: https://github.com/vllm-project/recipes/pull/776
- vLLM model-support PR: https://github.com/vllm-project/vllm/pull/51655
- Model: https://huggingface.co/meta-models/Muse-Glimmer-30B (config: `/raw/main/config.json`)
- Model blog: https://huggingface.co/blog/muse-glimmer
- AITER RDNA gate: https://github.com/vllm-project/vllm/issues/51136
- gfx1151 invalid-device-function: https://github.com/ROCm/ROCm/issues/4909
- UMA 15.5 GB bug: https://github.com/ROCm/ROCm/issues/5444
- Chunked-prefill hang on RDNA: https://github.com/vllm-project/vllm/issues/5013
- vLLM ROCm attention backends blog: https://vllm.ai/blog/2026-02-27-rocm-attention-backend
- Strix Halo vLLM from-source how-to: https://community.frame.work/t/how-to-compiling-vllm-from-source-on-strix-halo/77241
- Strix Halo vLLM benchmarks: https://kyuz0.github.io/amd-strix-halo-vllm-toolboxes/
- ROCm compatibility matrix: https://rocm.docs.amd.com/en/latest/compatibility/compatibility-matrix.html
- TheRock gfx1151 wheels: https://rocm.nightlies.amd.com/v2/gfx1151/
- AMD vLLM-on-ROCm guide: https://rocm.docs.amd.com/projects/ai-ecosystem/en/latest/inference/vllm.html
