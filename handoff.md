# Handoff — Muse-Glimmer-30B-ROCm

**Status as of 2026-08-12: complete.** Both inference paths are implemented,
validated on real gfx1151 hardware, benchmarked head-to-head, and committed to
`master`. All 10 tasks in the implementation plan are done.

This document is the fast-path orientation for anyone picking the project up.
Authoritative details live in the linked docs; this is the map.

---

## 1. What this project is

Run Meta's **Muse-Glimmer-30B** (dense 29.6B vision-language model, `muse_glimmer`
arch, Apache 2.0, not gated) on **gfx1151** — AMD Ryzen AI MAX+ PRO 395 / Radeon
8060S "Strix Halo", RDNA 3.5, 94 GiB unified LPDDR5X, ROCm 7.2.1, kernel 6.17 —
via two paths:

1. **vLLM (BF16)** — the full-feature path (vision, native reasoning + tool-call
   parsing, 128K context, continuous batching).
2. **llama.cpp (Q4 K-quant GGUF)** — the fast/light path (smaller, quicker,
   trivial install).

The educational core is the **CDNA → RDNA adaptation** (the MI300X recipe retargeted
to gfx1151); every delta is documented in [`docs/adaptation.md`](docs/adaptation.md).

---

## 2. Headline result — vLLM vs llama.cpp on gfx1151

Same harness, `/v1/completions`, 512 output tokens, GPU (HIP):

| Concurrency | vLLM — BF16 (tok/s) | llama.cpp — Q4 kquant (tok/s) | Q4 speedup |
|---|---|---|---|
| 1  | 4.2   | **10.5**  | 2.5× |
| 4  | 14.0  | **21.3**  | 1.5× |
| 16 | 40.8  | **102.0** | 2.5× |

**llama.cpp Q4 is faster at every concurrency** (~4× less memory traffic per
token; decode is bandwidth-bound on the APU). **vLLM's edge is features**, not
speed: native `muse_glimmer` reasoning + ATEM tool parsers, vision/multimodal,
128K context, automatic batching. Both produce correct inference (e.g.
*"17 × 24 → 408"*). Full comparison: [`docs/results/benchmark.md`](docs/results/benchmark.md).

---

## 3. How to run

```bash
cd /home/amd/Desktop/muse-rocm
export PATH="$HOME/.local/bin:$PATH"      # uv lives here

# --- vLLM (BF16) path: one-time setup, then serve ---
uv sync                                  # TheRock gfx1151 torch + deps (Python 3.12)
bash scripts/00-check-env.sh             # ROCm 7.2.x · kernel ≥6.16.9 · gfx1151 · ≥60 GiB
bash scripts/01-build-vllm.sh            # source-build vLLM (PR #51655) — ~1 h HIP compile
bash scripts/02-fetch-model.sh           # ~55 GiB BF16 weights via hf-mirror (parallel)
bash scripts/03-serve-vllm.sh            # OpenAI server on :8000 (~8 min startup)

# --- llama.cpp (GGUF) path: one script ---
bash scripts/gguf-quickstart.sh          # builds llama.cpp + fetches Q4 GGUF + serves on :8080

# --- benchmark either ---
BASE=http://127.0.0.1:8000 bash scripts/benchmark.sh   # vLLM
BASE=http://127.0.0.1:8080 bash scripts/benchmark.sh   # llama.cpp
```

**The model weights and llama.cpp build are already on disk** (gitignored): the
BF16 weights (~56 GiB) and the kquant GGUF (15.6 GiB) are in `models/`, and
`third_party/llama.cpp/build/bin/llama-server` is built. So a fresh run skips the
downloads/builds. Only `uv sync` + the vLLM source build need repeating if `.venv`
is removed.

---

## 4. The non-obvious findings (read these before changing anything)

These were discovered by running on hardware, not predicted. Each is documented
in `docs/troubleshooting.md`; the short version:

1. **`--attention-backend TRITON_ATTN`, not `FLASH_ATTN`.** FLASH_ATTN crashes at
   the profiling forward pass with `AssertionError: FlashAttention version not
   detected` — there's no flash-attn codegen for gfx1151. TRITON_ATTN (Triton
   kernels via the ROCm-patched Triton) works. This was the single change that
   took vLLM from crash-on-startup to serving. **Never set `VLLM_ROCM_USE_AITER`**
   (CDNA3+/RDNA4-only) and **don't try FP8** (no RDNA 3.5 path).

2. **Every `uv run` MUST pass `--no-sync`.** vLLM is source-installed *editable*
   and is intentionally not in `uv.lock`. A bare `uv run` re-syncs and **wipes
   the editable vLLM** (`ModuleNotFoundError: vllm`). All scripts/tests already
   pass `--no-sync`. If vLLM vanishes, re-run `scripts/01-build-vllm.sh`.

3. **Model fetch on a region-locked/slow link.** `huggingface.co` is slow/blocked
   here; `scripts/02-fetch-model.sh` defaults to `HF_ENDPOINT=https://hf-mirror.com`.
   But the mirror only proxies metadata — LFS blobs 302-redirect to the *same*
   signed CloudFront URL either way. The stock `hf download` is single-stream
   (~0.2 MiB/s/conn → days). `scripts/hf_parallel_get.py` opens 24 parallel HTTP
   range connections against the signed URL, re-resolves it hourly, and resumes
   per-chunk → ~5–16 MiB/s (~3 h for the weights). `HF_XET_HIGH_PERFORMANCE`
   401s through the mirror (CAS not proxied); `hf_transfer` is a no-op on
   huggingface_hub ≥1.27. Set `USE_HF_DOWNLOAD=1` for the stock tool.

4. **Chunked prefill is fine.** vLLM V1 defaults it ON; the historical RDNA hang
   ([vllm-project/vllm#5013](https://github.com/vllm-project/vllm/issues/5013))
   did **not** reproduce with TRITON_ATTN. We leave the default (for throughput).

5. **llama.cpp concurrency needs `-np`.** `llama-server` ships with 4 slots and
   plateaus at ~22 tok/s. To scale to c=16, pass e.g. `-np 16 -c 16384`
   (1024 ctx/slot) → 102 tok/s. vLLM needs no such tuning (continuous batching).

6. **`rocm-smi` VRAM is misleading on Strix Halo.** It reports only the ~32 GiB
   *dedicated carve-out* (and ~1 GiB "used"). The real footprint is unified memory
   (vLLM sees an 80 GiB pool: 56.5 GiB weights + 13.6 GiB KV). llama.cpp's GPU
   buffers don't increment the carve-out counter either. Trust vLLM's startup
   accounting, not `rocm-smi --showmeminfo vram`.

7. **Muse-Glimmer is a reasoning model.** It emits chain-of-thought in the
   `reasoning` channel *first*, then the answer in `content`. Tests/clients need
   enough `max_tokens` (≥~300) to finish reasoning and produce `content`; at
   `max_tokens=16` you get `finish_reason=length` and empty content.

---

## 5. Validated on hardware (evidence)

| Claim | Evidence |
|---|---|
| Env ready | `scripts/00-check-env.sh` exits 0; `test_env.py` / `test_env_torch.py` green |
| vLLM serves BF16 on gfx1151 | server boots; `Resolved architecture: MuseGlimmerForConditionalGeneration`; `Using TRITON_ATTN backend` |
| Weights fetched via mirror | both shards byte-exact vs HF (49.95 + 8.94 GiB) |
| `muse_glimmer` reasoning + ATEM tool parsers | live `tests/test_parsers.py` green |
| Chat round-trip | live `tests/test_smoke.py` green |
| llama.cpp runs the GGUF on GPU | correct output; ~8% CPU across 32 cores during gen (GPU-bound) |
| Throughput | benchmark JSON in `docs/results/`; tables in `docs/results/benchmark.md` |

Full suite on `master`: **13 passed, 5 skipped** (4 server tests skip when no
server is running; 1 = shellcheck not installed locally — CI runs it).

---

## 6. Repository layout

```
muse-rocm/
├── README.md                     # public face: two-path quick-start + results table
├── handoff.md                    # this file
├── pyproject.toml                # uv project: TheRock gfx1151 torch, uv index, pytest cfg
├── .python-version               # 3.12
├── configs/
│   ├── serve-args.conf           # vLLM serve flags (TRITON_ATTN) — single source of truth
│   └── vllm-gfx1151.env          # FLASH_ATTENTION_TRITON_AMD_ENABLE, HF_HUB_OFFLINE, …
├── scripts/
│   ├── 00-check-env.sh           # assert ROCm/kernel/gfx1151/VRAM
│   ├── 01-build-vllm.sh          # source-build vLLM @ pinned commit + amdsmi shim
│   ├── 02-fetch-model.sh         # two-phase fetch: hf download (small) + hf_parallel_get (shards)
│   ├── hf_parallel_get.py        # 24-conn resumable range downloader (the speed fix)
│   ├── 03-serve-vllm.sh          # launch vLLM (--no-sync)
│   ├── gguf-quickstart.sh        # build llama.cpp + fetch GGUF + serve
│   ├── bench_client.py           # async throughput client (aiohttp)
│   └── benchmark.sh              # sweep c=1/4/16, snapshot VRAM, write JSON
├── tests/                        # gpu / server / CI-safe markers (see conftest.py)
├── patches/
│   ├── vllm-amdsmi-import.diff   # import-amdsmi-before-torch shim
│   └── vllm-torch210-compat.diff
├── docs/
│   ├── adaptation.md             # MI300X→gfx1151 delta (the centerpiece)
│   ├── strix-halo-setup.md       # host prerequisites
│   ├── troubleshooting.md        # symptom → cause → fix (incl. the 7 findings above)
│   └── results/benchmark.md      # full vLLM-vs-llama.cpp comparison
├── models/                       # gitignored: Muse-Glimmer-30B/ (BF16) + kquant GGUF
└── third_party/                  # gitignored: vllm/ (source build), llama.cpp/ (build)
```

### Test markers
- `@pytest.mark.gpu` — needs the gfx1151 GPU (run locally: `uv run --no-sync pytest -m gpu`).
- `@pytest.mark.server` — needs a running server; `conftest.py` auto-skips them when
  nothing is listening on :8000, so a bare `pytest` is green out-of-the-box.
- CI (`.github/workflows/ci.yml`) runs `-m "not gpu and not server"` only — **CI has
  no gfx1151.**

---

## 7. Pinned stack (validated)

| Component | Version / source |
|---|---|
| vLLM | `0.1.dev1+g606a12cd7` — `xianbaoqian/vllm` @ `606a12cd` (branch `tiezhen/new-model-support`), source-built editable |
| PyTorch | `2.10.0+rocm7.13.0a20260513` (TheRock gfx1151 nightly) |
| ROCm | 7.2.1 |
| llama.cpp | v1 (`0b1bad1`), HIP build, `-DAMDGPU_TARGETS=gfx1151` |
| GGUF | `meta-models/Muse-Glimmer-30B-GGUF` / `muse-glimmer-30B-kquant-17gb.gguf` (15.6 GiB) |

---

## 8. What is NOT done / future work

These are explicit non-goals or deferred items, not bugs:

- **DFlash speculative decoding** — upstream registry bug (`DFlashMuseGlimmer…`);
  needs a patched fork. The drafter GGUF (`dflash-kquant.gguf`) is available if
  someone tries. Documented in `docs/adaptation.md`.
- **Vision via llama.cpp** — `mmproj-kquant.gguf` exists and `gguf-quickstart.sh`
  supports `WITH_MMProj=1`, but the live vision path was not benchmarked.
  vLLM's vision path is native and untested-for-throughput too (only text here).
- **ROCm 7.14.0 path** — gfx1151 is "officially" supported at 7.14.0; this project
  uses 7.2.1 (community-verified). 7.14.0 is documented as an alternative
  (`docs/strix-halo-setup.md`), not tested.
- **No git remote / PR** — the repo is local-only (branch: `master`). If a GitHub
  remote is added, the `ci.yml` workflow will run the no-GPU tests on push.
- **GGUF live build is reproducible but slow to re-do** — `gguf-quickstart.sh`
  clones llama.cpp from GitHub (slow from this host) + ~15 min cmake build + ~16 GiB
  GGUF. The artifacts are already on disk, so this only matters on a fresh clone.
- **shellcheck** not installed locally (test skips; `ci.yml` installs it).

---

## 9. Commit history (on `master`)

```
e57cec8 feat: validate llama.cpp GGUF path + vLLM-vs-llama.cpp comparison
36df79c docs(plan): note the uv --no-sync global constraint
54dbf5c docs: validated gfx1151 benchmark results + env manifest
c1a88d4 feat: validate gfx1151 serve (TRITON_ATTN) + live smoke/parser tests
c7f176b test: skip server-marked tests when no server is listening
3ddd7b0 feat: throughput + VRAM benchmark harness
bb177b3 feat: GGUF llama.cpp quick-start path for gfx1151
7bfe509 docs: README quick-start, CDNA->RDNA adaptation, setup, troubleshooting
bbd415f ci: no-GPU script/config lint tests + workflow
7fb782d feat: gfx1151-adapted vLLM serve config + launch script
398f42a feat: model fetch via hf-mirror with parallel range downloader
8ced6fd feat: source-build vLLM (PR #51655) for gfx1151 with amdsmi shim
315cc7c feat: environment verification script for gfx1151/ROCm 7.2.1
5cf4a8b feat: uv project scaffolding with TheRock gfx1151 torch
```

Spec: `docs/superpowers/specs/2026-08-11-muse-glimmer-30b-rocm-design.md`.
Plan: `docs/superpowers/plans/2026-08-11-muse-glimmer-30b-rocm.md`.

---

## 10. First things to check if something breaks

- **vLLM won't import** → someone ran `uv run` without `--no-sync`. Re-run
  `scripts/01-build-vllm.sh`. (See finding #2.)
- **Server crashes at startup with "FlashAttention version not detected"** → the
  backend got changed back to FLASH_ATTN. It must be `TRITON_ATTN`
  (`configs/serve-args.conf`). (Finding #1.)
- **Model download stalls at KB/s** → make sure `HF_ENDPOINT=https://hf-mirror.com`
  and the shards are fetched via `hf_parallel_get.py`, not stock `hf download`.
  (Finding #3.)
- **`pytest` shows server-test failures** → start a server (the conftest skips
  them only when nothing is on :8000).
