# Handoff — Muse-Glimmer-30B-ROCm

**Status as of 2026-08-13: complete (master) + DFlash/vision matrix complete
(`feat/llamacpp-dflash-benchmark`).** Both inference paths are implemented,
validated on real gfx1151 hardware, benchmarked head-to-head, and committed to
`master`. The v1-deferred items — **DFlash speculative decoding** and **vision
via `--mmproj`** — are now validated on the llama.cpp path on branch
`feat/llamacpp-dflash-benchmark`, with a 3-study benchmark matrix
(Study 1 / 2 / 3) committed as immutable raw per-cell JSON
(`docs/results/matrix/cell-*.json`). **ROCm 7.14.0 remains a separate, gated
Part 2 plan** — see §8.

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

### DFlash + vision (new this branch) — one-line summary

| Result | Number | Where |
|---|---|---|
| DFlash speedup @ greedy batch 1 (17gb) | **2.20×** (10.48 → 23.03 tok/s), acceptance 0.233 | [Study 1](docs/results/benchmark.md#study-1--dflash-anchor-greedy-batch-1--meta-comparable) |
| DFlash speedup @ greedy batch 1 (dynamic) | **2.39×** (9.14 → 21.82 tok/s), acceptance 0.237 | Study 1 |
| vs Meta anchors | RTX 5090 3.1× / M5 Max 1.8× → **gfx1151 sits between** | Study 1 |
| Byte-equivalence (greedy spec-decode exactness) | **PASS** (both emit `391` for `17 × 23`) | [METHODOLOGY §9](docs/results/METHODOLOGY.md#9-byte-equivalence-greedy-spec-decode-exactness) |
| Vision via `--mmproj` | loads + answers; **+2–3 GiB VmPeak** vs text-only (≈ Meta's ~+2 GB) | [Study 3](docs/results/benchmark.md#study-3--vision-axis-temp-10---mmproj-test-image) |
| **⚠ c=16 + DFlash** | **pathological — do not use** (>1000× slower per-request) | [warning](docs/results/benchmark.md#c16--dflash-do-not-use) |

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
   **DFlash + mmproj GGUFs are Xet-backed** — fetch them **direct from
   huggingface.co with `HF_HUB_DISABLE_XET=1`** (the mirror can't proxy Xet's
   CAS). See [troubleshooting.md#dflash-mmproj-xet](docs/troubleshooting.md#dflash-mmproj-xet).

4. **Chunked prefill is fine.** vLLM V1 defaults it ON; the historical RDNA hang
   ([vllm-project/vllm#5013](https://github.com/vllm-project/vllm/issues/5013))
   did **not** reproduce with TRITON_ATTN. We leave the default (for throughput).

5. **llama.cpp concurrency needs `-np`.** `llama-server` ships with 4 slots and
   plateaus at ~22 tok/s. To scale to c=16, pass e.g. `-np 16 -c 16384`
   (1024 ctx/slot) → 102 tok/s. vLLM needs no such tuning (continuous batching).

6. **`rocm-smi` VRAM is misleading on Strix Halo.** It reports only the ~32 GiB
   *dedicated carve-out* (and ~1 GiB "used"). The real footprint is unified memory
   (vLLM sees an 80 GiB pool: 56.5 GiB weights + 13.6 GiB KV). llama.cpp's GPU
   buffers don't increment the carve-out counter either, and `/proc` `VmHWM`
   undercounts too (mmap + GPU offload). **Trust VmPeak** (~24–32 GiB on the
   matrix). See [troubleshooting.md#memory-footprint-apu](docs/troubleshooting.md#memory-footprint-apu)
   + [`docs/results/METHODOLOGY.md §5`](docs/results/METHODOLOGY.md#5-the-memory-methodology--trust-vmpeak-not-rocm-smi-or-vmhwm).

7. **Muse-Glimmer is a reasoning model.** It emits chain-of-thought in the
   `reasoning` channel *first*, then the answer in `content`. Tests/clients need
   enough `max_tokens` (≥~300) to finish reasoning and produce `content`; at
   `max_tokens=16` you get `finish_reason=length` and empty content.
   `reasoning_strength` (`low`/`medium`/`high`/`xhigh`, default `high`) controls
   thinking length; it **cannot** be switched off (`--reasoning off` is a no-op).
   See [troubleshooting.md#reasoning-length](docs/troubleshooting.md#reasoning-length).

8. **DFlash is a silent no-op without `--spec-type draft-dflash`.** `llama-server`'s
   `--spec-type` defaults to `none`; `-md dflash.gguf -ngld 99` alone loads the
   drafter but never drafts (1.0×, null acceptance). Always pass
   `--spec-type draft-dflash --spec-draft-n-max 16` (n_max=16 is the measured
   sweet spot). See [troubleshooting.md#dflash-silent-noop](docs/troubleshooting.md#dflash-silent-noop).

9. **⚠ DFlash @ c=16 is pathological — do not use.** >1000× slower per-request
   than the c=16 baseline (a 16×48 batch completed 0 of 768 tokens in 28 s;
   the dynamic REPS=5 cell was aborted after 5 h 16 m at 0.18 % acceptance).
   c=16 baseline is fine (31–34 tok/s); the pathology is DFlash-specific.
   DFlash is a win at c ≤ 4 (best at c=1: ~2.2×). Best-practice table +
   evidence: [`README.md`](README.md#best-practices--pitfalls--read-this-before-using-dflash-or-c16),
   [`docs/results/benchmark.md`](docs/results/benchmark.md#c16--dflash-do-not-use),
   [troubleshooting.md#dflash-c16-pathological](docs/troubleshooting.md#dflash-c16-pathological).

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
| Throughput (v1) | benchmark JSON in `docs/results/`; tables in `docs/results/benchmark.md` |
| **DFlash speedup @ greedy batch 1** | Study 1 cell JSONs: 17gb 2.20× (10.48→23.03), dynamic 2.39× (9.14→21.82); acceptance 0.233 / 0.237 |
| **DFlash byte-equivalence** | `scripts/check_dflash_equiv.sh` PASS — both emit `'391'` for `17 × 23` |
| **Throughput under load (Study 2)** | 12 cell JSONs (10 completed + 2 `pathological:true` c=16 DFlash non-completions) |
| **Vision via `--mmproj` (Study 3)** | 5 cell JSONs; mmproj loads, answers; +2–3 GiB VmPeak ≈ Meta's ~+2 GB |
| **llama-bench cross-check** | `matrix/llama-bench.json` tg128: 17gb 10.73 / dynamic 9.28 vs Study 1 baseline 10.48 / 9.14 (within 2.4 %) |

Full suite on `master`: **13 passed, 5 skipped** (4 server tests skip when no
server is running; 1 = shellcheck not installed locally — CI runs it).
On `feat/llamacpp-dflash-benchmark`: the GGUF-bench harness adds CI-safe tests
under `tests/test_gguf_bench.py` (config validation, JSON schema, renderer).

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

- **DFlash speculative decoding — DONE on llama.cpp, still blocked on vLLM.**
  The llama.cpp path is validated (2.20× / 2.39× at greedy batch 1; see
  [`docs/results/benchmark.md` Study 1](docs/results/benchmark.md#study-1--dflash-anchor-greedy-batch-1--meta-comparable)).
  The **vLLM path** is still blocked by the upstream registry bug
  (`DFlashMuseGlimmerAssistant`); the drafter GGUF is on disk if someone tries a
  patched fork. **Pitfall documented: do NOT combine DFlash with c=16 —
  pathological.** See [`README.md` best practices](README.md#best-practices--pitfalls--read-this-before-using-dflash-or-c16).
- **Vision via llama.cpp — DONE.** Study 3 (5 cells) confirms `--mmproj` loads +
  answers, with a memory delta matching Meta's envelope. See
  [`docs/results/benchmark.md` Study 3](docs/results/benchmark.md#study-3--vision-axis-temp-10---mmproj-test-image).
- **ROCm 7.14.0 path — separate, gated Part 2 plan.** gfx1151 is "officially"
  supported at 7.14.0; this project uses 7.2.1 (community-verified). 7.14.0 is
  documented as an alternative (`docs/strix-halo-setup.md`). The 7.2.1 matrix
  data is committed and immutable on `feat/llamacpp-dflash-benchmark`, so any
  7.14.0 work begins with a feasibility probe (can it install side-by-side on
  this host without removing 7.2.1?) and then re-runs the identical matrix with
  only ROCm differing. See spec §9 (P-D).
- **No git remote / PR** — the repo is local-only. `master` holds the v1
  deliverable; `feat/llamacpp-dflash-benchmark` holds the DFlash + vision matrix
  (this branch). If a GitHub remote is added, the `ci.yml` workflow will run the
  no-GPU tests on push.
- **GGUF live build is reproducible but slow to re-do** — `gguf-quickstart.sh`
  clones llama.cpp from GitHub (slow from this host) + ~15 min cmake build + ~16 GiB
  GGUF. The artifacts are already on disk, so this only matters on a fresh clone.
- **shellcheck** not installed locally (test skips; `ci.yml` installs it).

---

## 9. Commit history

On `master` (v1):

```
5884b11 docs: add handoff.md (project orientation for future work)
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

On `feat/llamacpp-dflash-benchmark` (DFlash + vision matrix; see
`git log master..HEAD`):

```
13fdac6 data(bench): complete Study 2 (incl. c=16) + Study 3 on ROCm 7.2.1
046c9ea data(bench): Study 3 vision cells (5/5) on ROCm 7.2.1
5585350 data(bench): Study 2 throughput-under-load cells (9/12; 3 c=16 pending)
6248e53 fix(bench): bump sock_read to 1200s for c=16 tail latency
9b37b4e fix(bench): lift aiohttp 5min default timeout for c=16 cells + render study3
290e596 fix(bench): check_dflash_equiv.sh must enable spec-decoding
4ebaa0d data(bench): Study 1 (Meta-comparable DFlash anchor) on ROCm 7.2.1
51c2149 fix(bench): engage DFlash spec decoding + capture acceptance from server log
4195f16 feat(bench): matrix driver (randomized cell order) + markdown render
9b1e4f2 fix(bench): review round 1 — scrape_metrics avg, include_usage test, cell robustness
49754c3 feat(bench): single-cell orchestrator (gguf-bench-cell.sh) + P0 validation
8fba282 feat(bench): streaming chat/completions client + TTFT/TPOT + prompt loop
45bc2fe fix(bench): enhance GGUF config parser + assertions
67f744a feat(bench): per-study config files (study1/2/3) + validation tests
eb084d3 fix(bench): render_matrix review findings - study2 test + study1 baseline filter + cleanup
8726ab8 data(bench): GGUF manifest (4 weights on disk)
f14d6c8 feat(bench): render_matrix JSON->markdown (TDD)
1ce437b feat(bench): capture_proc parsers (TDD) - spec acceptance + /metrics + /proc
4b271b3 feat(bench): bench_client pure metric functions (TDD)
<task-13 commit> docs(bench): METHODOLOGY + Study 1/2/3 results, c=16 warning, adaptation/troubleshooting updates
```

Spec (v1): `docs/superpowers/specs/2026-08-11-muse-glimmer-30b-rocm-design.md`.
Spec (v2 DFlash): `docs/superpowers/specs/2026-08-12-llamacpp-dflash-benchmark-design.md`.
Plan (v2): `docs/superpowers/plans/2026-08-12-llamacpp-dflash-benchmark.md`.

---

## 10. First things to check if something breaks

- **vLLM won't import** → someone ran `uv run` without `--no-sync`. Re-run
  `scripts/01-build-vllm.sh`. (See finding #2.)
- **Server crashes at startup with "FlashAttention version not detected"** → the
  backend got changed back to FLASH_ATTN. It must be `TRITON_ATTN`
  (`configs/serve-args.conf`). (Finding #1.)
- **Model download stalls at KB/s** → make sure `HF_ENDPOINT=https://hf-mirror.com`
  and the shards are fetched via `hf_parallel_get.py`, not stock `hf download`.
  (Finding #3.) For `dflash-kquant.gguf` / `mmproj-kquant.gguf` specifically →
  fetch direct from huggingface.co with `HF_HUB_DISABLE_XET=1` (Xet not proxied
  by the mirror).
- **DFlash shows no speedup** → you forgot `--spec-type draft-dflash
  --spec-draft-n-max 16`. `--spec-type` defaults to `none`; `-md …` alone is a
  silent no-op. (Finding #8.)
- **DFlash @ c=16 hangs forever** → that's the known pathology. Drop DFlash at
  c ≥ 8 and rerun; c=16 baseline is fine. (Finding #9.)
- **Footprint looks wrong (rocm-smi ~1 GiB)** → trust VmPeak, not rocm-smi/VmHWM.
  (Finding #6.)
- **`pytest` shows server-test failures** → start a server (the conftest skips
  them only when nothing is on :8000).
