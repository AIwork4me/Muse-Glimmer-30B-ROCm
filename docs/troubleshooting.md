# Troubleshooting

Every gotcha we hit, as **symptom → cause → fix**. Skim the left column.

| Symptom | Jump |
|---|---|
| ROCm sees only ~15.5 GB of memory | [#uma-bug](#uma-bug) |
| `AITER not found` / falls back to emulation | [#aiter](#aiter) |
| FP8 model errors / `invalid device function` | [#fp8](#fp8) / [#invalid-device-function](#invalid-device-function) |
| Server hangs under load | [#chunked-prefill](#chunked-prefill) |
| Import error / crash at vLLM startup | [#amdsmi](#amdsmi) |
| `hf download` stalls at KB/s or never finishes | [#model-fetch-slow](#model-fetch-slow) |
| `uv run` wipes vLLM (`ModuleNotFoundError: vllm`) | [#no-sync](#no-sync) |

---

## uma-bug

**Symptom:** `rocm-smi` / `rocminfo` reports only ~15.5 GB; vLLM OOMs on a model
that should fit; `00-check-env.sh` fails the VRAM check.

**Cause:** Kernel < 6.16.9 mishandles the KFD/HSA unified-memory accounting, so
ROCm only sees a sliver of the 94 GiB pool ([ROCm/ROCm#5444][uma]).

**Fix:** Upgrade to kernel **≥ 6.16.9** (this host runs 6.17). No ROCm change
recovers the missing memory on an old kernel.

## aiter

**Symptom:** logs show `AITER not found`, `falling back to emulation`, or kernels
silently no-op; throughput is terrible.

**Cause:** AITER is hard-gated to **CDNA3+ / RDNA4** via `get_cdna_version() > 2`
([vllm-project/vllm#51136][aiter]). gfx1151 is RDNA 3.5, so AITER is absent.
Setting `VLLM_ROCM_USE_AITER=1` only forces broken emulation.

**Fix:** Do **not** set `VLLM_ROCM_USE_AITER`. Use `--attention-backend
FLASH_ATTN` with `FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE` (the Triton-on-AMD
path); `TRITON_ATTN` is the fallback. This is encoded in
`configs/serve-args.conf` + `configs/vllm-gfx1151.env` and asserted by
`tests/test_serve_args.py`.

## fp8

**Symptom:** FP8 checkpoint or `--kv-cache-dtype fp8` errors out, or you get
`invalid device function` mid-run.

**Cause:** RDNA 3.5 has no usable vLLM FP8 path — the FP8 matrix units are
RDNA4/CDNA3+.

**Fix:** Run **BF16**. This project ships BF16 only; do not pass
`--quantization fp8` or `--kv-cache-dtype fp8`.

## invalid-device-function

**Symptom:** `invalid device function` / `no kernel image is available for
execution on the device` the first time a HIP kernel runs.

**Cause:** A prebuilt ROCm docker/wheel was codegen'd for CDNA targets, not
gfx1151 ([ROCm/ROCm#4909][invdev]).

**Fix:** Use the **source-built vLLM** from `scripts/01-build-vllm.sh`
(`PYTORCH_ROCM_ARCH=gfx1151`) and the **TheRock gfx1151** torch wheel. Do not use
`vllm/vllm-openai-rocm:nightly`.

## chunked-prefill

**Symptom:** the server starts but hangs / stalls under concurrent load.

**Cause:** chunked prefill hangs on RDNA ([vllm-project/vllm#5013][chunked]).

**Fix:** Leave chunked prefill **off** (the default once you do not pass
`--enable-chunked-prefill`). Its absence is asserted by the banned-flag test.

## amdsmi

**Symptom:** vLLM crashes at import / startup with an `amdsmi`-related error on
gfx1151.

**Cause:** `amdsmi` must be imported **before** torch on this APU.

**Fix:** Applied automatically by `scripts/01-build-vllm.sh`, which inserts
`import amdsmi` at the top of vLLM's `__init__.py`; the change is captured in
`patches/vllm-amdsmi-import.diff` for the pinned commit. If you build vLLM by
hand, apply that patch.

## model-fetch-slow

**Symptom:** `hf download` / `huggingface-cli download` of the ~55 GiB weights
stalls at a few KB/s or never completes; or it dies with
`HTTP 401 Unauthorized` from `cas-server.xethub.hf.co`.

**Cause:** three compounding issues, in rough order of likelihood on a
region-locked or slow link:

1. **huggingface.co is slow/blocked from your host.** Set
   `HF_ENDPOINT=https://hf-mirror.com` (or another mirror). The project's
   `scripts/02-fetch-model.sh` defaults to it.
2. **The mirror does not speed up the weights.** hf-mirror.com proxies the HF API
   and `/resolve` metadata, but LFS blobs 302-redirect to the **same signed
   CloudFront URL** (`us.aws.cdn.hf.co`) either way. So for the *shards* the
   mirror buys you nothing — and on a link that sustains only ~0.2 MiB/s **per
   connection**, the stock single-stream `hf download` takes days.
3. **The Xet fast-path does not work through a mirror.** `HF_XET_HIGH_PERFORMANCE=1`
   makes the native xet client talk directly to `cas-server.xethub.hf.co`, which
   the mirror cannot proxy and which 401s without direct HF access. And
   `HF_HUB_ENABLE_HF_TRANSFER` is a **no-op** on huggingface_hub ≥ 1.27 (it
   prints a deprecation warning pointing at the xet flag).

**Fix:** the project's two-phase fetch (`scripts/02-fetch-model.sh`) uses
`scripts/hf_parallel_get.py` for the two shards — an N-way **parallel HTTP range
downloader** that opens 24 connections against the signed CloudFront URL,
re-resolves the URL as it expires (~hourly), and resumes per-chunk via
`<file>.parts.json`. On this link that takes ~0.2 MiB/s × 24 ≈ **5 MiB/s**
(≈16× the stock tool), finishing in a few hours instead of days. Requirements:
`curl` (always present), no sudo, no GitHub. Override the connection count with
`NCONNS=…`. To fall back to the stock single-stream tool, `USE_HF_DOWNLOAD=1`.

## no-sync

**Symptom:** after running `uv run <something>` (without `--no-sync`), vLLM is
suddenly gone — `ModuleNotFoundError: No module named 'vllm'` — or it has been
replaced by a PyPI wheel that lacks `muse_glimmer`.

**Cause:** vLLM is **source-installed editable** (`scripts/01-build-vllm.sh`,
`--no-build-isolation`) and is intentionally **not** in `uv.lock`. A bare `uv
run` triggers `uv sync`, which reinstalls the env from the lock and **drops the
editable vLLM**.

**Fix:** every `uv run` in this project passes `--no-sync` (see the serve/test
scripts). Never drop it. If you wiped vLLM, re-run `scripts/01-build-vllm.sh`.

[uma]: https://github.com/ROCm/ROCm/issues/5444
[aiter]: https://github.com/vllm-project/vllm/issues/51136
[invdev]: https://github.com/ROCm/ROCm/issues/4909
[chunked]: https://github.com/vllm-project/vllm/issues/5013
