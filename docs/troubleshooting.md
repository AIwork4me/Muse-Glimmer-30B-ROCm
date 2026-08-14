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
| DFlash loads but no speedup (`-md` alone) | [#dflash-silent-noop](#dflash-silent-noop) |
| DFlash @ c=16 hangs / never finishes | [#dflash-c16-pathological](#dflash-c16-pathological) |
| `[spec] failed to measure draft model memory` warning | [#dflash-mem-warning](#dflash-mem-warning) |
| `dflash-kquant.gguf` / `mmproj-kquant.gguf` download fails | [#dflash-mmproj-xet](#dflash-mmproj-xet) |
| rocm-smi shows ~1 GiB but model is 16+ GiB | [#memory-footprint-apu](#memory-footprint-apu) |
| `finish_reason: length` / empty `content` | [#reasoning-length](#reasoning-length) |
| gguf-quickstart refuses: llama.cpp has tracked changes | [#dirty-llama-cpp-checkout](#dirty-llama-cpp-checkout) |
| `required command not found: git/cmake/curl/python3` | [#missing-tool](#missing-tool) |

---

## uma-bug

**Symptom:** `rocm-smi` / `rocminfo` reports only ~15.5 GB; vLLM OOMs on a model
that should fit; `00-check-env.sh` fails the VRAM check.

**Cause observed on the project host:** kernels below 6.16.9 lacked the
KFD/HSA behavior needed by this 94 GiB Strix Halo setup, so ROCm exposed only a
small part of the pool ([ROCm/ROCm#5444][uma]).

**Project-reference fix:** use kernel **≥ 6.16.9** (the recorded hosts run
6.17). For other installations, follow AMD's [distribution-specific RDNA3.5
kernel requirements](https://rocm.docs.amd.com/en/latest/reference/system-optimization/rdna3-5.html);
6.16.9 is not asserted as a universal ROCm 7.14 floor.

## aiter

**Symptom:** logs show `AITER not found`, `falling back to emulation`, or kernels
silently no-op; throughput is terrible.

**Cause:** AITER is hard-gated to **CDNA3+ / RDNA4** via `get_cdna_version() > 2`
([vllm-project/vllm#51136][aiter]). gfx1151 is RDNA 3.5, so AITER is absent.
Setting `VLLM_ROCM_USE_AITER=1` only forces broken emulation.

**Fix:** Do **not** set `VLLM_ROCM_USE_AITER`, and use **`--attention-backend
TRITON_ATTN`** (with `FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE`). `FLASH_ATTN` does
*not* work on gfx1151 either: this build logs `Using FlashAttention version None`
and the first forward pass asserts `FlashAttention version not detected` (no
flash-attn library codegen for RDNA 3.5). `TRITON_ATTN` (Triton kernels) is the
validated backend. This is encoded in
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

**Symptom:** an older or modified stack starts but hangs / stalls under
concurrent prefill.

**Cause:** earlier RDNA reports implicated chunked prefill
([vllm-project/vllm#5013][chunked]). This was an important investigation lead,
not a universal RDNA constraint.

**Fix:** the pinned vLLM V1 + `TRITON_ATTN` reference was validated with its
default-on chunked prefill. Do not disable it preemptively. If a newer stack
stalls, record the exact commit, backend, prompt length, concurrency, and logs,
then compare default-on and explicitly disabled behavior as a diagnostic. Treat
a disable flag as a stack-specific workaround, not part of the validated
reference.

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

1. **huggingface.co is slow/blocked from your host.** The project defaults to
   the official endpoint. Set `HF_ENDPOINT=https://hf-mirror.com` (or another
   endpoint you trust) as an optional regional fallback; the script prints the
   selected endpoint.
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

## dirty-llama-cpp-checkout

**Symptom:** `scripts/gguf-quickstart.sh` (or `scripts/quickstart.sh`, which
execs it) stops before building with:

```
ERROR: third_party/llama.cpp has uncommitted tracked changes;
```

**Cause:** the quickstart never discards your work. Before it switches
llama.cpp commits — or reuses an already-checked-out one — it runs the
dirty-tree guard and refuses while tracked files are modified: local patches,
a hand-applied fix, a stray edit. The cold-start false positive that used to
fire on *every* fresh clone (an index-less `--no-checkout` clone misread as
"all 3419 files deleted") is fixed; if you see this error now, the checkout
really is dirty. Confirm with:

```bash
git -C third_party/llama.cpp status --porcelain
```

**Fix:** keep your work or discard it, then rerun. To keep it:

```bash
git -C third_party/llama.cpp stash        # restore later: git stash pop
bash scripts/gguf-quickstart.sh
```

To discard the changes and take the selected commit:

```bash
git -C third_party/llama.cpp checkout -- .
bash scripts/gguf-quickstart.sh
```

The refusal itself prints the same `git status --porcelain` excerpt (first 10
lines) and these recovery commands, so you normally do not need this page.

## missing-tool

**Symptom:** `install-rocm-7.14.sh`, `00-check-env.sh`, or
`gguf-quickstart.sh` stops with

```
ERROR: required command not found: <tool>      # installer, quickstart
FAIL: required command not found: <tool>       # environment checker
```

for one of `git`, `cmake`, `curl`, or `python3`.

**Cause:** the default GGUF path needs those four host tools on `PATH` before
anything else: the installer guards `python3` and `curl` before it reads its
manifest, the checker itemizes all four (so its OK verdict means the
quickstart will not die on a missing command), and the quickstart checks all
four after resolving the ROCm toolchain.

**Fix:** install the missing tool for your distro — the same one-liners the
failure text prints:

```bash
sudo apt-get install git cmake curl python3   # Debian/Ubuntu
sudo dnf install git cmake curl python3       # Fedora/RHEL
sudo pacman -S git cmake curl python3         # Arch
```

---

## DFlash / llama.cpp benchmark gotchas

These come from the DFlash + full benchmark matrix work
([`docs/results/METHODOLOGY.md`](results/METHODOLOGY.md),
[`docs/results/benchmark.md`](results/benchmark.md)). The matrix was measured on
the llama.cpp path; vLLM still has DFlash off (registry bug).

## dflash-byte-equivalence

**Symptom:** under greedy decoding (`temp=0 seed=0`), DFlash output differs from
the baseline (e.g., different tokens for the same prompt).

**Cause:** spec-decode must produce **byte-identical** output to the baseline
when both use greedy sampling. A mismatch indicates a bug in the DFlash
implementation or the harness.

**Fix:** run `scripts/check_dflash_equiv.sh`. The current harness compares
baseline and actual DFlash canonical response messages across all six Study 1 prompts plus the
original `17 × 23` smoke prompt, and rejects a DFlash server with zero draft
activity. The recorded historical result covers the arithmetic smoke prompt
(`391`); the expanded corpus must be run before claiming a 6/6 pass. See
[`docs/results/benchmark.md` — Study 1](results/benchmark.md#study-1-dflash-anchor).
If you see a mismatch, preserve both responses and server logs when filing a
bug.

## dflash-silent-noop

**Symptom:** you launch `llama-server` with `-md models/dflash-kquant.gguf -ngld
99` and the throughput is **identical to the no-draft baseline** (no speedup,
draft acceptance is null/zero). The draft model loaded fine but did nothing.

**Cause:** `llama-server`'s `--spec-type` **defaults to `none`**. `-md … -ngld
99` loads the draft model into memory but never engages the speculative-decoding
loop, so the server runs as a normal single-model server. This is a *silent*
no-op — no warning is logged.

**Fix:** always pass `--spec-type draft-dflash --spec-draft-n-max 15` alongside
`-md`. The full DFlash invocation is:

```
-m models/<weight>.gguf -ngl 999 ... \
  -md models/dflash-kquant.gguf -ngld 99 \
  --spec-type draft-dflash --spec-draft-n-max 15
```

`--spec-draft-n-max 15` is the measured sweet spot. Upstream DFlash drafts at
most `block_size - 1` = 15 tokens per round and silently clamps any higher
request down to 15 with a warning line at every server start — so request 15
directly. A pre-matrix sweep gave n_max `3 / 8 / 16 / 32` →
**1.14× / 1.51× / 1.60× / 1.60×**; the curve is flat past the elbow (the 16
and 32 cells effectively ran at 15 via the clamp). With this engaged, the
17gb weight delivers **2.20×** (23.03 vs 10.48 tok/s) at greedy batch 1. Verification: the cell JSON's `acceptance` block is
non-null and the rate is ~0.23. If you see 1.0× with null acceptance, you have
fallen into this trap.

## dflash-c16-pathological

> **⚠ Headline pitfall. Do NOT combine DFlash with `-np 16` (high concurrency).
> It is pathologically slow — >1000× slower per-request than the c=16
> baseline.** This is documented prominently in
> [`README.md`](../README.md#known-good-and-known-bad),
> [`docs/results/benchmark.md`](results/benchmark.md#c16-dflash-do-not-use), and
> [`docs/results/METHODOLOGY.md`](results/METHODOLOGY.md#c16-dflash-pathology).

**Symptom:** a DFlash cell at `-np 16` hangs or appears to make no progress.
The dynamic c=16 DFlash REPS=5 cell was **aborted after 5 h 16 m** with no
completion; a 17gb c=16 DFlash probe (16 concurrent × 48 tokens) completed
**0 of 768 tokens in 27.7 s**.

**Cause (verified):** at `-np 16` the drafter fires for all 16 slots
simultaneously, generating an enormous draft volume (in the dynamic run:
**3,270,000 draft tokens**) that is almost entirely rejected (**6,060 accepted
= 0.18 %**). The full generate+verify compute for *all those rejected drafts* is
paid in full. The draft model's predictions diverge badly from the target under
batched concurrent load, so spec-decode goes into reverse — it costs more than
it saves.

**Fix:** at `c ≥ 8` (especially `c = 16`), **drop DFlash** (just omit `-md` and
the `--spec-*` flags) and run the baseline. c=16 baselines are healthy:
**17gb 34.5 tok/s, dynamic 31.0 tok/s aggregate.** DFlash is a clear win only at
`c ≤ 4` (best at `c = 1`: ~2.2×). The full best-practice table is in
[`README.md`](../README.md#known-good-and-known-bad).

> c=16 itself is fine — the pathology is DFlash-specific. Both c=16 DFlash cells
> are recorded as `pathological: true` evidence-based non-completions in
> `docs/results/matrix/` (see `cell-study2-{17gb,dynamic}-np16-df1-vis0.json`),
> not as missing data.

## dflash-mem-warning

**Symptom:** at `llama-server` startup with `-md …`, this is logged:

```
[spec] failed to measure draft model memory
```

**Cause:** harmless. The drafter's GPU memory footprint query isn't implemented
in this build path; the drafter still loads and drafts correctly.

**Fix:** none needed — ignore it. The drafter's memory cost shows up in the
process VmPeak (~+2.5–3 GiB vs baseline), which is the footprint we report. See
[`docs/results/METHODOLOGY.md §5`](results/METHODOLOGY.md#memory-methodology).

## dflash-mmproj-xet

**Symptom:** `hf download` (or `huggingface-cli download`) of
`dflash-kquant.gguf` or `mmproj-kquant.gguf` fails through the `hf-mirror.com`
mirror with one of:

- `Distant resource does not seem to be on huggingface.co`
- `HTTP 401 Unauthorized` from `cas-server.xethub.hf.co`
- `hf_parallel_get.py` failing its Content-Range probe on a
  `us.aws.cdn.hf.co/xet-bridge` redirect

**Cause:** these two artifacts are **Xet-backed** (deduplicated via Xet's CAS).
`hf-mirror.com` proxies the HF API and `/resolve` metadata, but it does **not**
proxy Xet's CAS — so any Xet client through the mirror either 401s (native xet)
or fails the redirect (parallel range downloader). The large text shards are
*not* Xet-backed, which is why `hf_parallel_get.py` works for them.

**Fix:** fetch these two **direct from `huggingface.co`**, with Xet disabled so
the client uses classic LFS:

```bash
HF_HUB_DISABLE_XET=1 HF_ENDPOINT=https://huggingface.co \
  hf download meta-models/Muse-Glimmer-30B-GGUF \
    dflash-kquant.gguf mmproj-kquant.gguf \
  --local-dir models
```

(The big text shards use the selected endpoint plus the parallel range
downloader; the official endpoint remains the default. See
[model-fetch-slow](#model-fetch-slow).) Manifest of all four artifacts:
[`docs/results/matrix/gguf-manifest.md`](results/matrix/gguf-manifest.md).

## memory-footprint-apu

**Symptom:** `rocm-smi --showmeminfo vram` reports ~1 GiB used and ~32 GiB total
for a 16+ GiB model; `/proc/<pid>/status` `VmHWM` is 1–10 GiB for the same
process. Both look wrong.

**Cause:** on Strix Halo (a unified-memory APU), `rocm-smi` VRAM reports only the
~32 GiB **dedicated carve-out**, and the carve-out counter only ticks for buffers
allocated through that specific path. The mmap'd GGUF and the GPU-offloaded
unified-host-visible buffers don't increment it. `VmHWM` undercounts because
mmap'd file pages are paged in/out by the kernel and many GPU-offloaded pages
don't show as resident.

**Fix:** use **`VmPeak`** from `/proc/<pid>/status` as the historical
process-level mapped-memory envelope for this workload (~24–32 GiB in this
matrix), not as literal resident physical memory. Every cell JSON and rendered
matrix labels that metric explicitly. Future runs should add
`smaps_rollup`/PSS, system `MemAvailable` deltas, cgroup accounting, and
relevant GTT counters. See
[`docs/results/METHODOLOGY.md §5`](results/METHODOLOGY.md#memory-methodology)
for the methodology and empirical deltas.

## reasoning-length

**Symptom:** the model returns `finish_reason: "length"` and (at very small
`max_tokens`) an empty `content` field, even though the prompt is trivial.

**Cause:** Muse-Glimmer is a **reasoning model** — it emits chain-of-thought in
the `reasoning` channel *first*, then the answer in `content`. With
`reasoning_strength=high` (the default, and **not switchable off** —
`--reasoning off` is a no-op) the thinking can run for hundreds of tokens before
the answer. At `max_tokens=16` (or even 64) the model burns the entire budget on
reasoning and never reaches `content`, so you see `finish_reason=length` and
empty content.

**Fix:** give the model room to think. `max_tokens ≥ 256` is the practical floor
for short answers (the Study 1 greedy run uses 256 and still finishes on
`length` for some prompts — that's expected and does not affect tok/s, which is
computed over the generated tokens). For real chat turns, `max_tokens ≥ 512` is
safer. To shorten thinking, set `reasoning_strength` to `low` or `medium` via
`chat_template_kwargs`:

```json
{"messages":[...], "chat_template_kwargs":{"reasoning_strength":"low"}}
```

(`low`/`medium`/`high`/`xhigh` are the supported values; default is `high`.)

A second, related trap: `llama-server` divides `-c` across `-np` slots, so each
request gets `-c / -np` of context. If the per-slot context is below your
`max_tokens`, generation will be **silently truncated** with no error. The
benchmark harness uses `-c = np × 8192` so each slot has 8192 — comfortably above
`max_tokens`. If you configure concurrency manually, keep per-slot context well
above `max_tokens`.
