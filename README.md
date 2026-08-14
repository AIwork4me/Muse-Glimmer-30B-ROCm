# Muse-Glimmer-30B-ROCm

[![CI](https://github.com/AIwork4me/Muse-Glimmer-30B-ROCm/actions/workflows/ci.yml/badge.svg)](https://github.com/AIwork4me/Muse-Glimmer-30B-ROCm/actions/workflows/ci.yml)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

> **The reproducible RDNA reference for Meta Muse-Glimmer-30B — from
> MI-series recipes to Ryzen AI and Radeon.**

Meta's upstream recipe establishes the model on MI300X/MI355X (CDNA). This
repository records the engineering delta required for RDNA, validates it on real
hardware, preserves raw results and failures, and makes the same protocol
available to Radeon contributors.

**Actually validated here:** AMD Ryzen AI MAX+ PRO 395 / Radeon 8060S
(`gfx1151`, RDNA 3.5) and Radeon PRO W7900 (`gfx1100`, RDNA 3).

The working method is:

> **Adapt → Validate → Benchmark → Explain → Reproduce**

## Choose a path

| Path | Best for | What you get |
|---|---|---|
| **Fast local path — llama.cpp + Meta GGUF** | Local chat, low memory, quick evaluation | Pinned HIP build, validated K-quant, optional vision and DFlash |
| **Full inference stack — vLLM + BF16** | Reasoning/tool parsing, vision, 128K context, continuous batching | Pinned source build, TheRock gfx1151 runtime, `TRITON_ATTN` |

### Fast local path

```bash
git clone https://github.com/AIwork4me/Muse-Glimmer-30B-ROCm.git
cd Muse-Glimmer-30B-ROCm
bash scripts/00-check-env.sh
bash scripts/gguf-quickstart.sh
# OpenAI-compatible server: http://127.0.0.1:8080
```

The quick start checks out the validated llama.cpp commit, downloads from
official Hugging Face at the recorded model revision, verifies size/SHA256, and
reuses a matching build on reruns. Add `WITH_MMPROJ=1` for vision or
`WITH_DFLASH=1` for single-stream speculative decoding.

### Full inference stack

```bash
uv sync
bash scripts/00-check-env.sh
bash scripts/01-build-vllm.sh
bash scripts/02-fetch-model.sh
bash scripts/03-serve-vllm.sh
# OpenAI-compatible server: http://127.0.0.1:8000
```

This path needs about 60 GiB of GPU-visible unified memory and a source build.
The environment checker treats that threshold as a hard requirement for this
validated BF16 configuration.

## Headline validated results

The `gfx1151` figures below are historical evidence from the preserved ROCm
7.2.1 reference stack; the `gfx1100` (W7900) rows in Study 2 are the Radeon
dGPU validation. Neither set is a ROCm 7.14 claim.

### Study 1 — Meta-aligned DFlash anchor

Batch 1, greedy decoding, llama.cpp, Meta K-Quant weights and quantized drafter:

| Configuration | Baseline | DFlash | Speedup |
|---|---:|---:|---:|
| gfx1151, K-Quant-17GB | 10.48 tok/s | 23.03 tok/s | **2.20×** |
| gfx1151, dynamic K-Quant | 9.14 tok/s | 21.82 tok/s | **2.39×** |
| W7900 (`gfx1100`), K-Quant-17GB | 33.19 tok/s | 63.98 tok/s | **1.93×** |
| W7900 (`gfx1100`), dynamic K-Quant | 30.23 tok/s | 58.26 tok/s | **1.93×** |

This is a **methodology-aligned comparison**, not an identical reproduction of
Meta's prompt corpus: Meta did not publish that corpus. The recorded arithmetic
smoke check is byte-identical under greedy decoding. The repository also
provides a six-prompt corpus-wide checker; no expanded pass claim is made until
that GPU test is run and recorded.

The W7900 (`gfx1100`) rows are new greedy measurements on the same corpus
(draft acceptance ~0.24); DFlash is ~1.93×, a touch below gfx1151 because the
desktop card's much faster baseline is less memory-bound.

### Study 2 — original throughput-under-load study

`c` is llama.cpp concurrency (`-np`); an original concurrency study, not a
Meta-aligned anchor. gfx1151 was measured to `c=16`; the W7900 desktop GPU
extends to `c=32`. DFlash is shown at `c≤4`, where it helps.

| GPU | Weight | c=1 baseline / DFlash | c=4 baseline / DFlash | c=16 baseline | c=32 baseline |
|---|---|---:|---:|---:|---:|
| gfx1151 | 17GB | 10.52 / 22.26 tok/s | 15.60 / 27.30 tok/s | 34.47 tok/s | — |
| gfx1151 | dynamic | 9.09 / 19.89 tok/s | 20.90 / 28.22 tok/s | 31.05 tok/s | — |
| W7900 (`gfx1100`) | 17GB | 33.72 / 56.35 tok/s | 66.46 / 77.44 tok/s | 218.56 tok/s | 293.29 tok/s |
| W7900 (`gfx1100`) | dynamic | 30.61 / 51.92 tok/s | 65.63 / 71.51 tok/s | 209.65 tok/s | 265.71 tok/s |

W7900 runs ~3.2–3.4× the gfx1151 APU at `c=1` and up to ~6.8× at `c=16`
(dedicated GDDR6 vs unified LPDDR5X); peak VRAM ≤ 24.9 GiB of 48. Full detail +
reproduction: [W7900 results](docs/results/w7900-gfx1100.md), [`scripts/w7900-repro/`](scripts/w7900-repro/).

**Recommended W7900 serving presets** — high throughput: `-np 16 -c 1048576`
(16 slots × 64K, DFlash off); long context (full 128K, single request):
`-np 1 -c 131072` + DFlash on, or `-np 2 -c 262144` for two 128K streams. See
[recommended configurations](docs/results/w7900-gfx1100.md#recommended-serving-configurations-w7900).

### Study 3 — original multimodal validation

Five vision cells loaded the fixed test image through
`mmproj-kquant.gguf`, produced image-grounded responses, and captured
throughput/latency/mapped-memory deltas. These results validate only the
recorded `gfx1151` stack.

Full definitions, raw JSON, variance and negative cells:

- [Benchmark report](docs/results/benchmark.md)
- [Methodology](docs/results/METHODOLOGY.md)
- [Raw ROCm 7.2.1 matrix](docs/results/matrix/)
- [Validation-track index](docs/results/README.md)

## Why this repository exists

The upstream CDNA recipe is the reference point, not baggage. The contribution
here is the documented translation layer:

| Engineering layer | Upstream MI-series / CDNA reference | Validated RDNA translation |
|---|---|---|
| Packaging | ROCm vLLM image / recipe | Source build for `gfx1151` |
| Kernels | AITER-oriented recipe | `TRITON_ATTN`; AITER disabled |
| Precision | BF16 and supported FP8 paths | BF16 for vLLM; Meta K-quant for llama.cpp |
| Topology | MI300X/MI355X tensor parallel | Single integrated GPU, TP=1 |
| Memory | Dedicated HBM accounting | Unified-memory caveats and process mapped-memory envelope |
| Speculative decoding | DFlash in upstream model work | Validated llama.cpp path; vLLM limitation recorded |
| Evidence | Upstream MI validation | Raw RDNA cells, exact flags, failures, hashes and manifests |

This is not a collection of random workarounds. It is a
[documented CDNA → RDNA adaptation map](docs/adaptation.md), with each delta
classified as architectural, version-specific, temporary upstream limitation,
validated workaround, or historical workaround.

## Reproducibility contract

Two machine-readable files define the reference:

- [`configs/validated-stack.json`](configs/validated-stack.json) pins hardware
  evidence, host/runtime layers, Python, PyTorch, vLLM, llama.cpp, model
  revisions, backend, precision and patches.
- [`configs/artifact-manifest.json`](configs/artifact-manifest.json) records
  exact byte sizes and SHA256 hashes for the BF16 weights/configuration and all
  GGUF, DFlash and projector files used by the published results.

The defaults are the **validated reference**. Overrides are explicit and
reported as **latest/experimental**:

```bash
# Optional regional mirror; the official endpoint remains the default.
HF_ENDPOINT=https://hf-mirror.com bash scripts/02-fetch-model.sh

# Deliberately follow newer upstream state; benchmark claims no longer apply.
MODEL_REVISION=main bash scripts/02-fetch-model.sh
LLAMA_CPP_REF=master GGUF_REVISION=main bash scripts/gguf-quickstart.sh
```

Mirrors are transport fallbacks, not trust roots. Validated artifacts are still
checked against the committed manifest.

## Hardware validation matrix

| Status | Platform | Evidence |
|---|---|---|
| ✅ **Validated** | Ryzen AI MAX+ PRO 395 / Radeon 8060S, `gfx1151` | Full recorded reference in this repository |
| ✅ **Validated** | Radeon W7900, `gfx1100` | Study 2 throughput — [docs/results/w7900-gfx1100.md](docs/results/w7900-gfx1100.md) |
| 🚧 **Planned** | Other RDNA3 Radeon | Requires a comparable community submission |
| 🚧 **Planned** | RDNA4 Radeon | Requires a comparable community submission |
| 📘 **Upstream recipe** | MI300X / MI355X, CDNA | vLLM recipes PR #776; not revalidated here |

`🧪 Community validated` is reserved for a submission that includes the
required manifest, command, logs and results. See
[hardware-validation.md](docs/hardware-validation.md) to add one.

## Known good and known bad

| Configuration | Status |
|---|---|
| vLLM BF16 + `TRITON_ATTN` + TP=1 | Validated on `gfx1151` |
| llama.cpp HIP + 17GB/dynamic K-quant | Validated on `gfx1151` |
| llama.cpp DFlash, c=1 or light c≤4 | Validated; speedup depends on workload |
| `-md dflash.gguf` without `--spec-type draft-dflash` | Silent no-op; do not use |
| DFlash + `-np 16` | Pathological; one cell aborted after 5 h 16 m |
| AITER or FP8 vLLM paths on `gfx1151` | Not supported by this validated stack |
| llama.cpp HIP + 17GB/dynamic K-quant on `gfx1100` (W7900) | Validated (Study 2 throughput; DFlash helps at c≤4) |
| Other RDNA3 / RDNA4 dGPUs | Pending evidence |

**Negative results are results.** The project preserves the silent-DFlash
discovery, non-completing c=16 cells, backend failures and measurement
limitations because they prevent other developers from repeating the same
mistakes.

## Memory terminology

For this mmap + GPU-offload workload, **VmPeak is the most useful
process-level mapped-memory envelope observed**. It is virtual address-space
size, not resident physical memory. RSS/VmHWM and `rocm-smi` VRAM counters
alone materially under-report the effective unified-memory footprint on this
system. The methodology preserves all three and describes stronger future
measurements such as `smaps_rollup`, PSS, system `MemAvailable` deltas and
cgroup accounting.

## Why this helps AMD AI developers

- Reuse a reviewed CDNA → RDNA adaptation instead of rediscovering it.
- Choose llama.cpp or vLLM from measured workload and feature tradeoffs.
- See which precision, attention and speculative-decoding combinations worked.
- Avoid the silent DFlash no-op and high-concurrency pathology.
- Audit exact model/runtime inputs instead of trusting a version label alone.
- Run a ~30B multimodal reasoning/tool-use model locally on validated Ryzen
  AI-class hardware.
- Contribute Radeon evidence that is comparable rather than anecdotal.

## ROCm validation tracks

- **Validated historical/reference stack:** ROCm 7.2.1 host toolchain plus the
  recorded TheRock runtime. Existing benchmark JSON is immutable evidence.
- **Current official gfx1151 track:** ROCm 7.14. Results are **pending** until
  the checklist is rerun. No 7.2.1 result is relabeled or overwritten.

See [ROCm 7.14 validation](docs/results/rocm-7.14/README.md).

## Requirements and operating notes

- Linux kernel **≥ 6.16.9**; the comparison includes patch level.
- Python **3.12** for the recorded TheRock wheel line.
- About 20 GiB available for the default GGUF path; at least 60 GiB
  GPU-visible unified memory for the validated BF16 path.
- `uv`, `git`, `cmake`, `curl`, and a gfx1151-capable HIP toolchain.
- `uv run --no-sync` remains required after the editable vLLM source install;
  a normal sync reconstructs the locked environment and removes that editable
  install.

Setup and failure modes:

- [Strix Halo setup](docs/strix-halo-setup.md)
- [Troubleshooting](docs/troubleshooting.md)
- [Adaptation map](docs/adaptation.md)

## Contributing and security

Read [CONTRIBUTING.md](CONTRIBUTING.md) before submitting hardware or benchmark
claims. Use the issue templates for bugs, regressions, platform requests and
hardware validation. Report security issues through [SECURITY.md](SECURITY.md).

## License and acknowledgements

Repository code is [Apache-2.0](LICENSE). Model artifacts are published by Meta
under Apache-2.0 with a separate usage policy; downloading them means accepting
that policy.

Thanks to Meta for Muse-Glimmer-30B and its model artifacts, vLLM and llama.cpp
contributors for the inference implementations, and AMD TheRock/ROCm
contributors for the `gfx1151` runtime and toolchain work.
