# Muse-Glimmer-30B-ROCm

[![CI](https://github.com/AIwork4me/Muse-Glimmer-30B-ROCm/actions/workflows/ci.yml/badge.svg)](https://github.com/AIwork4me/Muse-Glimmer-30B-ROCm/actions/workflows/ci.yml)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

> **The reproducible RDNA reference for Meta Muse-Glimmer-30B — from
> MI-series recipes to Ryzen AI and Radeon.**

Meta's upstream recipe establishes the model on MI300X/MI355X (CDNA). This
repository records the engineering delta required for RDNA, validates it on real
hardware, preserves raw results and failures, and makes the same protocol
available to Radeon contributors.

> **Recommended stack: ROCm 7.14.0** (AMD's first official gfx1151 release), with
> the **llama.cpp / GGUF** path as the default for single-user scenarios. ROCm
> 7.2.1 is preserved as the fully-validated historical reference.

<!-- BEGIN GENERATED: validated-platform -->
**Actually validated here:** AMD Ryzen AI MAX+ PRO 395 / Radeon 8060S,
`gfx1151` (RDNA 3.5). Radeon dGPUs are planned, not claimed as validated.
<!-- END GENERATED: validated-platform -->

The working method is:

> **Adapt → Validate → Benchmark → Explain → Reproduce**

## Choose a path

| Path | Best for | What you get |
|---|---|---|
| **llama.cpp + Meta GGUF (default, ROCm 7.14)** | Single-user chat, low memory, quick start | Pinned HIP build, validated K-quant, optional vision + DFlash |
| **vLLM + BF16 (deferred / pending)** | Multi-user, agentic tool parsing, 128K context | Validated only on the ROCm 7.2.1 reference; not the focus for single-user gfx1151 |

### Default path — llama.cpp + Meta GGUF (ROCm 7.14)

```bash
git clone https://github.com/AIwork4me/Muse-Glimmer-30B-ROCm.git
cd Muse-Glimmer-30B-ROCm
bash scripts/00-check-env.sh
bash scripts/gguf-quickstart.sh
# OpenAI-compatible server: http://127.0.0.1:8080
```

The quick start defaults to **ROCm 7.14.0** (the official gfx1151 release). If
`~/rocm-7.14.0` is absent it falls back to the system `/opt/rocm` (7.2.1); install
7.14 explicitly with `bash scripts/install-rocm-7.14.sh`. It checks out the
validated llama.cpp commit, downloads from official Hugging Face at the recorded
revision, verifies size/SHA256, and reuses a matching build on reruns. Add
`WITH_MMPROJ=1` for vision or `WITH_DFLASH=1` for single-stream speculative
decoding.

### vLLM + BF16 (deferred)

vLLM/BF16 is **deferred (pending)** here: Ryzen AI MAX+ 395 / Radeon are
predominantly single-user, where the lighter llama.cpp path fits better, and a
rocBLAS BF16-GEMM proxy showed **no 7.14 compute gain over 7.2.1**. vLLM remains
validated on the **ROCm 7.2.1** historical reference (full features: reasoning +
ATEM tool parsing, vision, 128K context, continuous batching) for those who need
them:

```bash
uv sync
bash scripts/01-build-vllm.sh     # builds against /opt/rocm (7.2.1)
bash scripts/02-fetch-model.sh
bash scripts/03-serve-vllm.sh
# OpenAI-compatible server: http://127.0.0.1:8000
```

## Headline validated results

These figures are validated on **ROCm 7.14.0** (the recommended default) via the
GGUF/llama.cpp matrix ([`docs/results/matrix-714/`](docs/results/matrix-714/)),
and reproduce the ROCm 7.2.1 reference within noise (TPOT within ±2%; see
[`matrix-714/comparison.md`](docs/results/matrix-714/comparison.md)). They are
`gfx1151` results, not Radeon dGPU claims.

### Study 1 — Meta-aligned DFlash anchor

Batch 1, greedy decoding, llama.cpp, Meta K-Quant weights and quantized drafter:

| Configuration | Baseline | DFlash | Speedup |
|---|---:|---:|---:|
| gfx1151, K-Quant-17GB | 10.42 tok/s | 23.08 tok/s | **2.22×** |
| gfx1151, dynamic K-Quant | 9.11 tok/s | 22.49 tok/s | **2.47×** |

This is a **methodology-aligned comparison**, not an identical reproduction of
Meta's prompt corpus: Meta did not publish that corpus. The recorded arithmetic
smoke check is byte-identical under greedy decoding. The repository also
provides a six-prompt corpus-wide checker; no expanded pass claim is made until
that GPU test is run and recorded.

### Study 2 — original throughput-under-load study

| Weight | c=1 baseline / DFlash | c=4 baseline / DFlash | c=16 baseline |
|---|---:|---:|---:|
| 17GB | 10.50 / 21.37 tok/s | 21.93 / 32.42 tok/s | 34.47 tok/s ⚠ |
| dynamic | 9.21 / 19.69 tok/s | 20.99 / 31.10 tok/s | 31.05 tok/s ⚠ |

⚠ c=16 is **ROCm 7.2.1-only** (deferred on the 7.14 reduced matrix); c=1/c=4 are
7.14 values. This is an original concurrency study, not a Meta-aligned anchor.

### Study 3 — original multimodal validation

Five vision cells loaded the fixed test image through
`mmproj-kquant.gguf`, completed image-conditioned generation, and captured
throughput/latency/mapped-memory deltas. The raw cells do not retain response
text, so this is functional-path evidence rather than a vision-quality study.
These results apply only to the recorded `gfx1151` stack.

Full definitions, raw JSON, variance and negative cells:

- [Benchmark report](docs/results/benchmark.md)
- [Methodology](docs/results/METHODOLOGY.md)
- [ROCm 7.14 matrix (headline)](docs/results/matrix-714/) · [7.2.1 vs 7.14 comparison](docs/results/matrix-714/comparison.md)
- [Historical ROCm 7.2.1 matrix (supplementary)](docs/results/matrix/) — full 21-cell matrix, vLLM head-to-head, llama-bench
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

Three machine-readable files define the reference and public claim boundary:

- [`configs/validated-stack.json`](configs/validated-stack.json) pins hardware
  evidence, host/runtime layers, Python, PyTorch, vLLM, llama.cpp, model
  revisions, backend, precision and patches.
- [`configs/artifact-manifest.json`](configs/artifact-manifest.json) records
  exact byte sizes and SHA256 hashes for the BF16 weights/configuration and all
  GGUF, DFlash and projector files used by the published results.
- [`configs/public-claims.json`](configs/public-claims.json) controls the
  validated/planned hardware and ROCm-track status rendered above.

Versioned [JSON Schemas](schemas/) and
`scripts/check_claim_consistency.py` make these boundaries auditable in CI.

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

<!-- BEGIN GENERATED: hardware-matrix -->
| Status | Platform | Evidence |
|---|---|---|
| ✅ **Validated** | Ryzen AI MAX+ PRO 395 / Radeon 8060S, `gfx1151` | Full recorded reference in this repository |
| 🚧 **Planned** | Radeon W7900, `gfx1100` | No project evidence yet |
| 🚧 **Planned** | Other RDNA3 / future RDNA4, `various` | Requires a comparable community submission |
| 📘 **Upstream recipe** | MI300X / MI355X, `CDNA` | Upstream evidence; not revalidated here |
<!-- END GENERATED: hardware-matrix -->

`🧪 Community validated` is reserved for a submission that includes the
required manifest, command, logs and results. See
[hardware-validation.md](docs/hardware-validation.md) to add one.

## Known good and known bad

| Configuration | Status |
|---|---|
| vLLM BF16 + `TRITON_ATTN` + TP=1 | Validated on the **7.2.1 reference** (deferred on 7.14) |
| llama.cpp HIP + 17GB/dynamic K-quant | Validated on `gfx1151` |
| llama.cpp DFlash, c=1 or light c≤4 | Validated; speedup depends on workload |
| `-md dflash.gguf` without `--spec-type draft-dflash` | Silent no-op; do not use |
| DFlash + `-np 16` | Pathological; one cell aborted after 5 h 16 m |
| AITER or FP8 vLLM paths on `gfx1151` | Not supported by this validated stack |
| Radeon W7900 / other dGPUs | Pending evidence |

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
- Choose the **llama.cpp/GGUF** path (default, single-user) or the vLLM/BF16 path (ROCm 7.2.1 reference, multi-user/agentic) from measured tradeoffs.
- See which precision, attention and speculative-decoding combinations worked.
- Avoid the silent DFlash no-op and high-concurrency pathology.
- Audit exact model/runtime inputs instead of trusting a version label alone.
- Run a ~30B multimodal reasoning/tool-use model locally on validated Ryzen
  AI-class hardware.
- Contribute Radeon evidence that is comparable rather than anecdotal.

## ROCm validation tracks

<!-- BEGIN GENERATED: validation-tracks -->
- **ROCm 7.14 gfx1151 (recommended default):** the reduced **GGUF/llama.cpp
  matrix is project-validated** on Ryzen AI MAX+ PRO 395 / Radeon 8060S,
  17 of 21 planned cells; the four `np=16` cells
  were deferred. The BF16/vLLM track was **evaluated and is not pursued** (rocBLAS BF16-GEMM
  proxy: no 7.14 compute gain); vLLM/BF16 stays on the 7.2.1 reference, so ROCm 7.14 is not
  presented as a globally validated replacement for the historical stack.
- **ROCm 7.2.1 (historical reference, supplementary):** the full validated stack —
  the complete benchmark matrix, the vLLM-vs-llama.cpp head-to-head, and llama-bench — is
  preserved as immutable evidence. No result is relabeled or overwritten.
<!-- END GENERATED: validation-tracks -->

See [ROCm 7.14 scoped validation](docs/results/rocm-7.14/README.md) and its
[machine-readable manifest](configs/rocm-7.14-gguf-validation.json).

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
