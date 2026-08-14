# Muse-Glimmer-30B-ROCm

[![CI](https://github.com/AIwork4me/Muse-Glimmer-30B-ROCm/actions/workflows/ci.yml/badge.svg)](https://github.com/AIwork4me/Muse-Glimmer-30B-ROCm/actions/workflows/ci.yml)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

> **The reproducible RDNA reference for Meta Muse-Glimmer-30B — from
> MI-series recipes to Ryzen AI and Radeon.**

Meta's upstream recipe establishes the model on MI300X/MI355X (CDNA). This
repository records the engineering delta required for RDNA, validates it on real
hardware, preserves raw results and failures, and makes the same protocol
available to Radeon contributors.

> **Recommended stack: ROCm 7.14.0** (an AMD-supported gfx1151 release), with
> the **llama.cpp / GGUF** path as the default for single-user scenarios. ROCm
> 7.2.1 is preserved as the fully-validated historical reference.

<!-- BEGIN GENERATED: validated-platform -->
**Actually validated here:** AMD Ryzen AI MAX+ PRO 395 / Radeon 8060S,
`gfx1151` (RDNA 3.5). Every additional platform remains evidence-gated by the matrix below.
<!-- END GENERATED: validated-platform -->

AMD's [ROCm 7.14 release notes](https://rocm.docs.amd.com/en/docs-7.14.0/about/release-notes.html)
list Ryzen AI Max+ PRO 395 / Radeon 8060S as `gfx1151`. AMD platform support is
distinct from the Muse-Glimmer workload validation and benchmark evidence
produced independently by this project.

The working method is:

> **Adapt → Validate → Benchmark → Explain → Reproduce**

## Quick start — Ryzen AI, ROCm 7.14, GGUF

```bash
git clone https://github.com/AIwork4me/Muse-Glimmer-30B-ROCm.git
cd Muse-Glimmer-30B-ROCm
bash scripts/install-rocm-7.14.sh
bash scripts/00-check-env.sh
bash scripts/gguf-quickstart.sh
# OpenAI-compatible server: http://127.0.0.1:8080
# Leave this terminal running (Ctrl-C stops the server); open a second
# terminal for the verification requests below.
```

### Verify it works

From that second terminal:

```bash
curl http://127.0.0.1:8080/health
# {"status":"ok"}

curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Reply with exactly: OK"}],"max_tokens":512}'
```

The completion returns `"content":"OK"` once generation finishes (abridged;
the visible answer arrives after ~50–70 hidden reasoning tokens):

```json
{"choices":[{"finish_reason":"stop","message":{"role":"assistant","content":"OK","reasoning_content":"..."}}],"usage":{"completion_tokens":62,"prompt_tokens":61}}
```

> **Muse-Glimmer is reasoning-first:** it writes ~50–70 tokens of hidden
> chain-of-thought (`reasoning_content`) before any visible `content`, so a
> small `max_tokens` (e.g. 16) spends the whole budget on reasoning and returns
> empty `content` with `finish_reason:"length"` — HTTP 200, no error. Use
> `max_tokens` ≥ 512 or omit it
> ([details](docs/troubleshooting.md#reasoning-length)).

Prefer a confirmed one-command entry point? The optional wrapper prints the
manifest-derived ROCm/model download sizes and waits for approval before it
starts either download:

```bash
bash scripts/quickstart.sh       # interactive y/N confirmation
bash scripts/quickstart.sh --yes # explicit non-interactive approval
```

The installer is idempotent: it installs AMD's gfx1151 ROCm 7.14 archive at
`~/rocm-7.14.0` without overwriting `/opt/rocm`. The environment checker and
quickstart share one resolver, so both select that 7.14 installation. ROCm 7.2.x
at `/opt/rocm` is accepted only as a clearly reported historical fallback.

The first run downloads about 1.6 GiB for ROCm when it is not already installed,
then the 15.6 GiB default GGUF. The quickstart checks out the validated
llama.cpp commit, verifies the model's exact size and SHA256, builds HIP for
`gfx1151`, and reuses matching assets on reruns. It needs `git`, `cmake`,
`curl`, Python 3, and the selected HIP toolchain; it does **not** require
PyTorch, `uv`, or the vLLM environment.

Optional features use the same path:

```bash
WITH_MMPROJ=1 bash scripts/gguf-quickstart.sh  # image input
WITH_DFLASH=1 bash scripts/gguf-quickstart.sh  # single-stream speculative decoding
```

## Headline validated results

These figures are validated on **ROCm 7.14.0** (the recommended default) via the
GGUF/llama.cpp matrix ([`docs/results/matrix-714/`](docs/results/matrix-714/)).
Mean TPOT delta versus 7.2.1 was -0.4% at `np=1` and -1.7% at `np=4`;
individual cells ranged from -6.4% to +9.0% and -5.4% to +0.2%, respectively
([comparison](docs/results/matrix-714/comparison.md)). These are
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

## Optional advanced path — vLLM + BF16

| Path | Best for | Status |
|---|---|---|
| **llama.cpp + Meta GGUF** | Single-user chat, low-memory onboarding | Default; project-validated on ROCm 7.14 |
| **vLLM + BF16** | Multi-user serving, agentic tool parsing, 128K context | Optional / not prioritized for v0.1; ROCm 7.14 Muse-Glimmer validation pending |

**Optional / not prioritized for v0.1; ROCm 7.14 Muse-Glimmer vLLM validation
pending.** The upstream model-support [PR #51655](https://github.com/vllm-project/vllm/pull/51655)
is open and remains a separate dependency; AMD's ROCm platform support does not
imply Muse-Glimmer model support in vLLM. Historical vLLM/BF16 validation remains
available on the **ROCm 7.2.1** reference (reasoning + ATEM tool parsing, vision,
128K context, continuous batching). Current rocBLAS BF16-GEMM proxy results did
not justify prioritizing a ROCm 7.14 vLLM rebuild for v0.1; this does not
establish zero value for a future cohesive ROCm 7.14 vLLM stack. This optional
path has separate Python 3.12, `uv`, TheRock PyTorch, and
memory prerequisites:

```bash
uv sync --locked
ROCM_PREFIX=/opt/rocm bash scripts/00-check-env.sh --profile vllm
bash scripts/01-build-vllm.sh
bash scripts/02-fetch-model.sh
bash scripts/03-serve-vllm.sh
# OpenAI-compatible server: http://127.0.0.1:8000
```

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
| ✅ **Validated** | Ryzen AI MAX+ PRO 395 / Radeon 8060S, `gfx1151` | [Recorded project evidence](configs/validated-stack.json) |
| 🚧 **Planned** | Radeon W7900, `gfx1100` | Requires a comparable community submission |
| 🚧 **Planned** | Other RDNA3 / future RDNA4, `various` | Requires a comparable community submission |
| 📘 **Upstream recipe** | MI300X / MI355X, `CDNA` | Upstream evidence; not revalidated here |
<!-- END GENERATED: hardware-matrix -->

`🧪 Community validated` is reserved for a submission that includes the
required manifest, command, logs and results. See
[hardware-validation.md](docs/hardware-validation.md) to add one.

## Known good and known bad

| Configuration | Status |
|---|---|
| vLLM BF16 + `TRITON_ATTN` + TP=1 | Validated on the **7.2.1 reference**; ROCm 7.14 Muse-Glimmer validation pending |
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
  were deferred. **Optional / not prioritized for v0.1; ROCm 7.14 Muse-Glimmer vLLM
  validation pending.** Current rocBLAS BF16-GEMM proxy results did not justify prioritizing
  a 7.14 rebuild; vLLM/BF16 stays validated on the 7.2.1 reference, so ROCm 7.14 is not
  presented as a globally validated replacement for the historical stack.
- **ROCm 7.2.1 (historical reference, supplementary):** the full validated stack —
  the complete benchmark matrix, the vLLM-vs-llama.cpp head-to-head, and llama-bench — is
  preserved as immutable evidence. No result is relabeled or overwritten.
<!-- END GENERATED: validation-tracks -->

See [ROCm 7.14 scoped validation](docs/results/rocm-7.14/README.md) and its
[machine-readable manifest](configs/rocm-7.14-gguf-validation.json).

## Requirements and operating notes

- For this project's validated Strix Halo host, kernel **≥ 6.16.9** avoids
  the observed UMA/KFD issue. AMD's [current RDNA3.5 requirements](https://rocm.docs.amd.com/en/latest/reference/system-optimization/rdna3-5.html)
  use distribution-specific kernel lines; 6.16.9 is not a universal ROCm floor.
- About 20 GiB available for the default GGUF path; at least 60 GiB
  GPU-visible unified memory for the validated BF16 path.
- Disk (not GPU memory): the ROCm installer alone needs ~11 GiB free at its
  transient peak — the 1.6 GiB archive under `$TMPDIR` plus the 8.3 GiB
  extracted tree at `~/rocm-7.14.0`. The installer checks both filesystems
  before downloading and deletes the verified archive once it succeeds.
- Default GGUF path: git, cmake, curl, Python 3, and a gfx1151-capable HIP
  toolchain. PyTorch and uv are not required. If one of the four host tools is
  missing, the scripts stop with `required command not found` and print these
  same one-liners:

  ```bash
  sudo apt-get install git cmake curl python3   # Debian/Ubuntu
  sudo dnf install git cmake curl python3       # Fedora/RHEL
  sudo pacman -S git cmake curl python3         # Arch
  ```

- Optional vLLM path: Python 3.12, uv, and the locked TheRock runtime.
  uv run --no-sync remains required after the editable vLLM source install.

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
