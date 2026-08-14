# Hardware validation

This project distinguishes reproducible evidence from “it worked once.”

## Status vocabulary

| Mark | Meaning |
|---|---|
| ✅ Validated | Maintainer-reviewed evidence using the project protocol |
| 🧪 Community validated | Independent evidence with complete provenance and artifacts |
| 🚧 Planned | Target of interest; no accepted evidence yet |
| 📘 Upstream recipe | Evidence owned by an upstream project, not rerun here |

Current status:

| Platform | Status |
|---|---|
| Ryzen AI MAX+ PRO 395 / Radeon 8060S (`gfx1151`) | ✅ Validated |
| Radeon W7900 (`gfx1100`) | ✅ Validated |
| Other RDNA3 Radeon | 🚧 Planned |
| RDNA4 Radeon | 🚧 Planned |
| MI300X / MI355X | 📘 Upstream recipe |

> W7900 evidence: [results/w7900-gfx1100.md](results/w7900-gfx1100.md) — Study 2 (throughput under load), reproducible via [scripts/w7900-repro/](../scripts/w7900-repro/).

## Submission requirements

Open the **Hardware validation** issue and attach or link a durable artifact
bundle. Include:

- GPU marketing name and PCI/device identity
- gfx target and compute-unit count
- dedicated VRAM or unified-memory configuration
- BIOS/UMA settings when relevant
- ROCm release, package source and active prefix
- kernel and distribution
- Python, PyTorch and torchvision versions
- vLLM and/or llama.cpp repository plus full commit
- model repository revision
- size and SHA256 verification result for every model artifact
- inference path, precision, attention backend and patches
- exact environment and command
- pass/fail outcome for load and one deterministic request
- throughput, TTFT and TPOT where available
- concurrency, context, prompt set, warmup and repetition count
- peak-memory methodology and raw counters
- parser/tool-calling, vision and long-context status where applicable
- stability duration/request count
- known issues and negative findings
- raw logs/cell JSON and an artifact link

Do not report a number without its workload definition. Do not convert a
successful text load into a vision, parser, long-context or DFlash claim.

## Minimum validation sequence

1. Run artifact verification and capture the stack.
2. Run a deterministic text smoke test.
3. Run the applicable study cells without changing their prompts or flags.
4. Run vision/parser/long-context checks only if claiming those capabilities.
5. Preserve failures and timeouts with their stopping criteria.
6. Submit raw evidence before requesting a status change.

## Artifact bundle layout

A suggested layout:

```text
hardware-validation/<gpu>-<rocm>-<date>/
  manifest.json
  commands.txt
  environment.txt
  smoke/
  matrix/
  logs/
  notes.md
```

`manifest.json` should repeat the required fields in machine-readable form.
Secrets, tokens, hostnames and unrelated system information must be removed
before submission.

A maintainer may request a rerun when the workload, source revision, artifact
identity or measurement method cannot be audited.
