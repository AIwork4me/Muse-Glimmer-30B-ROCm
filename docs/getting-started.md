# Getting started — details

Companion to the [README quick start](../README.md). Everything here is
optional reading: prerequisites, what the installer actually does, download
sizes, optional features, and the reproducibility knobs that keep runs on the
validated reference.

## Prerequisites

Default GGUF path: `git`, `cmake`, `curl`, Python 3, and a gfx1151-capable HIP
toolchain. PyTorch, `uv` and the vLLM environment are **not** required. If one
of the four host tools is missing, the scripts stop with
`required command not found`; the installer and environment checker print the
matching per-distro install command for the missing tool. Install all four up
front:

```bash
sudo apt-get install git cmake curl python3   # Debian/Ubuntu
sudo dnf install git cmake curl python3       # Fedora/RHEL
sudo pacman -S git cmake curl python3         # Arch
```

Optional vLLM path: Python 3.12, `uv`, and the locked TheRock runtime.
`uv run --no-sync` remains required after the editable vLLM source install.

## Disk and memory

- About 20 GiB available for the default GGUF path; at least 60 GiB
  GPU-visible unified memory for the validated BF16 path.
- Disk (not GPU memory): the ROCm installer alone needs ~11 GiB free at its
  transient peak — the 1.6 GiB archive under `$TMPDIR` plus the 8.3 GiB
  extracted tree at `~/rocm-7.14.0`. The installer checks both filesystems
  before downloading and deletes the verified archive once it succeeds.
- For this project's validated Strix Halo host, kernel **≥ 6.16.9** avoids the
  observed UMA/KFD issue. AMD's [current RDNA3.5 requirements](https://rocm.docs.amd.com/en/latest/reference/system-optimization/rdna3-5.html)
  use distribution-specific kernel lines; 6.16.9 is not a universal ROCm floor.

## What the installer does

`scripts/install-rocm-7.14.sh` is idempotent: it installs AMD's gfx1151 ROCm
7.14 archive at `~/rocm-7.14.0` without overwriting `/opt/rocm`. The
environment checker and quickstart share one resolver, so both select that 7.14
installation. ROCm 7.2.x at `/opt/rocm` is accepted only as a clearly reported
historical fallback.

The quickstart then checks out the validated llama.cpp commit, verifies the
model's exact size and SHA256, builds HIP for `gfx1151`, and reuses matching
assets on reruns.

## Download sizes

| Artifact | Size | Used by |
|---|---|---|
| ROCm 7.14 archive (if not installed) | ~1.6 GiB | every path |
| `muse-glimmer-30B-kquant-17gb.gguf` | 15.6 GiB | default GGUF path |
| `mmproj-kquant.gguf` | ~1.3 GiB | optional image input |
| `dflash-kquant.gguf` | ~1.5 GiB | optional speculative decoding |

Each optional artifact is one manifest-verified download on first use.

## Optional wrapper and features

Prefer a confirmed one-command entry point? The optional wrapper prints the
manifest-derived ROCm/model download sizes and waits for approval before it
starts either download:

```bash
bash scripts/quickstart.sh       # interactive y/N confirmation
bash scripts/quickstart.sh --yes # explicit non-interactive approval
```

```bash
WITH_MMPROJ=1 bash scripts/gguf-quickstart.sh  # image input
WITH_DFLASH=1 bash scripts/gguf-quickstart.sh  # single-stream speculative decoding
```

## Reproducibility contract

Three machine-readable files define the reference and public claim boundary:

- [`configs/validated-stack.json`](../configs/validated-stack.json) pins
  hardware evidence, host/runtime layers, Python, PyTorch, vLLM, llama.cpp,
  model revisions, backend, precision and patches.
- [`configs/artifact-manifest.json`](../configs/artifact-manifest.json)
  records exact byte sizes and SHA256 hashes for the BF16
  weights/configuration and all GGUF, DFlash and projector files used by the
  published results.
- [`configs/public-claims.json`](../configs/public-claims.json) controls the
  validated/planned hardware and ROCm-track status rendered in the README.

Versioned [JSON Schemas](../schemas/) and
`scripts/check_claim_consistency.py` make these boundaries auditable in CI.

The defaults are the **validated reference**. Overrides are explicit. Only
revision overrides (`MODEL_REVISION`, `LLAMA_CPP_REF`, `GGUF_REVISION`) move a
run to the reported-as-**latest/experimental** track, where benchmark claims no
longer apply; the transport and location knobs (`HF_ENDPOINT`, `MODEL_DEST`)
change where artifacts come from or land and keep the validated manifest
checks:

```bash
# Optional regional mirror; the official endpoint remains the default.
HF_ENDPOINT=https://hf-mirror.com bash scripts/02-fetch-model.sh

# Store the ~15.6 GiB model outside the clone; the same hash-verified file
# is then reused across clones instead of re-downloaded per clone.
MODEL_DEST=/shared/models bash scripts/gguf-quickstart.sh

# Deliberately follow newer upstream state; benchmark claims no longer apply.
MODEL_REVISION=main bash scripts/02-fetch-model.sh
LLAMA_CPP_REF=master GGUF_REVISION=main bash scripts/gguf-quickstart.sh
```

Mirrors are transport fallbacks, not trust roots. Validated artifacts are still
checked against the committed manifest.

## Where to go next

- [Strix Halo setup](strix-halo-setup.md) — host preparation and failure modes
- [Troubleshooting](troubleshooting.md) — observed failures, negative results,
  workarounds
- [Adaptation map](adaptation.md) — the CDNA → RDNA engineering deltas
- [Benchmark report](results/benchmark.md) ·
  [methodology](results/METHODOLOGY.md) · [raw matrices](results/README.md)
