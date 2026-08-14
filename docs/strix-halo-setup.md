# Strix Halo setup

Target validated by this project: **AMD Ryzen AI MAX+ PRO 395 / Radeon 8060S**
(`gfx1151`, RDNA 3.5) with 94 GiB unified LPDDR5X.

## Default path: llama.cpp + GGUF

This path uses ROCm 7.14 directly. It needs a compatible Linux host, `git`,
`cmake`, `curl`, Python 3, and enough storage for the 1.6 GiB ROCm archive,
the 15.6 GiB default GGUF, and the llama.cpp build. It does **not** need
Python 3.12 specifically, `uv`, PyTorch, TheRock wheels, or vLLM.

### 1. Check the project host floor

For this project's validated Strix Halo host, kernel 6.16.9 or newer avoids the
observed KFD/HSA unified-memory issue that exposed only about 15.5 GiB to ROCm.

AMD's [current RDNA3.5 guidance](https://rocm.docs.amd.com/en/latest/reference/system-optimization/rdna3-5.html)
uses distribution-specific kernel lines. The 6.16.9 value here is the project's
reproduction floor, not AMD's universal ROCm 7.14 requirement.

```bash
uname -r
```

The environment checker performs a numeric major/minor/patch comparison and
links to [the UMA troubleshooting entry](troubleshooting.md#uma-bug) if the host
is below that project floor.

### 2. Install ROCm 7.14 side-by-side

```bash
bash scripts/install-rocm-7.14.sh
```

The installer is idempotent. It installs the recorded AMD gfx1151 archive at
`~/rocm-7.14.0`, verifies its exact byte size and SHA256, stages extraction,
and does not overwrite `/opt/rocm`. Archive identity is recorded in
[`configs/rocm-7.14-gguf-validation.json`](../configs/rocm-7.14-gguf-validation.json).

### 3. Verify gfx1151 and memory policy

```bash
~/rocm-7.14.0/bin/hipcc --version
~/rocm-7.14.0/bin/rocminfo | grep -i gfx1151
bash scripts/00-check-env.sh --profile gguf
```

The GGUF profile selects the same ROCm prefix as the quickstart. It requires the
default GGUF weights to fit the GPU-visible pool, but a lower-memory configuration
that remains large enough for the weights receives a warning rather than a false
unsupported-hardware failure. Published project evidence remains scoped to the
recorded 94 GiB Strix Halo configuration.

### 4. Start Muse-Glimmer

```bash
bash scripts/gguf-quickstart.sh
# OpenAI-compatible server: http://127.0.0.1:8080
```

The first run checks out the pinned llama.cpp commit, builds HIP for `gfx1151`,
downloads the pinned default GGUF from official Hugging Face, and verifies its
size and SHA256. Matching source, toolchain, and model assets are reused on later
runs.

Optional vision and DFlash use the same setup:

```bash
WITH_MMPROJ=1 bash scripts/gguf-quickstart.sh
WITH_DFLASH=1 bash scripts/gguf-quickstart.sh
```

## Optional advanced path: vLLM + BF16

**Optional / not prioritized for v0.1; ROCm 7.14 Muse-Glimmer vLLM validation
pending.** This is a separate, maintainer-oriented path for multi-user serving
and features such as agentic tool parsing and 128K context. It remains validated
on the historical ROCm 7.2.1 reference stack.

Only this path requires:

- Python 3.12;
- `uv` and the locked TheRock gfx1151 PyTorch runtime;
- the complete ROCm 7.2.1 development toolchain at `/opt/rocm`;
- at least 60 GiB of GPU-visible unified memory for the validated BF16 envelope;
- a pinned vLLM source build plus the two committed compatibility patches.

```bash
uv sync --locked
ROCM_PREFIX=/opt/rocm bash scripts/00-check-env.sh --profile vllm
bash scripts/01-build-vllm.sh
bash scripts/02-fetch-model.sh
bash scripts/03-serve-vllm.sh
# OpenAI-compatible server: http://127.0.0.1:8000
```

Do not apply the 60 GiB BF16 requirement or TheRock/PyTorch setup to a GGUF-only
installation. The exact hybrid reference layers and why they are needed are
documented in the [CDNA to RDNA adaptation map](adaptation.md#packaging-and-runtime-layers).

## Historical ROCm 7.2.1 reference

The system `/opt/rocm` installation is preserved for the historical vLLM/BF16
stack and regression evidence. It is a fallback, not the recommended first-run
path. The immutable 7.2.1 matrix and the scoped ROCm 7.14 GGUF matrix remain
separate in the [validation-track index](results/README.md).
