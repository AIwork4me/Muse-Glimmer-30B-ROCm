# Maintainer handoff

This file is a durable map for maintainers, not a snapshot of a particular
machine, branch, or work session. Current platform and release status belongs in
the structured manifests and validation-track documentation linked below.

## Project boundary

Muse-Glimmer-30B-ROCm is the reproducible RDNA adaptation reference for Meta
Muse-Glimmer-30B. The upstream MI-series/CDNA recipe is the reference point;
this repository documents, validates, benchmarks, and explains the engineering
delta for Ryzen AI and Radeon.

Only Ryzen AI MAX+ PRO 395 / Radeon 8060S (`gfx1151`) has project-validated
evidence today. ROCm 7.2.1 remains the full historical reference stack. The
ROCm 7.14 GGUF/llama.cpp track is project-validated within a reduced 17-cell
scope; the BF16/vLLM track was evaluated and not pursued (a rocBLAS BF16-GEMM proxy showed no 7.14 compute gain over 7.2.1), and the Radeon dGPU track remains pending.

## Sources of truth

- [Validated stack](configs/validated-stack.json): hardware, host/runtime layers,
  engine commits, model revisions and historical reference status.
- [ROCm 7.14 scoped manifest](configs/rocm-7.14-gguf-validation.json): archive,
  runtime, scope, engine/model identities and evidence checksums.
- [Artifact manifest](configs/artifact-manifest.json): model file sizes and
  SHA256 digests.
- [Public claims](configs/public-claims.json): validated/planned/upstream status
  used by the README and hardware matrix.
- [Adaptation map](docs/adaptation.md): the CDNA → RDNA engineering deltas and
  their durability taxonomy.
- [Benchmark methodology](docs/results/METHODOLOGY.md) and
  [historical raw matrix](docs/results/matrix/): workload definitions and
  immutable evidence.
- [Validation-track index](docs/results/README.md): historical versus forward
  validation boundaries.
- [Troubleshooting](docs/troubleshooting.md): observed failures, negative
  results, and workarounds.
- [Hardware validation](docs/hardware-validation.md): evidence required before
  promoting another GPU.
- [Release checklist](docs/RELEASE_CHECKLIST.md): publication gates.

Do not duplicate versions, commits, hashes, benchmark numbers, or validation
status here. Update the appropriate source of truth and run the consistency
checks.

## Maintenance invariants

1. Never rewrite historical raw benchmark cells to represent a newer stack.
2. Never promote hardware or ROCm status without accepted, auditable evidence.
3. Preserve negative and non-completing results; they are findings.
4. Keep official Hugging Face as the default endpoint. Mirrors are explicit
   transport overrides and do not change artifact identity.
5. Keep llama.cpp, vLLM, and model revisions pinned for the validated path.
   Experimental overrides must identify themselves and forfeit reference claims.
6. Do not infer resident physical memory from VmPeak. It is the most useful
   process-level mapped-memory envelope for the recorded mmap + GPU-offload
   workload, alongside RSS/VmHWM and VRAM counters.
7. A source-built editable vLLM install is outside the uv lock. Runtime scripts
   use `uv run --no-sync` so uv does not remove it.
8. DFlash requires `--spec-type draft-dflash`; the historical c=16 negative
   cells must remain visible.

## CI responsibilities

[Fast CI](.github/workflows/ci.yml) installs only the lightweight `ci`
dependency group and runs no-GPU/no-server tests, shell syntax and ShellCheck.
[Dependency smoke](.github/workflows/dependency-smoke.yml) installs the locked
TheRock/ROCm project graph only when dependency-defining files change or a
maintainer dispatches it. This separation keeps routine pull requests fast
without giving up lock-resolution coverage.

Before opening a pull request:

```bash
uv sync --only-group ci --locked
uv run --no-sync pytest -m "not gpu and not server" -v
bash -n scripts/*.sh scripts/lib/*.sh
uv run --no-sync shellcheck -x scripts/*.sh scripts/lib/*.sh
python3 scripts/check_claim_consistency.py
git diff --check
```

Use `uv sync --dev --locked` only when the full TheRock/ROCm dependency smoke
is relevant. GPU or server claims require the corresponding marked tests and
raw evidence; hosted CI cannot supply that evidence.

## Safe change routing

- Adaptation behavior or workaround classification → `docs/adaptation.md`
- Historical result interpretation → `docs/results/benchmark.md` or
  `docs/results/METHODOLOGY.md`; never edit the raw cell to improve wording
- Platform/track status → evidence bundle, scoped manifest and review, then
  `configs/public-claims.json`; keep the historical stack identity separate
- Artifact identity → `configs/artifact-manifest.json`, only from verified bytes
- New benchmark record format → a versioned schema and migration plan; do not
  silently change an active validation protocol

If these invariants conflict with a convenient cleanup, preserve the evidence
and make the limitation explicit.
