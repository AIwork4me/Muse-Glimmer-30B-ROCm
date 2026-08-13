# Validation tracks

Benchmark results are separated by evidence status. A newer ROCm label never
replaces an older validated record.

## Validated historical/reference stack — ROCm 7.2.1

- Methodology: [METHODOLOGY.md](METHODOLOGY.md)
- Interpretation: [benchmark.md](benchmark.md)
- Immutable raw cells: [matrix/](matrix/)
- Stack definition: [../../configs/validated-stack.json](../../configs/validated-stack.json)
- Artifact hashes: [../../configs/artifact-manifest.json](../../configs/artifact-manifest.json)

The raw matrix contains completed cells and two evidence-based pathological
non-completions. Both are results.

## ROCm 7.14 gfx1151 distribution — scoped project validation

Status: **GGUF/llama.cpp reduced matrix project-validated (2026-08-13);
BF16/vLLM evaluated and not pursued (no 7.14 compute gain).** This is not a global ROCm 7.14 validation claim.

The 17-cell run used AMD's official gfx1151 ROCm 7.14.0 tarball on Ryzen AI
MAX+ PRO 395 / Radeon 8060S. AMD's 7.14 release notes do not list this exact
SKU, so the hardware result is project evidence rather than an AMD support
claim. Mean TPOT deltas were −0.4% at `np=1` and −1.7% at `np=4`; all 17 cells
had a lower VmPeak mapped-address-space envelope (mean −2.8%). DFlash acceptance
rates were similar, with the largest observed difference 1.21 percentage
points. No incident was observed during the six-hour run, but raw system logs
were not retained, so this is not a standalone stability qualification.

See the [result summary](rocm-7.14/README.md), [scoped manifest](../../configs/rocm-7.14-gguf-validation.json),

Rules:

1. Never edit or relabel the ROCm 7.2.1 cells.
2. Use the same llama.cpp commit, model hashes, prompts, seeds and flags.
3. Record the actual ROCm version in every new cell.
4. Keep failed or unstable cells with evidence rather than silently omitting them.
5. Publish conclusions only after the completion and review gates are satisfied.
