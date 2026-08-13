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

## Current official gfx1151 track — ROCm 7.14

Status: **in progress / not yet published as a validated result set**.

A small number of raw cells may exist in `matrix-714/` while the rerun is in
progress. Partial cells are not a completed comparison and must not be promoted
to the README headline. Follow the [ROCm 7.14 checklist](rocm-7.14/README.md);
render and compare only after its gates pass.

Rules:

1. Never edit or relabel the ROCm 7.2.1 cells.
2. Use the same llama.cpp commit, model hashes, prompts, seeds and flags.
3. Record the actual ROCm version in every new cell.
4. Keep failed or unstable cells with evidence rather than silently omitting them.
5. Publish conclusions only after the completion and review gates are satisfied.
