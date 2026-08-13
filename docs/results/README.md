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

Status: **GGUF matrix validated (2026-08-13); vLLM/BF16 track pending.**

The 7.14 GGUF matrix (17 cells, c=16 deferred) ran against official stable
ROCm 7.14.0 side-by-side with 7.2.1 — same llama.cpp `0b1bad1`, flags, weights,
prompts, seeds; only ROCm differs. **Result: 7.14.0 ≈ 7.2.1 on per-token decode
throughput (TPOT c=1 −0.4%, c=4 −1.7%), with a consistent −2.8% VmPeak and
identical DFlash acceptance; zero stability incidents in 6 h.** Aggregate
`tok/s` at `temp=1.0` is length-confounded (sampling divergence inflates it) —
see the [result summary](rocm-7.14/README.md) and
[METHODOLOGY.md §12](METHODOLOGY.md). Raw cells: [matrix-714/](matrix-714/).

Rules:

1. Never edit or relabel the ROCm 7.2.1 cells.
2. Use the same llama.cpp commit, model hashes, prompts, seeds and flags.
3. Record the actual ROCm version in every new cell.
4. Keep failed or unstable cells with evidence rather than silently omitting them.
5. Publish conclusions only after the completion and review gates are satisfied.
