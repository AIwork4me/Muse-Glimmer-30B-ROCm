# Validation tracks

Benchmark results are separated by evidence status. A newer ROCm label never
replaces an older validated record. **ROCm 7.14.0 is the recommended default**;
ROCm 7.2.1 is the fully-validated historical reference (supplementary).

## ROCm 7.14 gfx1151 — recommended default (scoped project validation)

Status: **GGUF/llama.cpp reduced matrix project-validated (2026-08-13);
optional / not prioritized for v0.1; ROCm 7.14 Muse-Glimmer vLLM validation
pending.** Current proxy results did not justify prioritizing that rebuild; this
is not a global ROCm 7.14 validation or permanent vLLM value claim.

The 19-cell run used AMD's official gfx1151 ROCm 7.14.0 tarball on Ryzen AI
MAX+ PRO 395 / Radeon 8060S. AMD's [ROCm 7.14 release notes](https://rocm.docs.amd.com/en/docs-7.14.0/about/release-notes.html)
list that platform as `gfx1151`. Muse-Glimmer workload behavior, performance,
DFlash, and vision evidence here are independent project results, not AMD
workload-support claims. Mean TPOT deltas were −0.4% at `np=1` and −1.7% at `np=4`; all 17 cells
of the original pass had a lower VmPeak mapped-address-space envelope (mean
−2.8%; of the `np=16` baselines added later, 17gb is +16.1% and dynamic
−1.6% — see the scoped result). DFlash acceptance
rates were similar, with the largest observed difference 1.21 percentage
points. No incident was observed during the six-hour run, but raw system logs
were not retained, so this is not a standalone stability qualification.

See the [result summary](rocm-7.14/README.md) and
[scoped manifest](../../configs/rocm-7.14-gguf-validation.json).

## ROCm 7.2.1 — historical reference (supplementary)

The full validated stack — methodology, the complete benchmark matrix, the
vLLM-vs-llama.cpp head-to-head, and llama-bench — is preserved as immutable
historical evidence:

- Methodology: [METHODOLOGY.md](METHODOLOGY.md)
- Interpretation: [benchmark.md](benchmark.md)
- Immutable raw cells: [matrix/](matrix/)
- Stack definition: [../../configs/validated-stack.json](../../configs/validated-stack.json)
- Artifact hashes: [../../configs/artifact-manifest.json](../../configs/artifact-manifest.json)

The raw matrix contains completed cells and two evidence-based pathological
non-completions. Both are results.

## Rules

1. Never edit or relabel the ROCm 7.2.1 cells.
2. Use the same llama.cpp commit, model hashes, prompts, seeds and flags.
3. Record the actual ROCm version in every new cell.
4. Keep failed or unstable cells with evidence rather than silently omitting them.
5. Publish conclusions only after the completion and review gates are satisfied.

## W7900 (gfx1100) throughput validation

**Study 2 (throughput under load) plus Study 1 (greedy DFlash anchor)**
validated on a single **Radeon PRO W7900** (`gfx1100` / RDNA3, 48 GiB) — a
distinct platform that neither replaces nor relabels the gfx1151 records above.
Three labeled passes: the original community submission (ROCm 7.2.4 image),
an independent reproduction (ROCm 7.2.1, every overlap within ±3%), and a
**full 14/14-cell pass on ROCm 7.14.0**, the recommended default.

- Results & interpretation: [w7900-gfx1100.md](w7900-gfx1100.md)
- Raw cells by ROCm version:
  [7.2.1](hardware-validation/w7900-gfx1100/cells-rocm-7.2.1/) ·
  [7.14.0](hardware-validation/w7900-gfx1100/cells-rocm-7.14.0/)
- Reproduction (two methods — Docker image or bare metal): [`../../scripts/w7900-repro/`](../../scripts/w7900-repro/)
