# Changelog

This project follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
The v0.1.0 entry was dated after its initial local and hosted release-candidate
gates passed. Tags and GitHub Releases are created only after final maintainer
approval.

## Unreleased

## [0.1.0] - 2026-08-14

### Added

- Machine-readable schemas for stack, artifact, public-claim, historical
  benchmark-cell, and community hardware-validation manifests.
- A scoped ROCm 7.14 GGUF/llama.cpp validation manifest and SHA256 evidence
  inventory, explicitly covering 17 of 21 planned cells.
- Schema validation for both historical and ROCm 7.14 benchmark cells.
- Automated public-claim consistency checks.
- A lightweight hosted-CI dependency group and a separate TheRock/ROCm
  dependency-resolution smoke workflow.
- Release checklist and software citation metadata.

### Changed

- CI actions are pinned to immutable revisions and updated through Dependabot.
- ShellCheck follows sourced project libraries and lints them as explicit inputs.
- Public benchmark language distinguishes observations, smoke evidence, and
  evidence-supported mechanisms from uncollected profiling or quality evidence.
- The maintainer handoff is now a durable source-of-truth map rather than a
  machine-local work-session snapshot.
- Cross-ROCm comparison output now renders latency in the correct units, fails
  closed on malformed/duplicate cells, and leads with TPOT rather than aggregate tok/s.
- Reframed **ROCm 7.14.0 as the recommended default** — primary headline + default
  install/build (`scripts/install-rocm-7.14.sh`, `gguf-quickstart.sh` defaults to
  `~/rocm-7.14.0`). The llama.cpp/GGUF path is the focus for single-user gfx1151;
  vLLM/BF16 is optional / not prioritized for v0.1, with ROCm 7.14 Muse-Glimmer
  validation pending and historical 7.2.1 validation preserved. 7.2.1 is the
  supplementary historical reference. `public-claims.json` marks the 7.14 track
  `recommended: true` (asserted by the consistency checker).

### Preserved

- Historical ROCm 7.2.1 benchmark cells, including negative and
  non-completing findings.
- All committed ROCm 7.14 raw summary cells; the GGUF track is scoped to 17/21,
  while current rocBLAS BF16-GEMM proxy results did not justify prioritizing a
  ROCm 7.14 Muse-Glimmer vLLM rebuild for v0.1.
- Radeon dGPU validation status as planned.
