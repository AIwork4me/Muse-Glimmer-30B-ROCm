# Changelog

This project follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
The v0.1.0 entry was dated after its initial local and hosted release-candidate
gates passed. Tags and GitHub Releases are created only after final maintainer
approval.

## [Unreleased]

### Changed

- README restructured for first-visit readability: **what you get,
  performance highlights and the quick start now lead**; the detailed
  explanations (prerequisites with per-distro tool installs, installer
  internals, download sizes, the reproducibility contract and env overrides)
  moved to a new [docs/getting-started.md](docs/getting-started.md) page that
  the README links, and a documentation map was added. No claim changed: the
  three README generated blocks, the TPOT sentence, the known-good/bad table
  and the claims-hygiene caveats are preserved verbatim or linked to their
  existing homes (the memory-terminology explainer was redundant with
  METHODOLOGY §5, which the docs map now points at). The docs-UX tests that
  pinned the moved content were updated to its new location, and the
  hardware-validation README anchor now targets `#requirements`.

### Added

- The c=16 + DFlash pathology's deeper root cause was isolated via controlled
  diagnostic probes (fresh server per probe, bounded request waves) and
  **reported upstream as
  [ggml-org/llama.cpp#27117](https://github.com/ggml-org/llama.cpp/issues/27117)**:
  the draft model's predictions are corrupted per slot once roughly 8+ sequences
  draft concurrently — the trigger is the concurrently-drafting sequence count,
  not the server's `-np` configuration (an `-np 16` server with 4 in-flight
  requests reproduces `-np 4` bit-for-bit) and not the batch token count.
  `--spec-draft-n-max 1` avoids the corruption (0.83–0.92 acceptance at
  `-np 16`, ~2.2× the no-spec baseline in the probes). Reproduced on llama.cpp
  master `0177dcc` and on both ROCm runtimes. README, troubleshooting,
  methodology, benchmark and the scoped 7.14 result now link the issue; the
  two deferred `np=16` DFlash cells stay deferred pending an upstream fix.
  The probes are diagnostic records carried by the upstream issue, not matrix
  cells.
- The ROCm 7.14 GGUF matrix gains its 18th and 19th cells: **both `study2
  np=16` baselines**, measured 2026-08-15 with the corrected benchmark client
  — 17gb 36.97 tok/s aggregate vs 7.2.1's 34.47 (+7.3%) and dynamic 32.01
  vs 31.05 (+3.1%), TTFT within ~7%. Scope statements, the validation
  manifest, its schema, and the claim-consistency generator were updated
  together; the two `np=16` DFlash cells remain deferred. A GPU-concurrency
  overlap during the dynamic cell's first repetition is disclosed in the
  scoped result (median-of-five unaffected).
- The c=16 + DFlash pathology is now evidenced on 7.14 as well: a bounded
  16×48 probe completed healthy (16.9 tok/s, 19.3% acceptance) but the
  full-fidelity 17gb attempt decayed from ~74 to ~35 prompt-tokens/min and
  was aborted at ~2 h. The probe record is committed beside the matrix;
  the warning not to combine DFlash with `-np 16` now covers both runtimes.
- A single-user `llama-bench` flash-attn micro-sweep (exclusive GPU,
  both weights, raw records committed beside the matrix): `-fa on` is a
  consistent +1.7…+2.8% decode win at `np=1` on gfx1151 and `-ub` is
  insensitive for decode. Documented as descriptive evidence in the scoped
  result; the validated matrix flags are unchanged.

### Fixed

- The benchmark client now parses the SSE stream by newline framing instead
  of treating each raw HTTP chunk as one event. At high concurrency aiohttp
  delivers several coalesced events per chunk, and the old parser silently
  dropped them, undercounting tokens by orders of magnitude (real incident:
  an `np=16` cell recorded 96 tokens against ~173k server-side; the corrupt
  record was quarantined, not published). Fixed-client and legacy-client
  numbers are identical at `np=1`/`np=4`, where chunks rarely coalesce.

## [0.1.1] - 2026-08-15

### Added

- A README "Verify it works" quick-start block: the exact health and chat
  completion requests to run against the served model, with the
  reasoning-first `max_tokens` guidance that keeps a first answer non-empty.
- Per-distro host-tool install one-liners in the README requirements section,
  matching the per-distro hints the scripts print when a tool is missing.
- Documentation for `MODEL_DEST`, the knob that stores the ~15.6 GiB model
  outside the clone so one hash-verified download is reused across clones.

### Changed

- The ROCm 7.14 installer deletes the tarball after a successful install
  instead of leaving ~1.6 GiB behind in `/tmp`; a user-pre-placed
  `ROCM714_ARCHIVE` is verified, used, and left in place.
- DFlash speculative decoding requests `--spec-draft-n-max 15` (previously
  16): upstream drafts at most `block_size - 1` tokens and clamps higher
  requests with a warning line at every server start.
- The GGUF quickstart gates on the serving port before any setup work, so a
  busy or unusable `PORT` refuses in well under a second instead of after the
  fetch/build chain.

### Fixed

- Fresh-clone cold start no longer dead-ends: the quickstart now checks out
  the pinned llama.cpp ref immediately after its own no-checkout clone, so
  the dirty-tree guard can no longer misread that empty clone as every
  tracked file staged-deleted and refuse on every cold start.
- The ROCm 7.14 installer exits 0 after a successful install again; the
  version tail's `hipcc --version | head -1` pipeline used to die with exit
  141 (SIGPIPE) under `set -o pipefail`.
- The installer and quickstart refuse up front when the target filesystems
  cannot hold the download and the extracted tree, stating required vs
  available GiB and the escape hatches, instead of dying mid-install with a
  raw "No space left on device".
- The quickstart's port probe no longer leaks a raw Python traceback before
  its busy-port error, and reports a non-numeric `PORT` as unusable rather
  than in use.
- Dirty llama.cpp checkout refusals state what changed and give concrete
  recovery steps instead of dead-ending.
- `00-check-env.sh` failure exits state expected vs observed values and the
  next action.
- `00-check-env.sh` itemizes the quickstart's required host tools, failing
  with per-distro install hints when one is missing, and guards for a missing
  python3 before first use.
- `00-check-env.sh` pool checks label their thresholds as GPU-visible memory,
  so the README's ~20 GiB disk figure cannot be conflated with the
  GPU-visible envelope.

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
