# Changelog

This project follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
No release has been tagged yet; entries remain under **Unreleased** until the
ROCm 7.14 validation track and release gates are complete.

## Unreleased

### Added

- Machine-readable schemas for stack, artifact, public-claim, historical
  benchmark-cell, and community hardware-validation manifests.
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

### Preserved

- Historical ROCm 7.2.1 benchmark cells, including negative and
  non-completing findings.
- ROCm 7.14 and Radeon validation status as pending.
