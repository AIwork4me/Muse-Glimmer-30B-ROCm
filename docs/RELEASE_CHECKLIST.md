# Release checklist

Use this checklist to prepare a release; it does not authorize creating a tag
or GitHub Release. Record the exact commit and commands used to close each gate.

## Evidence boundary

- [ ] Historical raw benchmark files are unchanged from their accepted commits.
- [ ] Negative, aborted, and pathological results remain visible.
- [ ] No partial or incomplete forward-validation data is included.
- [ ] ROCm 7.14 remains pending unless a complete evidence set has passed review.
- [ ] Hardware status matches accepted evidence; planned Radeon targets have not
  been promoted by prose alone.
- [ ] Benchmark and performance wording has been reviewed against the raw record.
- [ ] No benchmark workload or generation protocol changed during an active run.

## Reproducibility

- [ ] `configs/validated-stack.json` passes its schema and matches public claims.
- [ ] `configs/artifact-manifest.json` passes its schema.
- [ ] Every published artifact size and SHA256 was checked against local bytes.
- [ ] Engine commits and model revisions are full immutable revisions.
- [ ] Default download endpoints and all overrides are documented.
- [ ] A clean clone can follow the documented fast path.
- [ ] The dependency-smoke workflow resolves and imports the locked TheRock
  runtime.

## Verification

- [ ] Fast CI is green.
- [ ] `uv sync --only-group ci --locked` succeeds without installing
  torch, torchvision, ROCm, or TheRock packages.
- [ ] `uv run --no-sync pytest -m "not gpu and not server" -v` passes.
- [ ] `bash -n scripts/*.sh scripts/lib/*.sh` passes.
- [ ] `uv run --no-sync shellcheck -x scripts/*.sh scripts/lib/*.sh` passes.
- [ ] JSON Schema, JSON, YAML, manifest, and claim-consistency checks pass.
- [ ] `git diff --check` passes.
- [ ] Modified Markdown links have been checked.
- [ ] Security/privacy scan finds no credentials, model weights, private logs,
  hostnames, or personal data.

## Publication

- [ ] `CHANGELOG.md` describes the intended release scope.
- [ ] README hardware and validation-track tables were reviewed.
- [ ] The release version and tag have been chosen deliberately.
- [ ] `CITATION.cff` version matches the chosen release.
- [ ] Release notes distinguish historical evidence from the current validation
  track.
- [ ] The release commit is signed or otherwise traceable under project policy.
- [ ] Required branch protections and review approvals are satisfied.
- [ ] Tag and GitHub Release creation are approved by a maintainer.

Do not publish while any evidence-boundary gate is unresolved.
