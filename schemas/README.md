# Machine-readable schemas

These JSON Schema Draft 2020-12 files validate the repository's authoritative
manifests and committed historical evidence.

- `validated-stack.schema.json`: the validated historical/reference stack.
- `artifact-manifest.schema.json`: pinned model artifacts and checksums.
- `public-claims.schema.json`: the small set of public status claims rendered
  in README.
- `benchmark-cell-v1.schema.json`: the existing ROCm 7.2.1 matrix cell format,
  including non-completing pathological cells.
- `hardware-validation.schema.json`: the submission shape for future community
  hardware evidence.

The benchmark v1 schema is descriptive, not a migration. CI validates only the
committed `docs/results/matrix/cell-*.json` files and never rewrites them.

A future benchmark schema v2 may add a run ID, full repository and engine SHAs,
model revision, artifact-manifest identity, and harness version. That format
must be introduced only after the active ROCm 7.14 validation track completes;
this release-readiness pass does not change benchmark generation.
