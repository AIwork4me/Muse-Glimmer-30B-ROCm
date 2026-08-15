# Machine-readable schemas

These JSON Schema Draft 2020-12 files validate authoritative manifests and
committed evidence without rewriting historical cells.

- `validated-stack.schema.json`: the historical ROCm 7.2.1 reference stack.
- `rocm-7.14-gguf-validation.schema.json`: the scoped 19-cell ROCm 7.14
  GGUF/llama.cpp validation, provenance and evidence boundary.
- `artifact-manifest.schema.json`: pinned model artifacts and checksums.
- `public-claims.schema.json`: public platform and per-track validation status.
- `benchmark-cell-v1.schema.json`: the committed ROCm 7.2.1 and 7.14 cell
  shape, including non-completing pathological cells.
- `hardware-validation.schema.json`: future community evidence submissions.

CI validates all 21 historical cells and all 19 scoped ROCm 7.14 cells against
the descriptive v1 schema. It also verifies the 7.14 SHA256 inventory. No raw
cell is normalized or migrated.

A future benchmark schema v2 may add a run ID, full repository and engine SHAs,
model revision, artifact-manifest identity and harness version. Introducing it
requires an explicit versioned protocol; this pass does not change benchmark
generation.
