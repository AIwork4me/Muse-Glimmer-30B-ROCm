# Contributing

Contributions are welcome when they improve reproducibility, RDNA coverage, or
the accuracy of the adaptation record. “It worked on my GPU” is useful as a
lead; a validation claim needs enough evidence for another developer to audit
it.

## Development checks

Use Python 3.12 and install the locked, CPU-only CI environment:

```bash
uv sync --only-group ci --locked
uv run --no-sync pytest -m "not gpu and not server" -v
bash -n scripts/*.sh scripts/lib/*.sh
uv run --no-sync shellcheck -x scripts/*.sh scripts/lib/*.sh
python scripts/check_claim_consistency.py
```

GPU and live-server tests are intentionally excluded from hosted CI. Run the
relevant marked tests locally when your change touches those paths and report
the exact command and result in the pull request.

Fast CI deliberately omits the project dependency set, so routine pull requests
do not download the TheRock/ROCm runtime. The separate
`dependency-smoke.yml` workflow runs `uv sync --dev --locked` when
`pyproject.toml`, `uv.lock`, or that workflow changes, and is also manually
dispatchable. Use that full sync locally when changing dependency definitions.

JSON Schema files under `schemas/` describe the manifests and historical
benchmark v1 evidence. Schema changes must remain compatible with committed
pathological/non-completing cells; do not rewrite evidence to fit a schema.

## Evidence rules

- Do not edit historical raw records in `docs/results/matrix/` to represent a
  new stack. Add a separate validation track.
- Do not claim hardware as validated without raw logs/artifacts and the fields
  in [docs/hardware-validation.md](docs/hardware-validation.md).
- Record failures, aborts, and performance regressions. Negative results are
  results.
- Pin source commits and model revisions. Record size and SHA256 where an
  artifact was actually available; otherwise leave verification pending.
- Separate methodology-aligned comparisons from original workloads, and state
  material differences in engine, precision, prompts, and measurement.
- Never include model weights, access tokens, private logs, or personal data.

## Pull requests

Keep changes focused. Explain the platform, exact commands, evidence, benchmark
methodology impact, and documentation changes. Update the validated stack or
artifact manifest only when the new value is independently verifiable.

Hardware validation submissions should start with the hardware-validation issue
form and attach or link the evidence bundle described in
[docs/hardware-validation.md](docs/hardware-validation.md). New machine-readable
submissions should follow
[`schemas/hardware-validation.schema.json`](schemas/hardware-validation.schema.json).
