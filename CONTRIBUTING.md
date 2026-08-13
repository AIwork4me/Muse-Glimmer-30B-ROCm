# Contributing

Contributions are welcome when they improve reproducibility, RDNA coverage, or
the accuracy of the adaptation record. “It worked on my GPU” is useful as a
lead; a validation claim needs enough evidence for another developer to audit
it.

## Development checks

Use Python 3.12 and install the locked test environment:

```bash
uv sync --dev --locked
uv run pytest -m "not gpu and not server" -v
bash -n scripts/*.sh scripts/lib/*.sh
shellcheck scripts/*.sh scripts/lib/*.sh
```

GPU and live-server tests are intentionally excluded from hosted CI. Run the
relevant marked tests locally when your change touches those paths and report
the exact command and result in the pull request.

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
[docs/hardware-validation.md](docs/hardware-validation.md).
