# Security policy

## Reporting a vulnerability

Do not disclose credentials, exploitable download behavior, or other sensitive
details in a public issue. Use GitHub's private vulnerability-reporting flow
from this repository's **Security** tab when it is available. If that flow is
unavailable, contact the repository owner through the GitHub profile and ask
for a private reporting channel before sharing details.

Include the affected commit, platform, reproduction steps, impact, and any safe
mitigation. Do not attach access tokens, model credentials, or private system
logs.

## Scope and support

Security-sensitive areas include model/download provenance, endpoint handling,
shell command construction, server exposure, dependency pins, and generated
benchmark artifacts. The current default branch is supported on a best-effort
basis; historical benchmark stacks remain evidence, not a promise of security
maintenance for every pinned upstream dependency.

Public, non-sensitive correctness bugs should use the bug-report issue form.
