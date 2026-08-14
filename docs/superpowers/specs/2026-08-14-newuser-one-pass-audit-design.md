# New-User One-Pass Reproduction & UX Audit — Design Spec

- **Date:** 2026-08-14
- **Status:** Approved (brainstorming phase) → awaiting implementation plan
- **Owner:** maintainer (AMD ROCm inference)
- **Approach chosen:** **A** — sequential live cold reproduction as a strict new user
- **Builds on:** v0.1.0 release candidate state (master `263e9b7` as the audit target)

---

## 1. Overview & problem statement

The repository ships a README-driven Quick start promising that a new user on a
gfx1151 (Strix Halo) machine can go from `git clone` to an OpenAI-compatible
server in one pass:

```bash
git clone https://github.com/AIwork4me/Muse-Glimmer-30B-ROCm.git
cd Muse-Glimmer-30B-ROCm
bash scripts/install-rocm-7.14.sh
bash scripts/00-check-env.sh
bash scripts/gguf-quickstart.sh
```

plus an optional one-command wrapper (`scripts/quickstart.sh`) and two optional
feature flags (`WITH_MMPROJ`, `WITH_DFLASH`).

This has never been tested end-to-end from a genuinely cold state by someone
restricted to README knowledge only. All prior runs happened on the development
machine where ROCm, models, and llama.cpp builds already existed — which is
exactly the condition that hides new-user friction.

The task: reproduce the full project as a new user in a fresh folder, determine
whether **One-Pass Success** is achievable, log every friction point with its
root cause, and fix the findings.

## 2. Audit decisions (confirmed with maintainer)

| Decision | Choice |
|---|---|
| Audit target | **GitHub `master` (`263e9b7`)** — what a new user clones today |
| Depth | **Full cold reproduction** — ROCm installer actually runs (existing `~/rocm-7.14.0` moved aside), 15.6 GiB model actually downloads, llama.cpp builds from scratch |
| Scope | Both documented entries (4-command sequence + `quickstart.sh` wrapper), then `WITH_DFLASH=1` smoke test; `WITH_MMPROJ` static review only |
| Fix landing | Branch off `master` (`fix/ux-audit-v010`) → PR back to `master`, CI green; `release/v0.1.0-rc` rhythm unaffected until merge |

## 3. New-user simulation protocol

**Knowledge boundary.** During execution, only information available in the README
and documents the README links may be used. Consulting linked docs (e.g.
troubleshooting) when stuck is allowed but is recorded as a **One-Pass breach
data point**. Prior maintainer knowledge of the repo internals must not be used
to bypass anything.

**Execution discipline.**

- Commands are typed verbatim as the README states them (including the bare
  `00-check-env.sh` — whether its default profile is correct is itself an audit
  point).
- No pre-installation or pre-download of anything the README does not declare.
- Per-step log: timestamp / command / expected vs actual / wait duration /
  new-user mental model ("what would the user think right now").

**Friction log format** (one row per finding):

| Field | Content |
|---|---|
| ID / severity | S0 blocker (cannot continue) · S1 workaround needed · S2 confused but passed · S3 polish |
| Symptom | What actually happened |
| Root cause | One of: missing docs · non-actionable error · undeclared prerequisite · flow ordering · environment assumption |
| Fix proposal | The corresponding change |

**Static review supplement** (for what cannot be simulated live): README
prerequisite claims vs actual script dependencies (`tar`, `sudo`, disk-space
checks), guidance for a bare-OS machine missing `cmake`, reachability of
kernel/UMA/BIOS guidance.

## 4. Execution phases

Workspace: `~/Desktop/muse-newuser/` (the clone lands at
`~/Desktop/muse-newuser/Muse-Glimmer-30B-ROCm/`).

| Phase | Content | Key observations |
|---|---|---|
| P0 | Move `~/rocm-7.14.0` → `~/rocm-7.14.0.dev-backup`; create workspace; confirm no process holds the old ROCm | — |
| P1 | Clone GitHub `master` → run the README 4-command sequence verbatim, cold | Per-step wall clock; quality of download/build/verify progress feedback; error actionability |
| P2 | Same clone, run `quickstart.sh` (interactive confirm) | Wrapper UX; the "reuses matching assets on reruns" promise |
| P3 | `WITH_DFLASH=1 bash scripts/gguf-quickstart.sh`, verified with a single request (no `-np` flag; the README-warned c16 pathology is out of scope) | Drafter download; speculative-decoding args in effect |
| P4 | `curl` the 8080 chat-completions endpoint until a completion returns; compare against the repo's own smoke definition | End-to-end success determination |
| P5 | Handle the freshly installed ROCm dir (verify equivalence with backup, keep one); keep workspace logs as evidence | — |
| P6 | Findings report: severity, root cause, evidence, fix proposal | Deliverable 1 |
| P7 | Fixes on `fix/ux-audit-v010` → PR to `master`, CI green (including `check_claim_consistency.py`) | Deliverable 2 |

Optional probes (only if time allows, ≤3): error quality when the port is
occupied; rerun behavior after Ctrl-C; whether a second clone re-downloads the
15.6 GiB model (per-clone `models/` isolation).

## 5. One-Pass Success criteria

**One-Pass Success** = from `git clone` to a returned completion at
`http://127.0.0.1:8080/v1/chat/completions` with: zero error interruptions,
zero troubleshooting consultation, zero non-README knowledge.

Three-level verdict:

- ✅ **One-Pass** — all of the above hold
- 🟡 **One-Pass with friction** — completed, but S2/S3 confusion points occurred
- ❌ **Failed** — any S0/S1 occurred

Each documented entry (4-command path, wrapper, DFlash) is judged separately.

## 6. Fix boundary and quality gates

**Fixes may touch exactly two thing classes:**

1. **Docs** — prerequisite declarations in README/install docs, missing guidance,
   misleading wording.
2. **Script UX** — error-message actionability, preflight checks (e.g. disk
   space, missing dependency commands with install hints), progress feedback.

**Explicitly off-limits:**

- Any benchmark data under `docs/results/`
- Validated claims in `configs/` (`public-claims.json`, `validated-stack.json`,
  `artifact-manifest.json`)
- The pinned/validated llama.cpp and GGUF revisions
- Anything that changes the meaning of a published number

**Quality gates:**

- Every fix maps to a finding ID + root cause, tabulated in the PR description
- Full test suite green + `check_claim_consistency.py` passes (CI enforces)
- At least one live regression check of a fixed critical path (warm-path
  verification is sufficient; no full cold rerun required)

**Severity → handling:**

- S0/S1: must fix; PR blockers
- S2: must fix within this scope unless fix risk exceeds benefit (then
  documented in the report with rationale)
- S3: fix the low-cost ones; file the expensive ones as issue candidates instead
  of forcing them into this PR

## 7. Risks & limitations

| Risk | Mitigation |
|---|---|
| Dev machine is not bare metal (cmake/git/kernel already compliant) | Acknowledged limitation; static review covers the gap; report separates "live-verified" from "statically-inferred" evidence |
| Network variance affects download duration | Durations are qualitative only; download-UX judgments (progress, resume, errors) are unaffected |
| Audit fails midway after P0 moved the ROCm dir | Backup dir stays untouched; `mv` back restores instantly; `/opt/rocm` is never involved |
| ~17 GiB model copy in the fresh clone | 1.5 T free disk; keep-or-clean decided in P5 |
| Long download/build waits | Background tasks; no blocking of interaction |
| Auditor bias (the auditor knows the repo) | Protocol §3 enforced; raw logs preserved as evidence; findings presented to the maintainer for severity adjudication before fixing |
