# New-User One-Pass Reproduction & UX Audit — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reproduce the repo's README Quick start as a strict new user (full cold), judge One-Pass Success, log every friction point with root cause, and land fixes via branch + PR to `master`.

**Architecture:** A live sequential audit (Approach A) in an isolated workspace, followed by a static review, a findings report with a maintainer adjudication gate, then a per-finding TDD fix loop in the dev repo ending in a CI-green PR.

**Tech Stack:** bash, git/GitHub CLI (`gh`), curl, `uv run --no-sync pytest`, shellcheck.

**Spec:** `docs/superpowers/specs/2026-08-14-newuser-one-pass-audit-design.md`

## Global Constraints

- Audit target is **GitHub `master` = `263e9b7ef282e1d123f0188df7f37b894f76e21e`**, cloned fresh from `https://github.com/AIwork4me/Muse-Glimmer-30B-ROCm.git`.
- **Knowledge boundary:** only README + README-linked docs. Consulting a linked doc when stuck is allowed but is logged as a One-Pass breach (finding or breach note).
- **Never touch `/opt/rocm`. Never modify `~/rocm-7.14.0.dev-backup` until Task 11.**
- Fixes must not alter `docs/results/` data or validated `configs/` (`public-claims.json`, `validated-stack.json`, `artifact-manifest.json`), nor pinned llama.cpp/GGUF revisions.
- Fix gates (must be green before PR): `uv run --no-sync shellcheck -x scripts/*.sh scripts/lib/*.sh` and `uv run --no-sync pytest -m "not gpu and not server" -v`.
- Port 8080 must be free before every serve step; kill servers between tasks.
- Long-running steps run backgrounded with output teed to workspace logs.
- Path variables used throughout: `WS="$HOME/Desktop/muse-newuser"`, `CLONE="$WS/Muse-Glimmer-30B-ROCm"`, `DEV="$HOME/Desktop/muse-rocm"`, `BACKUP="$HOME/rocm-7.14.0.dev-backup"`, `LOGS="$WS/logs"`.

**Finding ID scheme** (defined once, used everywhere): `F-NN` sequential; severity `S0` blocker / `S1` workaround / `S2` confusion / `S3` polish; evidence class `LIVE` (observed in the cold run) or `STATIC` (from review). Friction log lives in `$WS/audit-log.md`; consolidated report in `$WS/FINDINGS.md`.

---

### Task 1: P0 — Pre-flight safety and workspace

**Files:**
- Create: `$WS/`, `$WS/logs/`, `$WS/audit-log.md`, `$WS/timings.tsv`

**Interfaces:**
- Consumes: nothing.
- Produces: workspace dirs; empty audit log; ROCm moved to `$BACKUP` (restore path: `mv "$BACKUP" ~/rocm-7.14.0`).

- [ ] **Step 1: Verify baseline is safe to disturb**

```bash
ss -ltn | grep ':8080' || echo "port 8080 free"
pgrep -af 'llama-server|vllm' || echo "no inference servers running"
pgrep -af 'rocm-smi|rocminfo' || echo "no rocm tooling running"
df -h /home/amd | tail -1
uname -r && cmake --version | head -1 && python3 --version && git --version
```

Expected: port free, no servers, ≥100 GiB disk free (Avail column), kernel ≥ 6.16.9. Record the tool versions in `audit-log.md` as the "not bare metal" baseline note.

- [ ] **Step 2: Create workspace and move ROCm aside**

```bash
mkdir -p "$WS/logs"
mv "$HOME/rocm-7.14.0" "$BACKUP"
ls -d "$BACKUP" && ls "$HOME/rocm-7.14.0" 2>/dev/null || echo "old location gone (expected)"
```

Expected: backup dir exists, original path gone.

- [ ] **Step 3: Scaffold the audit log**

```bash
cat > "$WS/audit-log.md" <<'EOF'
# New-user one-pass audit — friction log

Target: GitHub master 263e9b7ef282e1d123f0188df7f37b894f76e21e
Machine baseline (dev machine, NOT bare metal): see P0 notes.

Entry format per executed step:

## [P<phase>.<step>] <command>
- start/end/elapsed:
- expected (from README):
- actual:
- key output:
- new-user mental model:
- friction: none | F-NN (S?, LIVE)

Findings: F-NN — severity — one-line symptom
Breaches (linked-doc consultations): <list>
EOF
printf 'phase\tstart\tend\tseconds\n' > "$WS/timings.tsv"
```

- [ ] **Step 4: Commit nothing (workspace is outside the repo); proceed to Task 2**

---

### Task 2: P1a — Fresh clone of GitHub master

**Files:**
- Create: `$CLONE/` (from GitHub)

**Interfaces:**
- Consumes: `$WS` from Task 1.
- Produces: `$CLONE` at sha `263e9b7…`; `$LOGS/p0-clone.log`.

- [ ] **Step 1: Clone exactly as the README shows**

```bash
cd "$WS"
{ date +%FT%T; git clone https://github.com/AIwork4me/Muse-Glimmer-30B-ROCm.git; rc=$?; date +%FT%T; echo "exit=$rc"; } 2>&1 | tee "$LOGS/p0-clone.log"
```

- [ ] **Step 2: Verify the audited sha**

```bash
git -C "$CLONE" rev-parse HEAD
```

Expected: `263e9b7ef282e1d123f0188df7f37b894f76e21e`. If different — STOP, confirm with maintainer which commit to audit before continuing (record as context, not a finding).

- [ ] **Step 3: Log the step** — append the `[P1.clone]` entry to `audit-log.md` per the scaffold format (expected: clone works; actual: observed; friction: note anything unexpected, e.g. clone size warnings).

---

### Task 3: P1b — README command 1: `install-rocm-7.14.sh` (cold)

**Files:**
- Create: `$LOGS/p1-install.log`, `~/rocm-7.14.0/` (fresh install)

**Interfaces:**
- Consumes: `$CLONE` from Task 2.
- Produces: fresh `~/rocm-7.14.0`; timing row `P1-install`.

- [ ] **Step 1: Run the README command verbatim, timed and teed**

```bash
cd "$CLONE"
set -o pipefail
{ date +%FT%T; bash scripts/install-rocm-7.14.sh; rc=$?; date +%FT%T; echo "exit=$rc"; } 2>&1 | tee "$LOGS/p1-install.log"
```

Expected by README: "installs AMD's gfx1151 ROCm 7.14 archive at `~/rocm-7.14.0` without overwriting `/opt/rocm`", ~1.6 GiB download. **Watch for:** undeclared prerequisites, silent long pauses with no progress, download resume behavior, disk-space checks, error actionability. Every hiccup → `F-NN` entry.

- [ ] **Step 2: Verify the install outcome the README implies**

```bash
ls "$HOME/rocm-7.14.0/bin/rocm-smi" && "$HOME/rocm-7.14.0/bin/rocm-smi" --version
du -sh "$HOME/rocm-7.14.0"
```

- [ ] **Step 3: Append timing row and log entry**

```bash
printf 'P1-install\t<start>\t<end>\t<sec>\n' >> "$WS/timings.tsv"
```

---

### Task 4: P1c — README command 2: bare `00-check-env.sh`

**Files:**
- Create: `$LOGS/p1-envcheck.log`

**Interfaces:**
- Consumes: fresh ROCm install from Task 3.
- Produces: pass/fail observation of the README-verbatim invocation.

- [ ] **Step 1: Run exactly what the README prints (no profile flag)**

```bash
cd "$CLONE"
set -o pipefail
{ date +%FT%T; bash scripts/00-check-env.sh; rc=$?; date +%FT%T; echo "exit=$rc"; } 2>&1 | tee "$LOGS/p1-envcheck.log"
```

**Watch for:** whether the bare invocation checks the GGUF path or defaults elsewhere; whether a passing result is clearly communicated; whether failure output names the fix.

- [ ] **Step 2: Log the entry; record verdict for this sub-step**

---

### Task 5: P1d — README command 3: `gguf-quickstart.sh` (full cold: clone + build + 15.6 GiB download + serve)

**Files:**
- Create: `$LOGS/p1-gguf.log`; `$CLONE/third_party/llama.cpp/`; `$CLONE/models/`

**Interfaces:**
- Consumes: ROCm install (Task 3).
- Produces: a running server on 8080 (or a blocking failure finding).

- [ ] **Step 1: Confirm port free, then launch in background**

```bash
ss -ltn | grep ':8080' && { echo "F-NN: port occupied before first serve"; }
cd "$CLONE"
nohup bash scripts/gguf-quickstart.sh > "$LOGS/p1-gguf.log" 2>&1 &
echo "launched pid $!"
```

**UX note to record:** the README presents this as the last command with the comment `# OpenAI-compatible server: http://127.0.0.1:8080` — a real user's terminal is now occupied by a foreground server with no README guidance to open a second terminal. Backgrounding is the auditor's mechanics; log the blocking-UX observation itself as review data.

- [ ] **Step 2: Poll for milestones (record wall time at each)**

```bash
for i in $(seq 1 240); do
  grep -nE "llama.cpp source|checkout|Configuring|Building object|Built target|fetching .*gguf|verified|Serving on|ERROR|error:" "$LOGS/p1-gguf.log" | tail -3
  grep -q "Serving on" "$LOGS/p1-gguf.log" && { echo "SERVER UP at $(date +%FT%T)"; break; }
  sleep 30
done
```

Timestamp each first occurrence (llama.cpp clone done / cmake configure done / build done / download start / download verified / serving). Expected total: ~15–45 min (build ≤15 min on 32 cores; 15.6 GiB download dominates).

- [ ] **Step 3: On failure instead of serving** — capture the tail (`tail -50 "$LOGS/p1-gguf.log"`), record `F-NN` with severity, then attempt only README-linked recovery (log each as a breach). If unrecoverable: stop Task 5–6, keep evidence, continue at Task 7.

- [ ] **Step 4: Log entries + timing rows** (`P1-build`, `P1-download`)

---

### Task 6: P1e — First completion and Entry-1 verdict

**Files:**
- Create: `$LOGS/p1-curl.log`

**Interfaces:**
- Consumes: running server from Task 5.
- Produces: Entry-1 One-Pass verdict (✅/🟡/❌) recorded in `audit-log.md`.

- [ ] **Step 1: Health check**

```bash
curl -s -m 10 http://127.0.0.1:8080/health; echo
```

- [ ] **Step 2: One chat completion, timed**

```bash
curl -s -m 300 -w '\nttfb=%{time_starttransfer}s total=%{time_total}s\n' \
  http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Reply with exactly: OK"}],"max_tokens":16}' \
  | tee "$LOGS/p1-curl.log"
```

Expected: JSON with non-empty `choices[0].message.content`.

- [ ] **Step 3: Record Entry-1 verdict** using the spec's three-level rule (zero interruptions / zero linked-doc consultations / zero non-README knowledge = ✅; S2/S3 present = 🟡; any S0/S1 = ❌).

- [ ] **Step 4: Stop the server**

```bash
pkill -INT -f llama-server; sleep 8; pgrep -af llama-server || echo "server stopped"
```

---

### Task 7: P2 — `quickstart.sh` wrapper (warm path)

**Files:**
- Create: `$LOGS/p2-wrapper.log`

**Interfaces:**
- Consumes: warm clone (models + build present).
- Produces: Entry-2 verdict; observations on the plan display and reuse messaging.

- [ ] **Step 1: Run the wrapper interactively-piped, backgrounded**

```bash
cd "$CLONE"
printf 'y\n' | nohup bash scripts/quickstart.sh > "$LOGS/p2-wrapper.log" 2>&1 &
for i in $(seq 1 60); do grep -q "Serving on" "$LOGS/p2-wrapper.log" && break; sleep 10; done
grep -nE "ROCm:|Model download|ROCm download|Continue|already present|not needed|reuse|fingerprint|Serving on" "$LOGS/p2-wrapper.log"
```

**Watch for:** does the plan print the promised sizes and "already present / not needed" reuse lines? Is the y/N prompt answerable in a plain terminal? (Piped `y` is mechanics; note prompt UX.)

- [ ] **Step 2: Confirm serving, curl once, stop server** (repeat Task 6 Steps 1–2 pattern into `$LOGS/p2-curl.log`, then `pkill -INT -f llama-server`).

- [ ] **Step 3: Record Entry-2 verdict + timing row `P2-wrapper`**

---

### Task 8: P3 — `WITH_DFLASH=1` smoke

**Files:**
- Create: `$LOGS/p3-dflash.log`, `$LOGS/p3-curl.log`; downloads `dflash-kquant.gguf` (~1.5 GiB)

**Interfaces:**
- Consumes: warm clone.
- Produces: Entry-3 verdict; confirmation speculative args are in effect.

- [ ] **Step 1: Launch with the flag, backgrounded**

```bash
cd "$CLONE"
WITH_DFLASH=1 nohup bash scripts/gguf-quickstart.sh > "$LOGS/p3-dflash.log" 2>&1 &
for i in $(seq 1 90); do grep -q "Serving on" "$LOGS/p3-dflash.log" && break; sleep 10; done
grep -inE "dflash|draft|spec" "$LOGS/p3-dflash.log" | head -20
```

Single request only (no `-np`; the README-warned c16 pathology is out of scope).

- [ ] **Step 2: One completion (Task 6 Step 2 pattern) teed to `$LOGS/p3-curl.log`; verify content non-empty**

- [ ] **Step 3: Stop server; record Entry-3 verdict + timing row `P3-dflash`**

---

### Task 9: OPTIONAL probes — only if time allows, ≤3, each is a finding source

**Files:**
- Create: `$LOGS/probe-*.log`

**Interfaces:**
- Consumes: warm clone.
- Produces: `F-NN` findings on error quality/recovery.

- [ ] **Probe A (port-occupied error):** occupy 8080 (`python3 -m http.server 8080 --bind 127.0.0.1 &`), run `bash scripts/gguf-quickstart.sh` in foreground briefly, capture the error text, release the port (`kill %1`). Judge: does it name the conflict and the `PORT=` remedy?
- [ ] **Probe B (Ctrl-C rerun):** start the quickstart, Ctrl-C (SIGINT) mid-download/build, rerun; does the rerun resume/reuse gracefully or restart from zero?
- [ ] **Probe C (per-clone isolation):** `git clone "$DEV" "$WS/clone2"` (local clone is mechanics; the user-visible question stands), start `bash scripts/gguf-quickstart.sh` in `clone2`, observe whether it begins re-downloading the 15.6 GiB model into `clone2/models/`; if yes record `F-NN`, abort the download, remove `clone2`.

---

### Task 10: Static UX review (bare-OS gap coverage)

**Files:**
- Read: `README.md`, `scripts/install-rocm-7.14.sh`, `scripts/00-check-env.sh`, `scripts/gguf-quickstart.sh`, `scripts/lib/rocm.sh`, `scripts/lib/llama_build.sh`, `docs/strix-halo-setup.md`, `docs/troubleshooting.md`
- Produces: `STATIC`-class findings.

- [ ] **Step 1: Prerequisite cross-check** — list every command/condition the scripts actually require (`grep -nE 'command -v|sudo|apt|dpkg|systemctl|modprobe' scripts/*.sh scripts/lib/*.sh`); compare against the README's declared list ("git, cmake, curl, Python 3, and a gfx1151-capable HIP toolchain"). Any undeclared requirement → `F-NN (STATIC)`.
- [ ] **Step 2: Bare-OS guidance reachability** — from README only: can a user missing `cmake` find an install command within two clicks? Is the kernel ≥ 6.16.9 / UMA / BIOS guidance linked from the Quick start path? Gaps → `F-NN (STATIC)`.
- [ ] **Step 3: Error-path walkthrough** — for each `exit 1` in the three entry scripts, judge: does the message state what failed, why, and the next action? Non-actionable → `F-NN (STATIC)`.

---

### Task 11: P5 — Environment restore decision + evidence preservation

**Files:**
- Modify: `~/rocm-7.14.0` / `$BACKUP` (keep exactly one, decision below)

**Interfaces:**
- Consumes: audit complete (Tasks 3–8 evidence in `$LOGS`).
- Produces: stable environment; preserved evidence.

- [ ] **Step 1: Equivalence check**

```bash
du -sb "$HOME/rocm-7.14.0" "$BACKUP"
diff <(cd "$HOME/rocm-7.14.0" && find . -type f | sort) <(cd "$BACKUP" && find . -type f | sort) | head
```

- [ ] **Step 2: Keep the fresh install (it is what the audit validated); retain the backup untouched until the PR merges** (disk cost ~1.6 GiB is acceptable). Revisit only if the maintainer objects.

- [ ] **Step 3: Preserve evidence** — `$WS/` (logs, audit-log.md, timings.tsv, FINDINGS.md from Task 12) stays in place; the `$CLONE` stays as-is until the PR merges (it is the warm-regression host for Task 13).

---

### Task 12: P6 — FINDINGS.md and the maintainer gate

**Files:**
- Create: `$WS/FINDINGS.md`

**Interfaces:**
- Consumes: `audit-log.md`, probe logs, static review notes.
- Produces: the adjudicated findings table that Task 13 consumes — **GATE: do not start Task 13 before the maintainer answers.**

- [ ] **Step 1: Compile the report**

```markdown
# One-Pass Audit — Findings (v0.1.0, master 263e9b7)

## Verdicts
| Entry | Verdict | Notes |
|---|---|---|
| 4-command Quick start | ✅/🟡/❌ | … |
| quickstart.sh wrapper | ✅/🟡/❌ | … |
| WITH_DFLASH smoke | ✅/🟡/❌ | … |

## Findings
| ID | Sev | Class | Phase | Symptom | Root cause | Evidence | Fix proposal |
|---|---|---|---|---|---|---|---|
| F-01 | S2 | LIVE | P1b | … | undeclared prerequisite | logs/p1-install.log:42 | add prereq line to README §Quick start |

## Breaches (linked-doc consultations)
…
```

- [ ] **Step 2: Present to the maintainer.** Message: verdicts, S0/S1 first, then the full table; ask to confirm severity and fix scope (S0/S1 blocking, S2 fix-by-default, S3 low-cost only). **Wait for the answer.**

---

### Task 13: P7a — Fix branch and per-finding fix loop

**Files:**
- Modify: `$DEV` on new branch `fix/ux-audit-v010` off `master`; files per finding.

**Interfaces:**
- Consumes: adjudicated FINDINGS.md (Task 12).
- Produces: one commit per finding, each mapping to an `F-NN`.

- [ ] **Step 1: Create the branch in the DEV repo (not the audit clone)**

```bash
cd "$DEV" && git status --porcelain && git switch -c fix/ux-audit-v010 master
```

- [ ] **Step 2: For each approved finding, run this loop** (docs-only findings skip the test steps):

1. Write the failing test (for script-UX fixes) in the matching existing module (`tests/test_scripts.py`, `tests/test_env.py`, `tests/test_quickstart_wrapper.py`, …) mirroring neighboring test style.
2. Run it, confirm it fails: `uv run --no-sync pytest tests/<module>::test_<name> -v`.
3. Make the minimal fix (script message / preflight check / README wording). Docs-only example pattern — README prerequisite line, verified by the existing link gate: `uv run --no-sync pytest tests/test_markdown_links.py -v`.
4. Re-run the test, confirm pass.
5. Commit: `git commit -m "fix(ux): F-NN <one-line> (<root-cause class>)"`.

Worked example of a docs finding (illustrates the loop; actual content comes from FINDINGS.md): README "Requirements" lists tools but gives no bare-OS install command → add under Quick start: `sudo apt install git cmake curl python3`-equivalent line matching the distro actually validated, link `docs/strix-halo-setup.md` from the Quick start section, verify with the link test above.

- [ ] **Step 3: Push and open the PR**

```bash
git push -u origin fix/ux-audit-v010
gh pr create --base master --title "fix(ux): new-user one-pass audit fixes" --body-file <(render FINDINGS.md table + per-fix mapping)
```

PR body template: verdicts table; findings table (ID/Sev/Symptom/Root cause/Fix commit); evidence pointers to workspace logs; "no validated data touched" attestation.

---

### Task 14: P7b — Gates, live regression, wrap-up

**Files:**
- Modify: nothing new; verification only.

**Interfaces:**
- Consumes: fix branch (Task 13), warm `$CLONE` (Task 11).
- Produces: green CI; regression evidence; final summary.

- [ ] **Step 1: Local gates (mirror CI exactly)**

```bash
cd "$DEV"
uv run --no-sync shellcheck -x scripts/*.sh scripts/lib/*.sh
uv run --no-sync pytest -m "not gpu and not server" -v
```

- [ ] **Step 2: Live warm regression on the fixed scripts** — check the fix branch out inside the audit clone and rerun the touched entry point:

```bash
git -C "$CLONE" fetch "$DEV" fix/ux-audit-v010
git -C "$CLONE" checkout FETCH_HEAD
cd "$CLONE" && nohup bash scripts/gguf-quickstart.sh > "$LOGS/regress.log" 2>&1 &
# poll "Serving on", curl once (Task 6 Step 2 pattern), pkill -INT -f llama-server
```

- [ ] **Step 3: Watch CI on the PR** (`gh pr checks --watch`), fix if red.

- [ ] **Step 4: Final summary to the maintainer** — verdicts, findings → fixes mapping, PR link, leftover S3 issue candidates, cleanup plan for `$WS`/`$BACKUP` after merge.

---

## Self-Review

- **Spec coverage:** P0→Task 1; P1→Tasks 2–6; P2→Task 7; P3→Task 8; P4→Tasks 6/8 (curl evidence + consolidated in Task 12 verdicts); P5→Task 11; P6→Task 12; P7→Tasks 13–14; optional probes→Task 9; static review→Task 10; protocol/logging→Global Constraints + Task 1 scaffold. ✅
- **Placeholder scan:** no TBD/TODO; every command is executable as written except `<start>/<end>/<sec>` timing fills and FINDINGS.md row content, which are audit outputs by design (data flow, not placeholders). ✅
- **Consistency:** `F-NN` scheme, `$WS`/`$CLONE`/`$DEV`/`$BACKUP`/`$LOGS` paths, and the three-entry verdict taxonomy are defined once in Global Constraints and used identically in all tasks. ✅
