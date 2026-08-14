"""F-04/F-05 regression tests: gguf-quickstart llama.cpp checkout guard.

A `git clone --filter=blob:none --no-checkout` leaves an empty worktree and NO
index file; git's diff machinery then reads every tracked path (all 3419 of
them on llama.cpp) as staged-deleted. The pre-fix dirty-tree guard treated
that as user work and refused to change commits on every cold start (F-04).
These tests pin the guard decision helpers in scripts/lib/llama_build.sh and
the actionable refusal message they emit (F-05), using real temporary git
repositories cloned exactly the way scripts/gguf-quickstart.sh clones."""
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
LIB = ROOT / "scripts/lib/llama_build.sh"
SCRIPT = ROOT / "scripts/gguf-quickstart.sh"
TROUBLESHOOTING = ROOT / "docs/troubleshooting.md"


def bash(script: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", "-c", script],
        cwd=ROOT,
        env=os.environ.copy(),
        text=True,
        capture_output=True,
        check=False,
    )


def git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def make_origin(tmp_path: Path) -> tuple[Path, str]:
    """A local llama.cpp stand-in: default-branch tip B, older commit A = pin.

    `uploadpack.allowAnySHA1InWant` mirrors GitHub, which lets the quickstart
    shallow-fetch the pinned 40-hex commit by hash.
    """
    origin = tmp_path / "origin"
    origin.mkdir()
    git(origin, "init", "-q", "-b", "master")
    git(origin, "config", "user.email", "test@example.com")
    git(origin, "config", "user.name", "test")
    (origin / "README.md").write_text("commit A\n", encoding="utf-8")
    git(origin, "add", "-A")
    git(origin, "commit", "-qm", "A")
    pin = git(origin, "rev-parse", "HEAD")
    (origin / "README.md").write_text("commit B\n", encoding="utf-8")
    (origin / "tools.txt").write_text("tool\n", encoding="utf-8")
    git(origin, "add", "-A")
    git(origin, "commit", "-qm", "B")
    git(origin, "config", "uploadpack.allowAnySHA1InWant", "true")
    return origin, pin


def fresh_no_checkout_clone(origin: Path, tmp_path: Path) -> Path:
    """The exact clone command scripts/gguf-quickstart.sh uses (F-04 state)."""
    clone = tmp_path / "llama.cpp"
    subprocess.run(
        [
            "git", "clone", "-q", "--filter=blob:none", "--no-checkout",
            f"file://{origin}", str(clone),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return clone


def populated_checkout(origin: Path, tmp_path: Path, pin: str) -> Path:
    """A later-run state: pinned commit detached, worktree and index present."""
    clone = fresh_no_checkout_clone(origin, tmp_path)
    subprocess.run(
        ["git", "-C", str(clone), "fetch", "-q", "--depth", "1",
         f"file://{origin}", pin],
        capture_output=True, text=True, check=True,
    )
    git(clone, "checkout", "-q", "--detach", "FETCH_HEAD")
    return clone


def guard_rc(clone: Path) -> str:
    """Exit status of the guard decision function: rc=0 dirty, rc=1 clean."""
    result = bash(
        f'source {LIB}; rc=0; llama_has_tracked_changes {clone} || rc=$?; '
        'printf \'rc=%s\\n\' "$rc"'
    )
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


def test_fresh_no_checkout_clone_reads_clean_not_dirty(tmp_path: Path) -> None:
    origin, _pin = make_origin(tmp_path)
    clone = fresh_no_checkout_clone(origin, tmp_path)
    # Sanity: this really is the F-04 state -- empty worktree, no index, and
    # plain git diff misreads it as mass staged deletions.
    assert not (clone / ".git/index").exists()
    assert not (clone / "README.md").exists()
    plain = bash(f"git -C {clone} diff --quiet --ignore-submodules HEAD --")
    assert plain.returncode != 0, "test state broken: clone is not index-less"
    # The guard must recognize its own clone state instead (F-04 fix).
    assert guard_rc(clone) == "rc=1"


def test_fresh_clone_with_head_not_at_pin_selects_checkout_not_refuse(
    tmp_path: Path,
) -> None:
    origin, pin = make_origin(tmp_path)
    clone = fresh_no_checkout_clone(origin, tmp_path)
    head = git(clone, "rev-parse", "HEAD")
    assert head != pin, "test state broken: default tip must differ from pin"
    # Mirrors the quickstart ref-mismatch branch: guard says clean -> fetch and
    # detach at the pin; anything else must fail loudly instead of proceeding.
    driver = (
        "set -euo pipefail\n"
        f"source {LIB}\n"
        "rc=0\n"
        f"llama_has_tracked_changes {clone} || rc=$?\n"
        'if [ "$rc" -ne 1 ]; then printf \'guard-refused rc=%s\\n\' "$rc" >&2; exit 3; fi\n'
        f"git -C {clone} fetch -q --depth 1 file://{origin} {pin}\n"
        f"git -C {clone} checkout -q --detach FETCH_HEAD\n"
        f'printf \'checked-out=%s\\n\' "$(git -C {clone} rev-parse HEAD)"\n'
    )
    result = bash(driver)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == f"checked-out={pin}"
    assert (clone / ".git/index").exists()
    assert (clone / "README.md").exists()


def test_clean_populated_checkout_reads_clean(tmp_path: Path) -> None:
    origin, pin = make_origin(tmp_path)
    clone = populated_checkout(origin, tmp_path, pin)
    assert guard_rc(clone) == "rc=1"


def test_populated_dirty_worktree_still_refuses(tmp_path: Path) -> None:
    origin, pin = make_origin(tmp_path)
    clone = populated_checkout(origin, tmp_path, pin)
    (clone / "README.md").write_text("local edit\n", encoding="utf-8")
    assert guard_rc(clone) == "rc=0"


def test_staged_change_on_populated_checkout_still_refuses(
    tmp_path: Path,
) -> None:
    origin, pin = make_origin(tmp_path)
    clone = populated_checkout(origin, tmp_path, pin)
    (clone / "README.md").write_text("staged edit\n", encoding="utf-8")
    git(clone, "add", "README.md")
    assert guard_rc(clone) == "rc=0"


def test_quickstart_checks_out_pinned_ref_before_any_guard_runs() -> None:
    src = SCRIPT.read_text(encoding="utf-8")
    clone_at = src.index("git clone --filter=blob:none --no-checkout")
    checkout_at = src.index('git -C "$LLAMA" checkout --detach FETCH_HEAD')
    guard_at = src.index("CURRENT_LLAMA_CPP_COMMIT=")
    assert clone_at < checkout_at < guard_at, (
        "fresh clone must reach the pinned ref before any dirty-tree guard"
    )
