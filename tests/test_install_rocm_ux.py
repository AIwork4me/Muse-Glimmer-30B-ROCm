"""UX-audit regression tests for scripts/install-rocm-7.14.sh.

Covers the installer-cluster findings (F-01, F-02, F-03, F-12, F-15) from the
2026-08-14 new-user one-pass audit. The shipped manifest pins a
1,713,449,440-byte tarball and its SHA256, so these tests drive the installer
end to end through its ROCM714_MANIFEST test seam using a tiny stand-in
tarball whose size and hash a fake manifest matches, a stub curl that
"downloads" it, and a stub bin/hipcc that emits five version lines - the
exact shape that SIGPIPEd the old cosmetic tail pipeline under
`set -o pipefail` (F-01).
"""
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import tempfile

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/install-rocm-7.14.sh"
SYSTEM_BASH = shutil.which("bash")

HIPCC_FIRST_LINE = "HIP version: stub-7.14.0-test"
HIPCC_EXTRA_LINES = [f"STUB-HIPCC-EXTRA-{n}" for n in range(2, 6)]

# The installer's disk preflight refuses to start unless the target filesystem
# has ~10 GiB free (manifest archive + extracted-tree floor). Tests that let
# the real df run past that preflight need that much room for real.
PREFLIGHT_FLOOR_BYTES = 10 * 1024**3


def needs_real_preflight_space(test):
    return pytest.mark.skipif(
        shutil.disk_usage(tempfile.gettempdir()).free < PREFLIGHT_FLOOR_BYTES,
        reason="installer disk preflight requires ~10 GiB free on the test filesystem",
    )(test)


def bash(script: str, *, env: dict | None = None) -> subprocess.CompletedProcess[str]:
    merged = os.environ.copy()
    if env:
        merged.update(env)
    return subprocess.run(
        ["bash", "-c", script],
        cwd=ROOT,
        env=merged,
        text=True,
        capture_output=True,
        check=False,
    )


def make_hipcc_tree(parent: Path) -> Path:
    """Staging tree for the stand-in tarball: bin/hipcc emits 5 lines."""
    tree = parent / "tree"
    (tree / "bin").mkdir(parents=True)
    body = "\n".join([HIPCC_FIRST_LINE, *HIPCC_EXTRA_LINES]) + "\n"
    (tree / "bin/hipcc").write_text(
        "#!/bin/sh\ncat <<'EOF'\n" + body + "EOF\n", encoding="utf-8"
    )
    (tree / "bin/hipcc").chmod(0o755)
    return tree


def make_tarball(parent: Path, *, with_hipcc: bool = True) -> Path:
    tree = make_hipcc_tree(parent) if with_hipcc else _tree_without_hipcc(parent)
    tarball = parent / "src.tar.gz"
    subprocess.run(
        ["tar", "-czf", str(tarball), "-C", str(tree), "."],
        capture_output=True,
        text=True,
        check=True,
    )
    return tarball


def _tree_without_hipcc(parent: Path) -> Path:
    tree = parent / "tree-nohipcc"
    tree.mkdir(parents=True)
    (tree / "README.txt").write_text("no hipcc in this one\n", encoding="utf-8")
    return tree


def install_fake_curl(bindir: Path, source: Path) -> None:
    """A curl that "downloads" the local stand-in tarball to its --output."""
    curl = bindir / "curl"
    curl.write_text(
        "#!/bin/sh\n"
        f"SRC={shlex.quote(str(source))}\n"
        "out=\nprev=\n"
        'for arg in "$@"; do\n'
        '  [ "$prev" = "--output" ] && out="$arg"\n'
        '  prev="$arg"\n'
        "done\n"
        '[ -n "$out" ] || { echo "fake curl: no --output given" >&2; exit 22; }\n'
        'exec cp -- "$SRC" "$out"\n',
        encoding="utf-8",
    )
    curl.chmod(0o755)


def install_fake_df(bindir: Path, avail_kb_by_marker: dict[str, int], default_kb: int) -> None:
    """A df(1) stub reporting canned availability, keyed by path markers.

    The real installer parses `df -Pk <dir>`; this stub emits the same POSIX
    shape with an Available column chosen by which path df was asked about.
    """
    arms = [
        "#!/bin/sh",
        'avail=' + str(default_kb),
        'case "$*" in',
    ]
    for marker, kb in avail_kb_by_marker.items():
        arms.append(f"  *{marker}*) avail={kb} ;;")
    arms += [
        "esac",
        "printf 'Filesystem 1024-blocks Used Available Capacity Mounted on\\n'",
        'printf \'fakefs 1 1 %s 99%% /fake\\n\' "$avail"',
    ]
    df = bindir / "df"
    df.write_text("\n".join(arms) + "\n", encoding="utf-8")
    df.chmod(0o755)


def write_manifest(
    parent: Path,
    tarball: Path,
    *,
    sha256: str | None = None,
    size_bytes: int | None = None,
) -> Path:
    manifest = parent / "manifest.json"
    payload = {
        "host": {
            "rocm_version": "7.14.0-test",
            "archive": {
                "url": "stub://rocm-test.tar.gz",
                "size_bytes": (
                    tarball.stat().st_size if size_bytes is None else size_bytes
                ),
                "sha256": sha256
                or hashlib.sha256(tarball.read_bytes()).hexdigest(),
            },
        }
    }
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    return manifest


def run_installer(
    tmp_path: Path,
    *,
    manifest: Path,
    prefix: Path | None = None,
    archive: Path | None = None,
    extra_env: dict[str, str] | None = None,
    path: str | None = None,
) -> subprocess.CompletedProcess[str]:
    home = tmp_path / "home"
    home.mkdir(exist_ok=True)
    env = os.environ.copy()
    env["HOME"] = str(home)
    env["ROCM714_MANIFEST"] = str(manifest)
    env["ROCM714_PREFIX"] = str(prefix or tmp_path / "rocm")
    env["ROCM714_ARCHIVE"] = str(archive or tmp_path / "archive.tar.gz")
    env.pop("TMPDIR", None)
    env["PATH"] = path or f"{tmp_path / 'bin'}:{env['PATH']}"
    if extra_env:
        env.update(extra_env)
    return subprocess.run(
        [SYSTEM_BASH, str(SCRIPT)],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


# ---------------------------------------------------------------------------
# F-01: the full-install success path must exit 0, not 141 (SIGPIPE)
# ---------------------------------------------------------------------------


def test_old_tail_pipeline_sigpipes_under_pipefail(tmp_path: Path) -> None:
    """Regression pin on the mechanism: a multi-line writer piped into `head -1`
    under `set -euo pipefail` dies with 141 once head closes the pipe."""
    stub = tmp_path / "slow-5-line-writer"
    stub.write_text(
        "#!/bin/sh\n"
        "printf 'line-1\\n'\n"
        "sleep 0.4\n"
        "printf 'line-2\\nline-3\\nline-4\\nline-5\\n'\n",
        encoding="utf-8",
    )
    stub.chmod(0o755)
    old = bash(
        'set -euo pipefail\n'
        f'"{stub}" --version | head -1\n'
    )
    assert old.returncode == 141, (
        f"expected the pre-fix pattern to SIGPIPE (141), got {old.returncode}; "
        "the test stub no longer reproduces F-01"
    )
    assert old.stdout == "line-1\n"


def test_capture_then_print_tail_survives_pipefail(tmp_path: Path) -> None:
    """The fixed construct: capture-then-print never hands the writer a pipe
    that can close early, so pipefail has nothing to promote."""
    stub = tmp_path / "slow-5-line-writer"
    stub.write_text(
        "#!/bin/sh\n"
        "printf 'line-1\\n'\n"
        "sleep 0.4\n"
        "printf 'line-2\\nline-3\\nline-4\\nline-5\\n'\n",
        encoding="utf-8",
    )
    stub.chmod(0o755)
    fixed = bash(
        'set -euo pipefail\n'
        f'head -1 <<<"$("{stub}" --version)"\n'
    )
    assert fixed.returncode == 0, fixed.stderr
    assert fixed.stdout == "line-1\n"


@needs_real_preflight_space
def test_full_install_path_exits_zero_and_prints_one_hipcc_line(
    tmp_path: Path,
) -> None:
    """F-01 acceptance: a complete download->verify->extract install (the path
    that used to end in exit 141) must exit 0 and print only hipcc's first
    version line."""
    bindir = tmp_path / "bin"
    bindir.mkdir()
    tarball = make_tarball(tmp_path)
    install_fake_curl(bindir, tarball)
    manifest = write_manifest(tmp_path, tarball)
    prefix = tmp_path / "rocm"
    archive = tmp_path / "archive.tar.gz"

    result = run_installer(tmp_path, manifest=manifest, prefix=prefix, archive=archive)

    assert result.returncode == 0, result.stdout + result.stderr
    assert (prefix / "bin/hipcc").exists()
    assert HIPCC_FIRST_LINE in result.stdout
    for extra in HIPCC_EXTRA_LINES:
        assert extra not in result.stdout


def test_script_version_tail_is_not_a_pipe() -> None:
    src = SCRIPT.read_text(encoding="utf-8")
    assert "--version | head" not in src, (
        "the cosmetic version tail must not pipe hipcc into head (F-01 SIGPIPE)"
    )
    assert '<<<"$("$PREFIX/bin/hipcc" --version)"' in src


# ---------------------------------------------------------------------------
# F-02: the verified archive must not silently stay behind in /tmp
# ---------------------------------------------------------------------------

GENEROUS_DF_KB = 200 * 1024 * 1024  # 200 GiB reported free, host-independent


@needs_real_preflight_space
def test_verified_archive_deleted_with_a_cleanup_line_after_success(
    tmp_path: Path,
) -> None:
    bindir = tmp_path / "bin"
    bindir.mkdir()
    tarball = make_tarball(tmp_path)
    install_fake_curl(bindir, tarball)
    manifest = write_manifest(tmp_path, tarball)
    archive = tmp_path / "archive.tar.gz"

    result = run_installer(tmp_path, manifest=manifest, archive=archive)

    assert result.returncode == 0, result.stdout + result.stderr
    assert not archive.exists(), "verified archive must be removed on success"
    assert str(archive) in result.stdout
    assert "cleaned up archive" in result.stdout.lower()


def test_failed_verification_keeps_the_archive_for_inspection(
    tmp_path: Path,
) -> None:
    bindir = tmp_path / "bin"
    bindir.mkdir()
    install_fake_df(bindir, {}, GENEROUS_DF_KB)  # keep the preflight out of it
    tarball = make_tarball(tmp_path)
    install_fake_curl(bindir, tarball)
    # Right size, wrong bytes: size check passes, SHA256 must fail.
    manifest = write_manifest(tmp_path, tarball, sha256="0" * 64)
    archive = tmp_path / "archive.tar.gz"

    result = run_installer(tmp_path, manifest=manifest, archive=archive)

    assert result.returncode != 0
    assert archive.exists(), (
        "failure paths must keep the archive so it can be inspected/retried"
    )


# ---------------------------------------------------------------------------
# F-03: an upfront disk-space preflight instead of a mid-install ENOSPC
# ---------------------------------------------------------------------------

ONE_GIB_KB = 1024 * 1024


def test_insufficient_archive_space_refuses_before_downloading(
    tmp_path: Path,
) -> None:
    bindir = tmp_path / "bin"
    bindir.mkdir()
    tmpdir = tmp_path / "tmpdir"
    tmpdir.mkdir()
    out = tmp_path / "out"
    out.mkdir()
    # 1 GiB free where the archive would land, plenty where the tree would.
    install_fake_df(bindir, {"*/tmpdir*": ONE_GIB_KB}, GENEROUS_DF_KB)
    tarball = make_tarball(tmp_path)
    install_fake_curl(bindir, tarball)
    # Manifest floor of 5 GiB for the archive filesystem.
    manifest = write_manifest(tmp_path, tarball, size_bytes=5 * 1024**3)
    archive = tmpdir / "archive.tar.gz"

    result = run_installer(
        tmp_path, manifest=manifest, prefix=out / "rocm", archive=archive
    )

    assert result.returncode == 1
    assert "not enough disk space" in result.stderr
    assert str(tmpdir) in result.stderr, "must name the target filesystem path"
    assert "5.0 GiB" in result.stderr, "must state required space"
    assert "1.0 GiB" in result.stderr, "must state available space"
    assert "TMPDIR" in result.stderr, "must offer the TMPDIR escape hatch"
    assert "Downloading" not in result.stdout, "must refuse before downloading"
    assert not archive.exists()


def test_insufficient_prefix_space_refuses_before_downloading(
    tmp_path: Path,
) -> None:
    bindir = tmp_path / "bin"
    bindir.mkdir()
    tmpdir = tmp_path / "tmpdir"
    tmpdir.mkdir()
    out = tmp_path / "out"
    out.mkdir()
    # Plenty for the small stub archive, only 4 GiB for the extracted tree.
    install_fake_df(bindir, {"*/out*": 4 * ONE_GIB_KB}, GENEROUS_DF_KB)
    tarball = make_tarball(tmp_path)
    install_fake_curl(bindir, tarball)
    manifest = write_manifest(tmp_path, tarball)
    archive = tmpdir / "archive.tar.gz"

    result = run_installer(
        tmp_path, manifest=manifest, prefix=out / "rocm", archive=archive
    )

    assert result.returncode == 1
    assert "not enough disk space" in result.stderr
    assert str(out) in result.stderr
    assert "9.0 GiB" in result.stderr, (
        "extracted-tree floor is 9 GiB (8.3 GiB validated tree, rounded up)"
    )
    assert "4.0 GiB" in result.stderr
    assert "ROCM714_PREFIX" in result.stderr, "must offer the prefix escape hatch"
    assert "Downloading" not in result.stdout
    assert not archive.exists()


# ---------------------------------------------------------------------------
# F-12: python3 and curl must be guarded before first use
# ---------------------------------------------------------------------------


def closed_bin(tmp_path: Path, *keep: str) -> Path:
    """A bin dir that IS the whole PATH: only the named tools resolve.

    The tool guard runs before anything else the script calls, so a closed
    PATH of {dirname[, python3]} reproduces a host missing python3 or curl
    without the real system PATH leaking the tool back in.
    """
    bindir = tmp_path / "closedbin"
    bindir.mkdir()
    for tool in keep:
        (bindir / tool).symlink_to(shutil.which(tool))
    return bindir


def assert_missing_tool_is_actionable(result, tool: str) -> None:
    assert result.returncode == 1
    assert f"required command not found: {tool}" in result.stderr
    for hint in ("apt-get install", "dnf install", "pacman -S"):
        assert hint in result.stderr, f"must carry a per-distro install hint ({hint})"
    assert f"{tool}: command not found" not in result.stderr, (
        "the guard must fire instead of a raw bash 'command not found' error"
    )


def test_missing_python3_fails_fast_with_install_hints(tmp_path: Path) -> None:
    tarball = make_tarball(tmp_path)
    manifest = write_manifest(tmp_path, tarball)
    bindir = closed_bin(tmp_path, "dirname")  # python3 absent

    result = run_installer(tmp_path, manifest=manifest, path=str(bindir))

    assert_missing_tool_is_actionable(result, "python3")


def test_missing_curl_fails_fast_with_install_hints(tmp_path: Path) -> None:
    tarball = make_tarball(tmp_path)
    manifest = write_manifest(tmp_path, tarball)
    bindir = closed_bin(tmp_path, "dirname", "python3")  # curl absent

    result = run_installer(tmp_path, manifest=manifest, path=str(bindir))

    assert_missing_tool_is_actionable(result, "curl")


def test_tool_guard_precedes_first_manifest_read() -> None:
    src = SCRIPT.read_text(encoding="utf-8")
    guard_at = src.index("command -v")
    first_read_at = src.index('URL="$(read_field')
    assert guard_at < first_read_at, (
        "the python3/curl guard must run before the manifest is read"
    )


# ---------------------------------------------------------------------------
# F-15: failure exits must carry expected-vs-got values and a next action
# ---------------------------------------------------------------------------


def test_size_mismatch_prints_expected_actual_bytes_and_action(
    tmp_path: Path,
) -> None:
    bindir = tmp_path / "bin"
    bindir.mkdir()
    install_fake_df(bindir, {}, GENEROUS_DF_KB)
    tarball = make_tarball(tmp_path)
    install_fake_curl(bindir, tarball)
    wrong_size = tarball.stat().st_size + 12345
    manifest = write_manifest(tmp_path, tarball, size_bytes=wrong_size)
    archive = tmp_path / "archive.tar.gz"

    result = run_installer(tmp_path, manifest=manifest, archive=archive)

    assert result.returncode == 1
    assert f"expected {wrong_size} bytes" in result.stderr
    assert f"got {tarball.stat().st_size} bytes" in result.stderr
    assert "rm -f" in result.stderr, "must give the delete-and-rerun action"
    assert str(archive) in result.stderr
    assert archive.exists(), "partial archive stays for inspection"


def test_sha_mismatch_prints_expected_actual_hash_and_action(
    tmp_path: Path,
) -> None:
    bindir = tmp_path / "bin"
    bindir.mkdir()
    install_fake_df(bindir, {}, GENEROUS_DF_KB)
    tarball = make_tarball(tmp_path)
    install_fake_curl(bindir, tarball)
    expected_sha = "0" * 64
    manifest = write_manifest(tmp_path, tarball, sha256=expected_sha)
    archive = tmp_path / "archive.tar.gz"

    result = run_installer(tmp_path, manifest=manifest, archive=archive)

    assert result.returncode == 1
    assert expected_sha in result.stderr
    actual_sha = hashlib.sha256(tarball.read_bytes()).hexdigest()
    assert actual_sha in result.stderr, "must show the hash actually computed"
    assert "rm -f" in result.stderr
    assert str(archive) in result.stderr
    assert archive.exists()


def test_missing_hipcc_after_extract_names_state_and_action(
    tmp_path: Path,
) -> None:
    bindir = tmp_path / "bin"
    bindir.mkdir()
    install_fake_df(bindir, {}, GENEROUS_DF_KB)
    tarball = make_tarball(tmp_path, with_hipcc=False)
    install_fake_curl(bindir, tarball)
    manifest = write_manifest(tmp_path, tarball)
    prefix = tmp_path / "rocm"
    archive = tmp_path / "archive.tar.gz"

    result = run_installer(
        tmp_path, manifest=manifest, prefix=prefix, archive=archive
    )

    assert result.returncode == 1
    assert "extraction incomplete" in result.stderr
    assert str(prefix) in result.stderr, "must name the prefix in the recovery"
    assert "rerun" in result.stderr
    assert not prefix.exists()
    leftovers = list(tmp_path.glob(".rocm-7.14.0.*"))
    assert leftovers == [], f"partial staging tree must be cleaned up: {leftovers}"
    assert archive.exists(), "failure paths keep the archive"
