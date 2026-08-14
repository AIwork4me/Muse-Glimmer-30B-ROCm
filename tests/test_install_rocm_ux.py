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
        ["bash", str(SCRIPT)],
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
