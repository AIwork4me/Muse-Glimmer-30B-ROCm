"""Behavior tests for the shared ROCm prefix resolver."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LIB = ROOT / "scripts/lib/rocm.sh"


def make_rocm(prefix: Path, version: str) -> None:
    (prefix / "bin").mkdir(parents=True)
    (prefix / ".info").mkdir()
    (prefix / ".info/version").write_text(version + "\n", encoding="utf-8")
    for command in ("hipcc", "rocminfo"):
        path = prefix / "bin" / command
        path.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
        path.chmod(0o755)


def run_resolver(
    home: Path,
    fallback: Path,
    *,
    rocm_prefix: str | None = None,
    rocm_path: str | None = None,
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["HOME"] = str(home)
    for name, value in (("ROCM_PREFIX", rocm_prefix), ("ROCM_PATH", rocm_path)):
        if value is None:
            env.pop(name, None)
        else:
            env[name] = value
    command = (
        f'source "{LIB}"; '
        f'resolve_rocm_prefix "{home}/rocm-7.14.0" "{fallback}" || exit $?; '
        'version="$(detect_rocm_version "$ROCM_PREFIX")"; '
        'print_selected_rocm "$version"'
    )
    return subprocess.run(
        ["bash", "-c", command], capture_output=True, text=True, env=env
    )


def test_home_714_is_preferred_over_historical_fallback(tmp_path: Path) -> None:
    home = tmp_path / "home"
    fallback = tmp_path / "opt-rocm"
    make_rocm(home / "rocm-7.14.0", "7.14.0")
    make_rocm(fallback, "7.2.1")

    result = run_resolver(home, fallback)

    assert result.returncode == 0, result.stderr
    assert f"prefix: {home}/rocm-7.14.0" in result.stdout
    assert "version: 7.14.0" in result.stdout
    assert "track: recommended" in result.stdout


def test_rocm_prefix_precedes_rocm_path_and_defaults(tmp_path: Path) -> None:
    home = tmp_path / "home"
    fallback = tmp_path / "opt-rocm"
    explicit_prefix = tmp_path / "explicit-prefix"
    explicit_path = tmp_path / "explicit-path"
    for prefix, version in (
        (home / "rocm-7.14.0", "7.14.0"),
        (fallback, "7.2.1"),
        (explicit_prefix, "7.14.1"),
        (explicit_path, "7.2.2"),
    ):
        make_rocm(prefix, version)

    result = run_resolver(
        home,
        fallback,
        rocm_prefix=str(explicit_prefix),
        rocm_path=str(explicit_path),
    )

    assert result.returncode == 0, result.stderr
    assert f"prefix: {explicit_prefix}" in result.stdout
    assert "version: 7.14.1" in result.stdout
    assert "source: ROCM_PREFIX override" in result.stdout


def test_rocm_path_precedes_defaults(tmp_path: Path) -> None:
    home = tmp_path / "home"
    fallback = tmp_path / "opt-rocm"
    explicit_path = tmp_path / "explicit-path"
    make_rocm(home / "rocm-7.14.0", "7.14.0")
    make_rocm(fallback, "7.2.1")
    make_rocm(explicit_path, "7.2.2")

    result = run_resolver(home, fallback, rocm_path=str(explicit_path))

    assert result.returncode == 0, result.stderr
    assert f"prefix: {explicit_path}" in result.stdout
    assert "source: ROCM_PATH override" in result.stdout


def test_historical_fallback_is_used_when_home_714_is_missing(tmp_path: Path) -> None:
    home = tmp_path / "home"
    fallback = tmp_path / "opt-rocm"
    make_rocm(fallback, "7.2.1")

    result = run_resolver(home, fallback)

    assert result.returncode == 0, result.stderr
    assert f"prefix: {fallback}" in result.stdout
    assert "version: 7.2.1" in result.stdout
    assert "track: historical reference" in result.stdout


def test_missing_rocm_fails_with_actionable_error(tmp_path: Path) -> None:
    home = tmp_path / "home"
    fallback = tmp_path / "opt-rocm"

    result = run_resolver(home, fallback)

    assert result.returncode != 0
    assert "no usable ROCm installation found" in result.stderr
    assert "bash scripts/install-rocm-7.14.sh" in result.stderr


def test_explicit_incomplete_prefix_fails_without_falling_back(tmp_path: Path) -> None:
    home = tmp_path / "home"
    fallback = tmp_path / "opt-rocm"
    explicit = tmp_path / "incomplete"
    make_rocm(home / "rocm-7.14.0", "7.14.0")
    make_rocm(fallback, "7.2.1")
    explicit.mkdir()

    result = run_resolver(home, fallback, rocm_prefix=str(explicit))

    assert result.returncode != 0
    assert f"selected ROCm prefix is incomplete: {explicit}" in result.stderr


def test_both_entrypoints_call_the_shared_resolver() -> None:
    for relative in ("scripts/00-check-env.sh", "scripts/gguf-quickstart.sh"):
        source = (ROOT / relative).read_text(encoding="utf-8")
        assert 'scripts/lib/rocm.sh"' in source
        assert "resolve_rocm_prefix" in source
