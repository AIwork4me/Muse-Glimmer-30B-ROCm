import json
import os
from pathlib import Path
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]
LIB = ROOT / "scripts/lib/llama_build.sh"


def bash(script: str, *, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
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


def fake_rocm(prefix: Path, identity: str) -> None:
    hipcc = prefix / "bin/hipcc"
    hipcc.parent.mkdir(parents=True, exist_ok=True)
    hipcc.write_text(f"#!/bin/sh\nprintf '%s\\n' '{identity}'\n", encoding="utf-8")
    hipcc.chmod(0o755)


def write_fingerprint(
    output: Path,
    prefix: Path,
    *,
    commit: str = "a" * 40,
    version: str = "7.14.0",
    target: str = "gfx1151",
) -> subprocess.CompletedProcess[str]:
    return bash(
        f"source {LIB!s}; "
        f"write_llama_build_fingerprint {output!s} {commit} {prefix!s} {version} {target}"
    )


def test_standard_and_custom_build_directories_are_isolated(tmp_path: Path) -> None:
    home = tmp_path / "home"
    recommended = home / "rocm-7.14.0"
    custom_a = tmp_path / "custom-a"
    custom_b = tmp_path / "custom-b"
    for path in (recommended, custom_a, custom_b):
        path.mkdir(parents=True)
    llama = tmp_path / "llama.cpp"
    command = (
        f"source {LIB!s}; "
        f"llama_build_dir {llama!s} {recommended!s} 7.14.0; "
        f"llama_build_dir {llama!s} /opt/rocm 7.2.1; "
        f"llama_build_dir {llama!s} {custom_a!s} 7.14.0; "
        f"llama_build_dir {llama!s} {custom_b!s} 7.14.0; "
        f"llama_build_dir {llama!s} {custom_a!s} 7.14.0 {tmp_path / 'override'!s}"
    )
    result = bash(command, env={"HOME": str(home)})

    assert result.returncode == 0, result.stderr
    recommended_dir, historical_dir, custom_a_dir, custom_b_dir, override = result.stdout.splitlines()
    assert recommended_dir == str(llama / "build-714")
    assert historical_dir == str(llama / "build")
    assert custom_a_dir.startswith(str(llama / "build-rocm-7.14.0-"))
    assert custom_b_dir.startswith(str(llama / "build-rocm-7.14.0-"))
    assert custom_a_dir != custom_b_dir
    assert override == str(tmp_path / "override")


def test_same_fingerprint_matches_and_records_toolchain_identity(tmp_path: Path) -> None:
    prefix = tmp_path / "rocm"
    fake_rocm(prefix, "HIP version 7.14; clang version 23")
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"

    assert write_fingerprint(first, prefix).returncode == 0
    assert write_fingerprint(second, prefix).returncode == 0
    match = bash(f"source {LIB!s}; llama_build_fingerprint_matches {first!s} {second!s}")
    data = json.loads(first.read_text(encoding="utf-8"))

    assert match.returncode == 0
    assert first.read_bytes() == second.read_bytes()
    assert data["rocm_prefix"] == str(prefix.resolve())
    assert data["rocm_version"] == "7.14.0"
    assert data["hipcc"] == "HIP version 7.14; clang version 23"
    assert data["amdgpu_targets"] == ["gfx1151"]
    assert data["cmake"] == {
        "AMDGPU_TARGETS": "gfx1151",
        "CMAKE_BUILD_TYPE": "Release",
        "GGML_HIP": True,
        "ROCM_PATH": str(prefix.resolve()),
        "hip_DIR": f"{prefix.resolve()}/lib/cmake/hip",
    }


@pytest.mark.parametrize("changed", ["commit", "version", "hipcc", "target", "prefix"])
def test_changed_toolchain_fingerprint_invalidates(tmp_path: Path, changed: str) -> None:
    prefix = tmp_path / "rocm"
    other_prefix = tmp_path / "other-rocm"
    fake_rocm(prefix, "HIP version 7.14; clang version 23")
    fake_rocm(other_prefix, "HIP version 7.14; clang version 23")
    recorded = tmp_path / "recorded.json"
    candidate = tmp_path / "candidate.json"
    assert write_fingerprint(recorded, prefix).returncode == 0

    kwargs = {"commit": "a" * 40, "version": "7.14.0", "target": "gfx1151"}
    selected_prefix = prefix
    if changed == "commit":
        kwargs["commit"] = "b" * 40
    elif changed == "version":
        kwargs["version"] = "7.15.0"
    elif changed == "hipcc":
        fake_rocm(prefix, "HIP version 7.14; clang version 24")
    elif changed == "target":
        kwargs["target"] = "gfx1100"
    elif changed == "prefix":
        selected_prefix = other_prefix

    result = write_fingerprint(candidate, selected_prefix, **kwargs)
    match = bash(f"source {LIB!s}; llama_build_fingerprint_matches {candidate!s} {recorded!s}")
    assert result.returncode == 0, result.stderr
    assert match.returncode != 0
