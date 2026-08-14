"""Behavior tests for environment-check profiles and memory policy."""

from __future__ import annotations

import os
import json
import re
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CHECKER = ROOT / "scripts/00-check-env.sh"
SYSTEM_BASH = shutil.which("bash")

# The README-declared host tools the gguf profile must verify (F-12), matching
# the required-commands loop in scripts/gguf-quickstart.sh.
GGUF_HOST_TOOLS = ("git", "cmake", "curl", "python3")


def make_rocm(prefix: Path, version: str, pool_gib: int, gpu: str = "gfx1151") -> None:
    (prefix / "bin").mkdir(parents=True)
    (prefix / ".info").mkdir()
    (prefix / ".info/version").write_text(version + "\n", encoding="utf-8")
    hipcc = prefix / "bin/hipcc"
    hipcc.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    hipcc.chmod(0o755)
    rocminfo = prefix / "bin/rocminfo"
    rocminfo.write_text(
        "#!/usr/bin/env bash\n"
        "cat <<'EOF'\n"
        f"  Name:                    {gpu}\n"
        "      Segment:                 GLOBAL; FLAGS: COARSE GRAINED\n"
        f"      Size:                    {pool_gib * 1024 * 1024}(0x0) KB\n"
        "EOF\n",
        encoding="utf-8",
    )
    rocminfo.chmod(0o755)


def make_tools(directory: Path, *, uv_exit: int) -> Path:
    directory.mkdir(exist_ok=True)
    uname = directory / "uname"
    uname.write_text(
        "#!/usr/bin/env bash\nprintf '%s\\n' 6.17.0-1032-oem\n", encoding="utf-8"
    )
    uname.chmod(0o755)
    uv = directory / "uv"
    uv.write_text(
        "#!/usr/bin/env bash\n"
        f"if [ {uv_exit} -ne 0 ]; then exit {uv_exit}; fi\n"
        "echo 'TheRock PyTorch: fake test runtime (HIP test)'\n",
        encoding="utf-8",
    )
    uv.chmod(0o755)
    return directory


def run_checker(
    tmp_path: Path,
    *,
    profile: str | None = None,
    version: str = "7.14.0",
    pool_gib: int = 80,
    uv_exit: int = 97,
    path: str | None = None,
    gpu: str = "gfx1151",
) -> subprocess.CompletedProcess[str]:
    prefix = tmp_path / "rocm"
    make_rocm(prefix, version, pool_gib, gpu)
    tools = make_tools(tmp_path / "tools", uv_exit=uv_exit)
    env = os.environ.copy()
    env["HOME"] = str(tmp_path / "home")
    env["ROCM_PREFIX"] = str(prefix)
    env.pop("ROCM_PATH", None)
    env["PATH"] = path if path is not None else f"{tools}:{env['PATH']}"
    args = [SYSTEM_BASH, str(CHECKER)]
    if profile is not None:
        args.extend(("--profile", profile))
    return subprocess.run(args, cwd=ROOT, env=env, capture_output=True, text=True)


def closed_tools(tmp_path: Path, tools_dir: Path, *keep: str) -> Path:
    """A bin dir that IS the whole PATH: only the named tools resolve.

    Mirrors the installer-cluster F-12 tests: a closed PATH reproduces a host
    missing a tool without the real system PATH leaking it back in. `uname` is
    the fake from make_tools so the kernel floor stays host-independent.
    """
    bindir = tmp_path / "closedbin"
    bindir.mkdir()
    (bindir / "uname").symlink_to(tools_dir / "uname")
    for tool in keep:
        (bindir / tool).symlink_to(shutil.which(tool))
    return bindir


def test_default_profile_is_gguf_and_does_not_invoke_uv(tmp_path: Path) -> None:
    result = run_checker(tmp_path, uv_exit=97)

    assert result.returncode == 0, result.stdout + result.stderr
    assert "Environment profile: gguf" in result.stdout
    assert "path: default llama.cpp + GGUF" in result.stdout
    assert "OK: gguf environment ready" in result.stdout


def test_explicit_gguf_profile_passes_on_rocm_714(tmp_path: Path) -> None:
    result = run_checker(tmp_path, profile="gguf")

    assert result.returncode == 0, result.stdout + result.stderr
    assert "version: 7.14.0" in result.stdout
    assert "track: recommended" in result.stdout


def test_gguf_warns_below_validated_memory_envelope(tmp_path: Path) -> None:
    result = run_checker(tmp_path, profile="gguf", pool_gib=32)

    assert result.returncode == 0, result.stdout + result.stderr
    assert "outside the validated 94 GiB Strix Halo" in result.stderr
    assert "may still run" in result.stderr


def test_gguf_fails_when_pool_is_smaller_than_default_artifact(tmp_path: Path) -> None:
    result = run_checker(tmp_path, profile="gguf", pool_gib=15)

    assert result.returncode != 0
    assert "15 GiB" in result.stderr, "must state the available GPU-visible pool"
    assert f"{default_gguf_floor_gib()} GiB" in result.stderr, (
        "must state the required GPU-visible floor in the same units"
    )
    assert "GGUF_FILE" in result.stderr, "must hint the smaller-quant escape hatch"


def test_vllm_profile_passes_at_reference_floor(tmp_path: Path) -> None:
    result = run_checker(
        tmp_path, profile="vllm", version="7.2.1", pool_gib=60, uv_exit=0
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "path: optional advanced vLLM + BF16" in result.stdout
    assert "TheRock PyTorch: fake test runtime" in result.stdout
    assert "OK: vllm environment ready" in result.stdout


def test_vllm_profile_fails_below_60_gib(tmp_path: Path) -> None:
    result = run_checker(
        tmp_path, profile="vllm", version="7.2.1", pool_gib=59, uv_exit=0
    )

    assert result.returncode != 0
    assert "GPU-visible pool < 60 GiB" in result.stderr


def test_vllm_profile_rejects_rocm_714(tmp_path: Path) -> None:
    result = run_checker(
        tmp_path, profile="vllm", version="7.14.0", pool_gib=80, uv_exit=0
    )

    assert result.returncode != 0
    assert "requires the validated ROCm 7.2.1 toolchain" in result.stderr


def test_reference_profile_certifies_recorded_kernel(tmp_path: Path) -> None:
    result = run_checker(
        tmp_path, profile="reference", version="7.2.1", pool_gib=80, uv_exit=0
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "path: exact historical vLLM/BF16 reference" in result.stdout
    assert "OK: reference environment ready" in result.stdout


def test_invalid_profile_exits_two(tmp_path: Path) -> None:
    result = subprocess.run(
        ["bash", str(CHECKER), "--profile", "invalid"],
        cwd=ROOT,
        env={**os.environ, "HOME": str(tmp_path)},
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "invalid profile: invalid" in result.stderr
    assert "--profile gguf|vllm|reference" in result.stderr


# ---------------------------------------------------------------------------
# F-12: the gguf profile must verify the README-declared host tools
# ---------------------------------------------------------------------------


def test_gguf_profile_itemizes_host_tools_before_the_verdict(tmp_path: Path) -> None:
    result = run_checker(tmp_path, profile="gguf")

    assert result.returncode == 0, result.stdout + result.stderr
    for tool in GGUF_HOST_TOOLS:
        match = re.search(rf"^  tool {re.escape(tool)}: (\S+)$", result.stdout, re.M)
        assert match, f"missing per-tool pass line for {tool}"
        assert match.group(1).startswith("/"), (
            f"the {tool} pass line must show the resolved path, got {match.group(1)!r}"
        )
    verdict_at = result.stdout.index("OK: gguf environment ready")
    for tool in GGUF_HOST_TOOLS:
        assert result.stdout.index(f"tool {tool}: ") < verdict_at, (
            f"the {tool} pass line must feed the verdict, not trail it"
        )


def test_gguf_profile_missing_cmake_fails_with_per_distro_hints(
    tmp_path: Path,
) -> None:
    tools = make_tools(tmp_path / "tools", uv_exit=97)
    # cmake is the one README-declared tool left out of the closed PATH.
    bindir = closed_tools(
        tmp_path, tools, "bash", "dirname", "python3", "git", "curl", "awk", "grep"
    )

    result = run_checker(tmp_path, profile="gguf", path=str(bindir))

    assert result.returncode == 1
    assert "FAIL: required command not found: cmake" in result.stderr
    for hint in (
        "sudo apt-get install cmake",
        "sudo dnf install cmake",
        "sudo pacman -S cmake",
    ):
        assert hint in result.stderr, f"must carry the per-distro install hint ({hint})"
    assert "cmake: command not found" not in result.stderr
    assert "OK: gguf environment ready" not in result.stdout


def test_missing_python3_fails_fast_with_install_hints(tmp_path: Path) -> None:
    tools = make_tools(tmp_path / "tools", uv_exit=97)
    bindir = closed_tools(tmp_path, tools, "bash", "dirname")  # python3 absent

    result = run_checker(tmp_path, path=str(bindir))

    assert result.returncode == 1
    assert "FAIL: required command not found: python3" in result.stderr
    for hint in (
        "sudo apt-get install python3",
        "sudo dnf install python3",
        "sudo pacman -S python3",
    ):
        assert hint in result.stderr
    assert "python3: command not found" not in result.stderr, (
        "the guard must fire instead of a raw bash 'command not found' error"
    )
    assert result.stdout == "", "the guard must fire before any diagnostic output"


def test_python3_guard_precedes_first_manifest_read() -> None:
    src = CHECKER.read_text(encoding="utf-8")
    guard_at = src.index("command -v python3")
    first_manifest_read = src.index("read_manifest host.minimum_kernel")
    assert guard_at < first_manifest_read, (
        "the python3 guard must run before read_manifest shells out to python3"
    )


# ---------------------------------------------------------------------------
# F-14: the passing-path pool line must state its thresholds and label units
# ---------------------------------------------------------------------------


def default_gguf_floor_gib() -> str:
    """The default-GGUF floor exactly as the checker derives it: the manifest
    size of muse-glimmer-30B-kquant-17gb.gguf rendered in GiB to 1 decimal."""
    files = json.loads(
        (ROOT / "configs/artifact-manifest.json").read_text(encoding="utf-8")
    )["sets"]["gguf"]["files"]
    size_bytes = next(
        f["size_bytes"]
        for f in files
        if f["path"] == "muse-glimmer-30B-kquant-17gb.gguf"
    )
    return f"{size_bytes / 1024**3:.1f}"


def pool_line(stdout: str) -> str:
    return next(l for l in stdout.splitlines() if l.startswith("GPU-visible pool:"))


def test_gguf_pool_line_states_floor_and_envelope(tmp_path: Path) -> None:
    result = run_checker(tmp_path, profile="gguf", pool_gib=80)

    assert result.returncode == 0, result.stdout + result.stderr
    line = pool_line(result.stdout)
    assert line.startswith("GPU-visible pool: 80 GiB")
    floor = default_gguf_floor_gib()
    assert f"default GGUF needs >= {floor} GiB GPU-visible" in line, (
        "the passing line must carry the manifest-derived default-GGUF floor"
    )
    assert "validated envelope 80 GiB GPU-visible" in line
    assert "warning boundary" in line, (
        "the 80 GiB envelope must be labeled as a warning boundary, not a minimum"
    )


def test_vllm_pool_line_states_profile_floor(tmp_path: Path) -> None:
    result = run_checker(
        tmp_path, profile="vllm", version="7.2.1", pool_gib=80, uv_exit=0
    )

    assert result.returncode == 0, result.stdout + result.stderr
    line = pool_line(result.stdout)
    assert line.startswith("GPU-visible pool: 80 GiB")
    assert ">= 60 GiB GPU-visible" in line, (
        "the non-gguf passing line must carry the profile's 60 GiB floor"
    )


# ---------------------------------------------------------------------------
# F-15: failure exits must state expected-vs-got and one next action
# ---------------------------------------------------------------------------


def test_rocm_version_mismatch_prints_expected_got_and_next_action(
    tmp_path: Path,
) -> None:
    result = run_checker(tmp_path, profile="gguf", version="6.4.0")

    assert result.returncode != 0
    assert "7.14" in result.stderr and "7.2" in result.stderr, (
        "must state the expected ROCm versions"
    )
    assert "got '6.4.0'" in result.stderr, "must state the observed version"
    assert "scripts/install-rocm-7.14.sh" in result.stderr, (
        "must point at the installer as the next action"
    )
    assert "ROCM_PREFIX" in result.stderr, "must name the ROCM_PREFIX escape hatch"


def test_gfx1151_absent_lists_observed_gpu_ids_and_doc_pointer(
    tmp_path: Path,
) -> None:
    result = run_checker(tmp_path, profile="gguf", gpu="gfx1100")

    assert result.returncode != 0
    assert "gfx1151 not found" in result.stderr
    assert "gfx1100" in result.stderr, "must list the observed GPU id(s)"
    assert "docs/hardware-validation.md" in result.stderr, (
        "must point non-gfx1151 platforms at the hardware validation doc"
    )
