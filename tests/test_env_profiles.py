"""Behavior tests for environment-check profiles and memory policy."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CHECKER = ROOT / "scripts/00-check-env.sh"


def make_rocm(prefix: Path, version: str, pool_gib: int) -> None:
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
        "  Name:                    gfx1151\n"
        "      Segment:                 GLOBAL; FLAGS: COARSE GRAINED\n"
        f"      Size:                    {pool_gib * 1024 * 1024}(0x0) KB\n"
        "EOF\n",
        encoding="utf-8",
    )
    rocminfo.chmod(0o755)


def make_tools(directory: Path, *, uv_exit: int) -> Path:
    directory.mkdir()
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
) -> subprocess.CompletedProcess[str]:
    prefix = tmp_path / "rocm"
    make_rocm(prefix, version, pool_gib)
    tools = make_tools(tmp_path / "tools", uv_exit=uv_exit)
    env = os.environ.copy()
    env["HOME"] = str(tmp_path / "home")
    env["ROCM_PREFIX"] = str(prefix)
    env.pop("ROCM_PATH", None)
    env["PATH"] = f"{tools}:{env['PATH']}"
    args = ["bash", str(CHECKER)]
    if profile is not None:
        args.extend(("--profile", profile))
    return subprocess.run(args, cwd=ROOT, env=env, capture_output=True, text=True)


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
    assert "smaller than the 16756681056-byte default GGUF" in result.stderr


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
