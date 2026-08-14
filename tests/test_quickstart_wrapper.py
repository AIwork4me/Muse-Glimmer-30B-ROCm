import os
from pathlib import Path
import shutil
import subprocess


ROOT = Path(__file__).resolve().parents[1]
WRAPPER = ROOT / "scripts/quickstart.sh"
SYSTEM_BASH = shutil.which("bash")


def fake_bash(directory: Path) -> Path:
    executable = directory / "bash"
    executable.write_text(
        "#!/bin/sh\nprintf '%s\\n' \"$*\" >> \"$CALL_LOG\"\n",
        encoding="utf-8",
    )
    executable.chmod(0o755)
    return executable


def run_wrapper(
    tmp_path: Path,
    *args: str,
    reply: str | None = None,
    extra_env: dict[str, str] | None = None,
) -> tuple[subprocess.CompletedProcess[str], Path]:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_bash(fake_bin)
    call_log = tmp_path / "calls.log"
    env = os.environ.copy()
    env.update(
        {
            "HOME": str(tmp_path / "home"),
            "MODEL_DEST": str(tmp_path / "models"),
            "CALL_LOG": str(call_log),
            "PATH": f"{fake_bin}:{env['PATH']}",
        }
    )
    env.pop("ROCM_PREFIX", None)
    env.pop("ROCM_PATH", None)
    if extra_env:
        env.update(extra_env)
    result = subprocess.run(
        [SYSTEM_BASH, str(WRAPPER), *args],
        cwd=ROOT,
        env=env,
        input=reply,
        text=True,
        capture_output=True,
        check=False,
    )
    return result, call_log


def test_cancel_shows_sizes_and_starts_nothing(tmp_path: Path) -> None:
    result, call_log = run_wrapper(tmp_path, reply="n\n")

    assert result.returncode == 0, result.stderr
    assert "Model download: 15.6 GiB" in result.stdout
    assert "ROCm download:  1.6 GiB" in result.stdout
    assert "Continue? [y/N]" in result.stdout
    assert "Cancelled; no installer" in result.stdout
    assert not call_log.exists()


def test_interactive_confirmation_runs_thin_orchestration(tmp_path: Path) -> None:
    result, call_log = run_wrapper(tmp_path, reply="y\n")

    assert result.returncode == 0, result.stderr
    calls = call_log.read_text(encoding="utf-8").splitlines()
    assert calls == [
        str(ROOT / "scripts/install-rocm-7.14.sh"),
        f"{ROOT / 'scripts/00-check-env.sh'} --profile gguf",
        str(ROOT / "scripts/gguf-quickstart.sh"),
    ]


def test_yes_is_noninteractive_and_keeps_plan_visible(tmp_path: Path) -> None:
    result, call_log = run_wrapper(tmp_path, "--yes")

    assert result.returncode == 0, result.stderr
    assert "Muse-Glimmer RDNA Quick Start" in result.stdout
    assert "Continue?" not in result.stdout
    assert len(call_log.read_text(encoding="utf-8").splitlines()) == 3


def test_explicit_rocm_override_uses_shared_resolver_without_install(tmp_path: Path) -> None:
    prefix = tmp_path / "custom-rocm"
    (prefix / "bin").mkdir(parents=True)
    for command in ("hipcc", "rocminfo"):
        executable = prefix / "bin" / command
        executable.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        executable.chmod(0o755)
    (prefix / ".info").mkdir()
    (prefix / ".info/version").write_text("7.14.9\n", encoding="utf-8")

    result, call_log = run_wrapper(
        tmp_path,
        "--yes",
        extra_env={"ROCM_PREFIX": str(prefix)},
    )

    assert result.returncode == 0, result.stderr
    assert f"7.14.9 at {prefix} (explicit override)" in result.stdout
    assert call_log.read_text(encoding="utf-8").splitlines() == [
        f"{ROOT / 'scripts/00-check-env.sh'} --profile gguf",
        str(ROOT / "scripts/gguf-quickstart.sh"),
    ]


def test_invalid_argument_fails_before_orchestration(tmp_path: Path) -> None:
    result, call_log = run_wrapper(tmp_path, "--unknown")

    assert result.returncode == 2
    assert "unknown argument" in result.stderr
    assert not call_log.exists()


def test_wrapper_delegates_rocm_policy_instead_of_copying_it() -> None:
    source = WRAPPER.read_text(encoding="utf-8")
    assert 'source "$HERE/scripts/lib/rocm.sh"' in source
    assert 'bash "$HERE/scripts/00-check-env.sh" --profile gguf' in source
    assert 'exec bash "$HERE/scripts/gguf-quickstart.sh"' in source
    assert "/opt/rocm" not in source
