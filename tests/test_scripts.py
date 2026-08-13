"""CI-safe tests (no GPU, no server). Validate that the scripts are well-formed
bash, the configs encode the gfx1151 adaptations, and none of the banned
adaptation flags leak in anywhere. These are the tests that actually run in CI
(the runner has no gfx1151)."""
import glob
import shutil
import subprocess

import pytest

ENTRYPOINTS = sorted(glob.glob("scripts/*.sh"))
LIBRARIES = sorted(glob.glob("scripts/lib/*.sh"))
SHELL_FILES = [*ENTRYPOINTS, *LIBRARIES]


def test_all_scripts_have_shebang_and_set_e():
    for s in ENTRYPOINTS:
        src = open(s).read()
        assert src.startswith("#!"), s
        assert "set -e" in src, s


def test_no_banned_adaptation_flags_anywhere():
    # The gfx1151 adaptation is as much about what is ABSENT as what is present.
    banned = [
        "ROCM_AITER_FA",
        "VLLM_ROCM_USE_AITER=1",
        "kv-cache-dtype fp8",
        "enable-chunked-prefill",
        "--quantization fp8",
    ]
    files = ["configs/serve-args.conf", "configs/vllm-gfx1151.env", *SHELL_FILES]
    for f in files:
        src = open(f).read()
        for b in banned:
            assert b not in src, f"banned token '{b}' found in {f}"


def test_pyproject_pins_gfx1151_index():
    toml = open("pyproject.toml").read()
    assert "rocm.nightlies.amd.com/v2/gfx1151/" in toml
    assert 'requires-python = "==3.12.*"' in toml


def test_shell_libraries_are_included_in_lint_scope():
    assert LIBRARIES, "scripts/lib/*.sh must not silently escape shell linting"
    for library in LIBRARIES:
        assert library.endswith(".sh")


@pytest.mark.skipif(not shutil.which("shellcheck"), reason="shellcheck not installed")
def test_shellcheck_clean():
    # -x follows project-local source directives. Passing libraries as
    # explicit inputs also ensures newly added helpers are linted directly.
    r = subprocess.run(
        ["shellcheck", "-x", *SHELL_FILES], capture_output=True, text=True
    )
    assert r.returncode == 0, r.stdout + r.stderr


def test_bash_syntax_clean():
    r = subprocess.run(["bash", "-n", *SHELL_FILES], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
