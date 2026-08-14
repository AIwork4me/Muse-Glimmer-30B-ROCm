"""Doc-content assertions for the UX-audit docs cluster (Fix-F).

Mirrors the existing doc-content test pattern (see
``test_dflash_nmax.test_troubleshooting_states_the_real_cap`` and
``test_llama_checkout_guard.test_troubleshooting_has_dirty_llama_checkout_entry``):
the docs promise behavior the scripts rely on, so pin the promise.

F-06/F-07/F-08 - the Quick start must carry the user past a bare server URL:
second-terminal guidance, a verified health + completion request pair, and the
reasoning-first `max_tokens` warning that keeps a first completion non-empty.
F-13 - per-distro host-tool one-liners must exist in README and match the
wording the scripts print on ``required command not found``.
F-17/F-10 - ``MODEL_DEST`` and the optional-asset sizes must be documented
next to the other env knobs / download sizes.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _src(relpath: str) -> str:
    return (ROOT / relpath).read_text(encoding="utf-8")


def _quickstart(readme: str) -> str:
    return readme.split("## Quick start", 1)[1].split("\n## ", 1)[0]


def test_quickstart_tells_the_user_about_the_foreground_server() -> None:
    quickstart = _quickstart(_src("README.md"))
    assert "# OpenAI-compatible server: http://127.0.0.1:8080" in quickstart
    # F-06: the serving command never returns; say so instead of implying
    # "now open this URL" in the occupied terminal.
    assert "Ctrl-C stops the server" in quickstart
    assert "second terminal" in quickstart


def test_quickstart_has_a_verify_it_works_block() -> None:
    quickstart = _quickstart(_src("README.md"))
    assert "### Verify it works" in quickstart
    # F-07: both verified endpoints, with the observed health body shown.
    assert "curl http://127.0.0.1:8080/health" in quickstart
    assert '{"status":"ok"}' in quickstart
    assert "curl http://127.0.0.1:8080/v1/chat/completions" in quickstart
    # F-07+F-08: the example request must budget for hidden reasoning tokens.
    assert '"max_tokens":512' in quickstart


def test_quickstart_documents_reasoning_first_behavior() -> None:
    quickstart = _quickstart(_src("README.md"))
    # F-08: reasoning-first is disclosed near the verify block, including the
    # empty-content-at-small-max_tokens trap and the remedy.
    assert "reasoning-first" in quickstart
    assert "reasoning_content" in quickstart
    assert 'finish_reason:"length"' in quickstart
    assert "max_tokens` ≥ 512" in quickstart
    assert "troubleshooting.md#reasoning-length" in quickstart


def test_requirements_have_per_distro_tool_install_one_liners() -> None:
    requirements = _src("README.md").split(
        "## Requirements and operating notes", 1
    )[1]
    # F-13: same verbs the scripts print on failure (apt-get/dnf/pacman).
    assert "sudo apt-get install git cmake curl python3" in requirements
    assert "sudo dnf install git cmake curl python3" in requirements
    assert "sudo pacman -S git cmake curl python3" in requirements


def test_troubleshooting_has_missing_tool_entry() -> None:
    doc = _src("docs/troubleshooting.md")
    assert "## missing-tool" in doc
    assert "[#missing-tool](#missing-tool)" in doc
    # Discoverable from the checker/installer wording (they print
    # "required command not found: <tool>" and point at this file bare).
    assert "required command not found" in doc


def test_model_dest_is_documented_with_the_other_env_knobs() -> None:
    readme = _src("README.md")
    contract = readme.split("## Reproducibility contract", 1)[1].split(
        "## Hardware validation matrix", 1
    )[0]
    assert "MODEL_DEST=" in contract
    assert "bash scripts/gguf-quickstart.sh" in contract
    # F-17's point: the 15.6 GiB model is reusable across clones.
    assert "across clones" in contract


def test_optional_features_size_their_extra_downloads() -> None:
    readme = _src("README.md")
    # Manifest-exact sizes (1.52 GiB drafter / 1.30 GiB mmproj).
    assert "1.5 GiB" in readme
    assert "1.3 GiB" in readme
    assert "WITH_DFLASH=1" in readme
    assert "WITH_MMPROJ=1" in readme
