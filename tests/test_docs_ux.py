"""Doc-content assertions for the UX-audit docs cluster (Fix-F).

Mirrors the existing doc-content test pattern (see
``test_dflash_nmax.test_troubleshooting_states_the_real_cap`` and
``test_llama_checkout_guard.test_troubleshooting_has_dirty_llama_checkout_entry``):
the docs promise behavior the scripts rely on, so pin the promise.

F-06/F-07/F-08 - the Quick start must carry the user past a bare server URL:
second-terminal guidance, a verified health + completion request pair, and the
reasoning-first `max_tokens` warning that keeps a first completion non-empty.
F-13 - per-distro host-tool one-liners must exist in README, which must also
describe what the scripts actually print on ``required command not found``
(the installer and checker name the single missing tool; the quickstart
prints no install hint).
F-17/F-10 - ``MODEL_DEST`` and the optional-asset sizes must be documented
next to the other env knobs / download sizes.
Fold-ins - the dirty-checkout entry's mid-switch recovery, the corrected
``--spec-draft-n-max 15`` in adaptation.md, and the hardware-validation
BIOS/UMA sizing pointer.
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
    # F-13 review follow-up: only the installer and checker print a hint, and
    # it names the single missing tool — the quickstart prints none. The docs
    # must describe that, not claim the failure text prints these blocks.
    assert "per-distro install command" in requirements
    assert "same one-liners" not in requirements


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


def test_overrides_banner_scopes_the_experimental_track_to_revision_overrides() -> None:
    readme = _src("README.md")
    contract = readme.split("## Reproducibility contract", 1)[1].split(
        "## Hardware validation matrix", 1
    )[0]
    banner = contract.split("```bash", 1)[0]
    # Final-review Minor-2: only revision overrides move a run to the
    # reported-as-latest/experimental track (what the scripts actually print);
    # HF_ENDPOINT/MODEL_DEST are transport/location knobs that keep the
    # validated manifest checks and must not sit under that sentence.
    assert "latest/experimental" in banner
    for knob in ("MODEL_REVISION", "LLAMA_CPP_REF", "GGUF_REVISION"):
        assert knob in banner, f"{knob} must be named as the experimental track"
    for knob in ("HF_ENDPOINT", "MODEL_DEST"):
        assert knob in banner, f"{knob} must be named as a transport/location knob"
    assert "Overrides are explicit and reported" not in banner, (
        "the blanket 'overrides ... reported as latest/experimental' wording"
        " mislabels MODEL_DEST/HF_ENDPOINT, which never leave the validated"
        " hash-checked track"
    )


def test_optional_features_size_their_extra_downloads() -> None:
    readme = _src("README.md")
    # Manifest-exact sizes (1.52 GiB drafter / 1.30 GiB mmproj).
    assert "1.5 GiB" in readme
    assert "1.3 GiB" in readme
    assert "WITH_DFLASH=1" in readme
    assert "WITH_MMPROJ=1" in readme


def test_troubleshooting_dirty_checkout_covers_mid_switch_interruption() -> None:
    doc = _src("docs/troubleshooting.md")
    entry = doc.split("## dirty-llama-cpp-checkout", 1)[1].split("\n## ", 1)[0]
    # Fix-A fold: an interrupted commit switch may need more than
    # `checkout -- .`; the reset/remove alternatives must be documented.
    assert "reset --hard" in entry
    assert "rm -rf third_party/llama.cpp" in entry


def test_adaptation_states_the_real_dflash_cap() -> None:
    doc = _src("docs/adaptation.md")
    assert "--spec-draft-n-max 15" in doc
    assert "--spec-draft-n-max 16" not in doc


def test_hardware_validation_has_bios_uma_sizing_pointer() -> None:
    doc = _src("docs/hardware-validation.md")
    # F-14 fold: hosts below the 80 GiB envelope need the checker thresholds
    # and the README requirements, not silence.
    assert "80 GiB" in doc
    assert "00-check-env.sh" in doc
    assert "README.md#requirements-and-operating-notes" in doc
    # Matches the checker's verbatim term (00-check-env.sh prints
    # "validated envelope 80 GiB GPU-visible — warning boundary, not a minimum").
    assert "warning boundary, not a minimum" in doc
