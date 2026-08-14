"""F-18 regression tests: every DFlash call site requests n-max 15.

Upstream DFlash drafts at most `block_size - 1` = 15 tokens per round: any
`--spec-draft-n-max` request above 15 (16, 32, ...) is clamped down with a
permanently-emitted warning line at every server start, and recorded sweep
labels go off by one (cells labeled n_max=16 and n_max=32 both ran at the
effective 15). The 15-vs-16 perf delta is nil (the project's own sweep is
flat past the elbow), so all three call sites request the effective maximum
directly and the warning disappears at the source.

`docs/results/` is frozen recorded data and still carries the historical
n_max=16 sweep labels by design; these tests pin only the live code and the
living troubleshooting entry.
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from gguf_bench_args import SPEC_DRAFT_N_MAX  # noqa: E402


def _src(relpath: str) -> str:
    return (ROOT / relpath).read_text(encoding="utf-8")


def test_quickstart_requests_upstream_cap_not_the_clamped_16():
    src = _src("scripts/gguf-quickstart.sh")
    assert "SPEC_DRAFT_N_MAX=15" in src
    assert '--spec-draft-n-max "$SPEC_DRAFT_N_MAX"' in src, (
        "the flag must consume the constant, not a second hardcoded copy"
    )
    assert "--spec-draft-n-max 16" not in src


def test_bench_args_constant_is_15_and_comment_states_the_real_cap():
    assert SPEC_DRAFT_N_MAX == 15
    src = _src("scripts/gguf_bench_args.py")
    assert "SPEC_DRAFT_N_MAX = 15" in src
    # The old comment claimed "n_max=32 caps at 16 because 16 is the DFlash
    # block_size" - wrong physics: 16 is *above* the cap, so it (and 32) are
    # clamped to block_size - 1 = 15.
    assert "caps at 16" not in src
    assert "block_size - 1" in src


def test_equiv_harness_requests_15():
    src = _src("scripts/check_dflash_equiv.sh")
    assert "--spec-draft-n-max 15" in src
    assert "--spec-draft-n-max 16" not in src


def test_troubleshooting_states_the_real_cap():
    doc = _src("docs/troubleshooting.md")
    assert "--spec-draft-n-max 15" in doc
    assert "--spec-draft-n-max 16" not in doc
    # The corrected physics: the cap is block_size - 1 = 15. The old entry
    # claimed 16 "equals the DFlash drafter's block_size" and recommended 16.
    assert "block_size - 1" in doc
    assert "equals the DFlash" not in doc
