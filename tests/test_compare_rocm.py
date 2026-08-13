"""Regression tests for evidence-safe ROCm matrix comparison."""

import json
import os
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from compare_rocm import load_matrix, render  # noqa: E402


def cell(*, np=1, tpot=0.1, ttft=0.25, tok_s=10.0):
    return {
        "study": "study1",
        "weight": "17gb",
        "np": np,
        "dflash": False,
        "vision": False,
        "metrics": {
            "agg_tok_s": tok_s,
            "ttft_p50": ttft,
            "ttft_p90": ttft + 0.01,
            "tpot_median": tpot,
        },
        "mem": {"VmPeak_gib": 24.0},
        "acceptance": {},
    }


def key(record):
    return (
        record["study"],
        record["weight"],
        record["np"],
        record["dflash"],
        record["vision"],
    )


def write_cell(path, record):
    path.write_text(json.dumps(record), encoding="utf-8")


def test_render_converts_recorded_seconds_to_display_milliseconds():
    before = cell(tpot=0.1, ttft=0.25)
    after = cell(tpot=0.09, ttft=0.2)

    output = render({key(before): before}, {key(after): after}, "A", "B")

    assert "A TTFT p50 ms" in output
    assert "250.0 | 200.0 | -20.0%" in output
    assert "100.0 | 90.0 | -10.0%" in output


def test_render_uses_tpot_summary_and_marks_aggregate_as_confounded():
    before = cell(tpot=0.1, tok_s=10.0)
    after = cell(tpot=0.09, tok_s=20.0)

    output = render({key(before): before}, {key(after): after}, "A", "B")

    assert "## Summary (TPOT, both arms measured)" in output
    assert "np=1: n=1, mean Δ **-10.0%**" in output
    assert "generation-length-confounded" in output
    assert "Summary (tok/s" not in output


def test_load_matrix_rejects_malformed_json(tmp_path):
    (tmp_path / "cell-broken.json").write_text("{", encoding="utf-8")

    with pytest.raises(ValueError, match="cannot parse benchmark cell"):
        load_matrix(str(tmp_path))


def test_load_matrix_rejects_missing_identity(tmp_path):
    write_cell(tmp_path / "cell-missing.json", {"study": "study1"})

    with pytest.raises(ValueError, match="lacks identity fields"):
        load_matrix(str(tmp_path))


def test_load_matrix_rejects_non_object_json(tmp_path):
    write_cell(tmp_path / "cell-list.json", [])

    with pytest.raises(ValueError, match="must contain a JSON object"):
        load_matrix(str(tmp_path))


def test_load_matrix_rejects_coercible_identity_types(tmp_path):
    record = cell()
    record["dflash"] = "false"
    write_cell(tmp_path / "cell-wrong-type.json", record)

    with pytest.raises(ValueError, match="invalid identity field types"):
        load_matrix(str(tmp_path))


def test_load_matrix_rejects_duplicate_identity(tmp_path):
    record = cell()
    write_cell(tmp_path / "cell-a.json", record)
    write_cell(tmp_path / "cell-b.json", record)

    with pytest.raises(ValueError, match="duplicate benchmark cell identity"):
        load_matrix(str(tmp_path))


def test_render_reports_one_sided_cells():
    before = cell(np=16)

    output = render({key(before): before}, {}, "A", "B")

    assert "⚠ no B cell" in output


def test_committed_comparison_is_current():
    before = load_matrix(str(ROOT / "docs/results/matrix"))
    after = load_matrix(str(ROOT / "docs/results/matrix-714"))
    expected = render(before, after, "7.2.1", "7.14.0")
    committed = (ROOT / "docs/results/matrix-714/comparison.md").read_text(
        encoding="utf-8"
    )

    assert committed == expected
