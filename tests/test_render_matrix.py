import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from render_matrix import render_studies

CELLS = [
    {"study": "study1", "weight": "17gb", "dflash": False, "vision": False, "np": 1,
     "metrics": {"agg_tok_s": 10.5, "ttft_p50": 0.4, "tpot_median": 0.09, "total_tokens": 1536},
     "mem": {"VmPeak_gib": 17.2, "weights_gib": 16.8, "kv_gib": 0.4},
     "acceptance": None, "reps": 3, "seed": 0},
    {"study": "study1", "weight": "17gb", "dflash": True, "vision": False, "np": 1,
     "metrics": {"agg_tok_s": 22.0, "ttft_p50": 0.4, "tpot_median": 0.045, "total_tokens": 1536},
     "mem": {"VmPeak_gib": 18.8, "weights_gib": 16.8, "kv_gib": 0.4},
     "acceptance": {"acceptance_rate": 0.78, "avg_accepted_per_step": 2.1},
     "reps": 3, "seed": 0},
    {"study": "study2", "weight": "17gb", "dflash": False, "vision": False, "np": 4,
     "metrics": {"agg_tok_s": 15.2, "ttft_p50": 0.35, "ttft_p90": 0.42, "tpot_median": 0.07, "total_tokens": 1536},
     "mem": {"VmPeak_gib": 18.5, "weights_gib": 16.8, "kv_gib": 1.7},
     "acceptance": None, "reps": 3, "seed": 0},
    # pathological c16 DFlash cell (did not complete) — must render a warning, not crash
    {"study": "study2", "weight": "17gb", "dflash": True, "vision": False, "np": 16,
     "pathological": True,
     "metrics": {"agg_tok_s": None, "ttft_p90": None, "tpot_median": None, "total_tokens": 0},
     "mem": {}, "acceptance": None,
     "pathological_evidence": {"verdict": "did not complete"}},
]


def test_render_pathological_cell_warns_not_crashes():
    md = render_studies(CELLS)
    assert "PATHOLOGICAL" in md
    # the pathological cell must NOT try to format None agg_tok_s (would crash)
    assert "did not complete" in md


def test_render_study1_has_speedup_column():
    md = render_studies(CELLS)
    assert "### Study 1" in md
    assert "Speedup" in md
    # speedup = 22.0 / 10.5 = 2.095...
    assert "2.10" in md
    assert "78%" in md               # acceptance rendered as percent


def test_render_handles_empty():
    assert "no cells" in render_studies([]).lower()


def test_render_study2_ttft_p90():
    md = render_studies(CELLS)
    assert "### Study 2" in md
    # ttft_p90 = 0.42 should be formatted as 0.42
    assert "0.42" in md
