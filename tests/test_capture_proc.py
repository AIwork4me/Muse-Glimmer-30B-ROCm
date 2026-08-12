import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from capture_proc import parse_proc_status, parse_rocm_smi_power, parse_rocm_smi_vram, scrape_metrics

PROC = """Name:   llama-server
VmPeak:   23499232 kB
VmRSS:    17196612 kB
VmHWM:    18432000 kB
RssShmem:   102400 kB
"""

ROCM = '{"card0": {"Average Graphics Package Power (W)": "45.3", "Temperature (Sensor edge) (C)": "72"}}'

METRICS = """
# HELP llama_speculative_accepted_draft_tokens_total ...
llama_speculative_accepted_draft_tokens_total 312.0
llama_speculative_draft_tokens_total 400.0
llama_speculative_avg_accepted 2.4
"""


def test_parse_proc_status_kib():
    d = parse_proc_status(PROC)
    assert d["VmHWM_kib"] == 18432000
    assert d["VmRSS_kib"] == 17196612
    assert d["VmPeak_kib"] == 23499232
    assert d["VmHWM_gib"] == round(18432000 / 1024 / 1024, 2)
    assert d["VmPeak_gib"] == round(23499232 / 1024 / 1024, 2)


def test_rocm_power_temp():
    d = parse_rocm_smi_power(ROCM)
    assert d["power_w"] == 45.3
    assert d["temp_c"] == 72


def test_scrape_metrics_acceptance():
    d = scrape_metrics(METRICS)
    assert d["accepted_draft_tokens"] == 312.0
    assert d["draft_tokens"] == 400.0
    assert abs(d["acceptance_rate"] - 312.0 / 400.0) < 1e-9


def test_scrape_metrics_empty():
    """Regression test: absent counters should return None values without raising."""
    d = scrape_metrics("")
    assert d["accepted_draft_tokens"] is None
    assert d["draft_tokens"] is None
    assert d["avg_accepted_per_step"] is None
    assert "acceptance_rate" not in d


def test_scrape_metrics_labeled():
    """Prometheus metrics commonly have labels like {model=\"x\"}. Regex should tolerate them."""
    LABELED = """
# HELP llama_speculative_accepted_draft_tokens_total ...
llama_speculative_accepted_draft_tokens_total{model="llama-3-8b"} 312.0
llama_speculative_draft_tokens_total{model="llama-3-8b"} 400.0
llama_speculative_avg_accepted{model="llama-3-8b"} 2.4
"""
    d = scrape_metrics(LABELED)
    assert d["accepted_draft_tokens"] == 312.0
    assert d["draft_tokens"] == 400.0
    assert abs(d["acceptance_rate"] - 312.0 / 400.0) < 1e-9


def test_scrape_metrics_real_llamacpp_names():
    """Real llama-server build 0b1bad1 exposes these names (verified on gfx1151).
    The parser must read them; baseline cells legitimately report zeros."""
    REAL = """
# HELP llamacpp:spec_decode_num_draft_tokens_total Total draft tokens generated
llamacpp:spec_decode_num_draft_tokens_total 0
llamacpp:spec_decode_num_accepted_tokens_total 0
llamacpp:spec_decode_num_drafts_total 0
"""
    d = scrape_metrics(REAL)
    assert d["accepted_draft_tokens"] == 0.0
    assert d["draft_tokens"] == 0.0
    # 0/0 is undefined; parser should not emit a NaN acceptance_rate
    assert "acceptance_rate" not in d


def test_scrape_metrics_real_llamacpp_populated():
    """DFlash cell: non-zero counters should produce an acceptance_rate."""
    POPULATED = """
llamacpp:spec_decode_num_draft_tokens_total 1000
llamacpp:spec_decode_num_accepted_tokens_total 680
llamacpp:spec_decode_num_drafts_total 500
"""
    d = scrape_metrics(POPULATED)
    assert d["accepted_draft_tokens"] == 680.0
    assert d["draft_tokens"] == 1000.0
    assert abs(d["acceptance_rate"] - 0.68) < 1e-9


VRAM = '{"card0": {"VRAM Total Memory (B)": "34359738368", "VRAM Total Used Memory (B)": "18253611008"}}'


def test_parse_rocm_smi_vram():
    d = parse_rocm_smi_vram(VRAM)
    assert d["vram_used_mib"] == round(18253611008 / 1024 / 1024, 1)
    assert d["vram_total_mib"] == round(34359738368 / 1024 / 1024, 1)
