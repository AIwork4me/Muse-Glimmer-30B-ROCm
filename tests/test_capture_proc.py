import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from capture_proc import parse_proc_status, parse_rocm_smi_power, scrape_metrics

PROC = """Name:   llama-server
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
    assert d["VmHWM_gib"] == round(18432000 / 1024 / 1024, 2)


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
