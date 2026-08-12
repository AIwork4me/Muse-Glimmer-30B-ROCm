import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from capture_proc import parse_proc_status, parse_rocm_smi_power, parse_rocm_smi_vram, scrape_metrics, parse_draft_acceptance

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
    """DFlash cell: non-zero REAL counters must produce avg_accepted_per_step AND
    acceptance_rate. avg_accepted_per_step = accepted/drafts (per verification
    STEP), not accepted/draft_tokens. Locks the 2026-08-12 fix that replaced the
    non-existent `llamacpp:spec_decode_avg_accepted` gauge with the derived ratio."""
    POPULATED = """
llamacpp:spec_decode_num_draft_tokens_total 1000
llamacpp:spec_decode_num_accepted_tokens_total 680
llamacpp:spec_decode_num_drafts_total 500
"""
    d = scrape_metrics(POPULATED)
    assert d["accepted_draft_tokens"] == 680.0
    assert d["draft_tokens"] == 1000.0
    # per-step average: 680 accepted across 500 verification steps
    assert d["avg_accepted_per_step"] is not None
    assert abs(d["avg_accepted_per_step"] - 680.0 / 500.0) < 1e-9
    # token-level acceptance rate: 680 / 1000 drafted
    assert abs(d["acceptance_rate"] - 0.68) < 1e-9


def test_scrape_metrics_real_llamacpp_zero_drafts():
    """Regression for the fix: baseline cells report num_drafts_total=0 (spec
    decoding inactive). avg_accepted_per_step MUST be null, never NaN (0/0)."""
    ZERO = """
llamacpp:spec_decode_num_draft_tokens_total 0
llamacpp:spec_decode_num_accepted_tokens_total 0
llamacpp:spec_decode_num_drafts_total 0
"""
    d = scrape_metrics(ZERO)
    assert d["accepted_draft_tokens"] == 0.0
    assert d["draft_tokens"] == 0.0
    assert d["avg_accepted_per_step"] is None
    assert "acceptance_rate" not in d


VRAM = '{"card0": {"VRAM Total Memory (B)": "34359738368", "VRAM Total Used Memory (B)": "18253611008"}}'


def test_parse_rocm_smi_vram():
    d = parse_rocm_smi_vram(VRAM)
    assert d["vram_used_mib"] == round(18253611008 / 1024 / 1024, 1)
    assert d["vram_total_mib"] == round(34359738368 / 1024 / 1024, 1)


# Authoritative acceptance source per the 2026-08-12 DFlash enablement fix:
# /metrics spec counters stay 0 even when spec-decoding is active (build
# instrumentation gap), so the SERVER LOG's per-slot print_timing line is the
# primary source. Format (build 0b1bad1):
#   ... slot print_timing: id  0 | task 0 | draft acceptance = 0.14996
#       (  175 accepted /  1167 generated), mean len =   3.19
DRAFT_LOG = """\
info: work loaded
slot update      id:  0 task: 0 ...
slot print_timing: id  0 | task 0 | draft acceptance = 0.14996 (  175 accepted /  1167 generated), mean len =   3.19
slot print_timing: id  0 | task 0 | draft acceptance = 0.20000 (  200 accepted /  1000 generated), mean len =   4.00
wrap up
"""


def test_parse_draft_acceptance_multi_line_sums():
    """Two print_timing lines: accepted and generated must SUM across lines;
    acceptance_rate is the pooled ratio (sum/sum), avg_accepted_per_step is the
    arithmetic mean of the per-line `mean len` values."""
    d = parse_draft_acceptance(DRAFT_LOG)
    assert d["accepted_draft_tokens"] == 375       # 175 + 200
    assert d["draft_tokens"] == 2167               # 1167 + 1000
    assert d["acceptance_rate"] is not None
    assert abs(d["acceptance_rate"] - 375 / 2167) < 1e-9
    assert d["avg_accepted_per_step"] is not None
    assert abs(d["avg_accepted_per_step"] - (3.19 + 4.00) / 2) < 1e-9


def test_parse_draft_acceptance_single_line():
    ONE = ("slot print_timing: id  0 | task 0 | "
           "draft acceptance = 0.14996 (  175 accepted /  1167 generated), mean len =   3.19\n")
    d = parse_draft_acceptance(ONE)
    assert d["accepted_draft_tokens"] == 175
    assert d["draft_tokens"] == 1167
    assert abs(d["acceptance_rate"] - 175 / 1167) < 1e-9
    assert abs(d["avg_accepted_per_step"] - 3.19) < 1e-9


def test_parse_draft_acceptance_empty_returns_none():
    """No print_timing lines -> all fields None, no crash, no NaN."""
    d = parse_draft_acceptance("")
    assert d["accepted_draft_tokens"] is None
    assert d["draft_tokens"] is None
    assert d["acceptance_rate"] is None
    assert d["avg_accepted_per_step"] is None


def test_parse_draft_acceptance_no_match_returns_none():
    """Log present but no draft-acceptance lines (e.g. baseline cell): no crash,
    all None. Crucially a baseline must NOT be misread as a 0-token DFlash run."""
    d = parse_draft_acceptance("server started\nserver idle\nno spec lines here\n")
    assert d["accepted_draft_tokens"] is None
    assert d["draft_tokens"] is None
    assert d["acceptance_rate"] is None
    assert d["avg_accepted_per_step"] is None


def test_parse_draft_acceptance_zero_generated_no_nan():
    """Defensive: a pathological `0 generated` line must not produce a NaN or
    ZeroDivisionError. acceptance_rate is null when total generated is 0."""
    ZERO = ("slot print_timing: id  0 | task 0 | "
            "draft acceptance = 0.00000 (  0 accepted /  0 generated), mean len =   0.00\n")
    d = parse_draft_acceptance(ZERO)
    assert d["accepted_draft_tokens"] == 0
    assert d["draft_tokens"] == 0
    assert d["acceptance_rate"] is None   # 0/0 undefined
    assert d["avg_accepted_per_step"] == 0.0
