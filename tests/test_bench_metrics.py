"""Unit tests for bench_client metric math (pure, no server needed)."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))


def test_percentile_linear_interp():
    from bench_client import percentile
    assert percentile([0.5, 0.6], 50) == 0.55
    assert percentile([1.0, 2.0, 3.0, 4.0], 90) == 3.7  # 0.9*(4-1)=2.7 -> idx interp
    assert percentile([], 50) is None


def test_compute_run_metrics_aggregate_and_distribution():
    from bench_client import compute_run_metrics
    # 2 concurrent requests, c=2
    reqs = [
        {"t0": 0.0, "first_t": 0.5, "last_t": 10.5, "t1": 10.6,
         "n_tokens": 100, "finish_reason": "stop"},
        {"t0": 0.0, "first_t": 0.6, "last_t": 12.0, "t1": 12.1,
         "n_tokens": 120, "finish_reason": "stop"},
    ]
    m = compute_run_metrics(reqs)
    assert m["concurrency"] == 2
    assert m["total_tokens"] == 220
    assert abs(m["wall_s"] - 12.1) < 1e-9          # max t1 - min t0
    assert abs(m["agg_tok_s"] - 220 / 12.1) < 1e-6
    assert abs(m["ttft_p50"] - 0.55) < 1e-9
    assert m["finish_reason_dist"] == {"stop": 2}
    # tpot per request = gen_phase / (n_tokens-1)
    tpot_a = (10.5 - 0.5) / 99
    tpot_b = (12.0 - 0.6) / 119
    assert abs(m["tpot_median"] - sorted([tpot_a, tpot_b])[0]) < 1e-9 or \
           abs(m["tpot_median"] - (tpot_a + tpot_b) / 2) < 1e-9


def test_median_run_metrics_reports_variance():
    from bench_client import median_run_metrics
    reps = [
        {"agg_tok_s": 10.0, "total_tokens": 100, "wall_s": 10.0, "ttft_p50": 0.4,
         "ttft_p90": 0.5, "tpot_median": 0.09, "finish_reason_dist": {"stop": 1}},
        {"agg_tok_s": 12.0, "total_tokens": 120, "wall_s": 10.0, "ttft_p50": 0.4,
         "ttft_p90": 0.5, "tpot_median": 0.08, "finish_reason_dist": {"stop": 1}},
        {"agg_tok_s": 14.0, "total_tokens": 140, "wall_s": 10.0, "ttft_p50": 0.4,
         "ttft_p90": 0.5, "tpot_median": 0.07, "finish_reason_dist": {"stop": 1}},
    ]
    m = median_run_metrics(reps)
    assert m["agg_tok_s"] == 12.0          # median across reps
    assert m["agg_tok_s_min"] == 10.0
    assert m["agg_tok_s_max"] == 14.0
    assert m["reps"] == 3
    assert m["finish_reason_dist"] == {"stop": 3}
