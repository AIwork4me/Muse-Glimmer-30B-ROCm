#!/usr/bin/env python3
"""Async throughput probe for an OpenAI-compatible server (vLLM :8000 or
llama.cpp :8080). Fires N concurrent /v1/completions requests and reports
aggregate output tok/s.

Usage:  bench_client.py [base_url] [concurrency] [max_tokens]
Env:    MODEL_NAME (default muse-glimmer), BENCH_PROMPT."""
import asyncio
import json
import os
import sys
import time

import aiohttp


def percentile(values, q):
    """Linear-interpolation percentile, q in [0,100]. None if empty."""
    if not values:
        return None
    xs = sorted(values)
    if len(xs) == 1:
        return xs[0]
    pos = (q / 100.0) * (len(xs) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(xs) - 1)
    frac = pos - lo
    return xs[lo] + (xs[hi] - xs[lo]) * frac


def compute_run_metrics(reqs):
    """Aggregate per-cell metrics from a list of per-request timing dicts.

    Each req: {t0, first_t, last_t, t1, n_tokens, finish_reason}.
    Returns a dict with aggregate tok/s, TTFT p50/p90, TPOT median, and the
    finish_reason distribution. t0/first_t/last_t/t1 are monotonic seconds.
    """
    from statistics import median

    conc = len(reqs)
    total = sum(r["n_tokens"] for r in reqs)
    wall = (max(r["t1"] for r in reqs) - min(r["t0"] for r in reqs)) if reqs else 0.0
    agg = total / wall if wall else 0.0
    ttfts = [r["first_t"] - r["t0"] for r in reqs]
    tpots = [
        (r["last_t"] - r["first_t"]) / max(r["n_tokens"] - 1, 1)
        for r in reqs if r["n_tokens"] > 1
    ]
    from collections import Counter
    dist = dict(Counter(r["finish_reason"] for r in reqs))
    return {
        "concurrency": conc,
        "total_tokens": total,
        "wall_s": wall,
        "agg_tok_s": agg,
        "ttft_p50": percentile(ttfts, 50),
        "ttft_p90": percentile(ttfts, 90),
        "ttft_max": max(ttfts) if ttfts else None,
        "tpot_median": median(tpots) if tpots else None,
        "finish_reason_dist": dist,
    }


def median_run_metrics(reps):
    """Element-wise median across per-rep metric dicts, with agg_tok_s min/max for
    variance reporting (spec §6.2: report median + min/max). reps = list of
    compute_run_metrics() outputs, one per rep."""
    from statistics import median
    if not reps:
        return {}
    keys = ["agg_tok_s", "total_tokens", "wall_s", "ttft_p50", "ttft_p90", "tpot_median"]
    out = {k: median([r[k] for r in reps if r.get(k) is not None]) for k in keys}
    toks = [r["agg_tok_s"] for r in reps]
    out["agg_tok_s_min"], out["agg_tok_s_max"] = min(toks), max(toks)
    out["reps"] = len(reps)
    from collections import Counter
    fr = Counter()
    for r in reps:
        fr.update(r.get("finish_reason_dist", {}))
    out["finish_reason_dist"] = dict(fr)
    return out


async def one(session, base, prompt, max_tokens, model):
    t0 = time.perf_counter()
    async with session.post(
        f"{base}/v1/completions",
        json={"model": model, "prompt": prompt,
              "max_tokens": max_tokens, "temperature": 1.0},
    ) as r:
        data = await r.json()
    dt = time.perf_counter() - t0
    comp = data.get("usage", {}).get("completion_tokens", 0)
    return {"wall_s": dt, "out_tokens": comp, "tok_s": comp / dt if dt else 0}


async def main(base, concurrency, prompt, max_tokens, model):
    async with aiohttp.ClientSession() as s:
        results = await asyncio.gather(*[
            one(s, base, prompt, max_tokens, model)
            for _ in range(concurrency)
        ])
    tot = sum(r["out_tokens"] for r in results)
    wall = max(r["wall_s"] for r in results)
    return {"concurrency": concurrency, "total_out_tokens": tot,
            "wall_s": wall, "agg_tok_s": tot / wall if wall else 0}


if __name__ == "__main__":
    base = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000"
    concurrency = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    max_tokens = int(sys.argv[3]) if len(sys.argv) > 3 else 512
    model = os.environ.get("MODEL_NAME", "muse-glimmer")
    prompt = os.environ.get(
        "BENCH_PROMPT", "Summarize the plot of Hamlet in three sentences.")
    print(json.dumps(asyncio.run(
        main(base, concurrency, prompt, max_tokens, model))))
