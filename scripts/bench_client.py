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
    # For each key, take the median of the non-None values across reps; if a key
    # is None for every rep (e.g. tpot_median when no rep produced >1 token),
    # emit null rather than raising median-of-empty.
    out = {}
    for k in keys:
        vals = [r[k] for r in reps if r.get(k) is not None]
        out[k] = median(vals) if vals else None
    toks = [r["agg_tok_s"] for r in reps if r.get("agg_tok_s") is not None]
    if toks:
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


import argparse

try:
    import base64
except ImportError:
    base64 = None


async def stream_one(session, base, payload):
    """Stream one request; return per-request timing dict. endpoint in payload meta."""
    t0 = time.perf_counter()
    first_t = last_t = None
    n_tokens = 0
    finish = None
    url = f"{base}{'/v1/chat/completions' if payload['_endpoint']=='chat' else '/v1/completions'}"
    body = {k: v for k, v in payload.items() if not k.startswith("_")}
    body["stream"] = True
    # Without include_usage, OpenAI-compatible servers (llama.cpp, vLLM) omit
    # the token-count chunk while streaming, leaving n_tokens=0 and forcing a
    # degenerate n_tokens=1 fallback that corrupts agg_tok_s and nulls
    # tpot_median. Require the usage chunk so metrics are real.
    body["stream_options"] = {"include_usage": True}
    async with session.post(url, json=body) as r:
        # Surface server errors (5xx etc.) explicitly. Without this a server
        # failure yields an empty read → n_tokens falls back to 1 → a corrupt
        # "fast" cell. The legacy one() path already reads JSON + raises.
        r.raise_for_status()
        # SSE framing: events are newline-delimited, but aiohttp's content
        # iterator yields ARBITRARY byte chunks — one chunk may carry several
        # events or a partial one. Treating each raw chunk as one event parses
        # only the chunks that happen to align with a single line; every
        # coalesced event (usage chunks included) silently fails json.loads.
        # At np=1/np=4 chunks rarely coalesce; at np=16 they mostly do, so the
        # np=16 17gb baseline cell of 2026-08-15 recorded 96 tokens where the
        # server log showed ~174k generated. Buffer and split on newlines so
        # parsing is independent of chunk boundaries.
        def _feed(line):
            # Returns True when the [DONE] sentinel was seen. Sets first_t/
            # last_t/n_tokens/finish on the caller's locals.
            nonlocal first_t, last_t, n_tokens, finish
            line = line.strip()
            if not line.startswith("data:"):
                return False
            data = line[5:].strip()
            if data == "[DONE]":
                return True
            try:
                obj = json.loads(data)
            except json.JSONDecodeError:
                return False
            now = time.perf_counter()
            if first_t is None:
                first_t = now
            last_t = now
            # count tokens from usage (last chunk) or choices delta
            if obj.get("usage"):
                n_tokens = obj["usage"].get("completion_tokens", n_tokens)
            ch = obj.get("choices") or []
            if ch and "finish_reason" in ch[0] and ch[0]["finish_reason"]:
                finish = ch[0]["finish_reason"]
            return False

        buf = ""
        done = False
        async for raw in r.content:
            buf += raw.decode(errors="ignore")
            while "\n" in buf:
                line, buf = buf.split("\n", 1)
                if _feed(line):
                    done = True
                    break
            if done:
                break
        if not done and buf:
            # EOF mid-line: parse the unterminated final event, matching the
            # pre-buffering behavior of treating a whole chunk as one line.
            _feed(buf)
    t1 = time.perf_counter()
    if n_tokens == 0 and last_t:  # fallback: no usage reported
        n_tokens = 1
    return {"t0": t0, "first_t": first_t or t0, "last_t": last_t or t0,
            "t1": t1, "n_tokens": n_tokens, "finish_reason": finish or "stop"}


def _payload(cell_args, prompt_text, image_b64=None):
    p = {"_endpoint": cell_args["endpoint"], "model": "muse-glimmer-30B",
         "max_tokens": cell_args["max_tokens"], "temperature": cell_args["temp"],
         "top_p": cell_args["top_p"], "top_k": cell_args["top_k"],
         "seed": cell_args["seed"]}
    if cell_args["endpoint"] == "chat":
        content = ([{"type": "text", "text": prompt_text}] if image_b64 is None
                   else [{"type": "text", "text": prompt_text},
                         {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}}])
        p["messages"] = [{"role": "user", "content": content}]
        p["chat_template_kwargs"] = {"reasoning_strength": cell_args["reasoning_strength"]}
    else:
        p["prompt"] = prompt_text
    return p


async def run_cell(base, cell_args, prompts, image_b64=None):
    """Run one cell: warmup, then REPS rounds (each round fires np concurrent requests
    per prompt). Returns a list of per-rep metric dicts — one compute_run_metrics() per
    rep — so the caller can take the median + report min/max (spec §6.2)."""
    conc = cell_args["np"]
    per_rep = []
    # High-concurrency cells (np=16) on a 30B APU model can keep a single request
    # decoding for many minutes under batch contention; the aiohttp DEFAULT
    # total=300s aborted those mid-read (verified 2026-08-12: c=16 cells died
    # with TimeoutError while the server stayed alive). At c=16 the TAIL latency
    # of the slowest of 16 concurrent requests (heavy contention + DFlash
    # verification overhead + 512 tokens) can exceed 600s between chunks, so
    # sock_read must be generous. total=None (no wall cap — a legitimate c=16
    # decode may take 10-20+ min); sock_read=1200s catches a genuinely dead
    # server within 20 min. This does NOT change the measurement —
    # ttft/tpot/agg_tok_s come from perf_counter markers in stream_one,
    # independent of this client-side ceiling.
    timeout = aiohttp.ClientTimeout(total=None, sock_connect=30, sock_read=1200)
    async with aiohttp.ClientSession(timeout=timeout) as s:
        for _ in range(cell_args.get("warmup", 2)):
            await asyncio.gather(*[stream_one(s, base, _payload(cell_args, prompts[0]["text"], image_b64))
                                   for _ in range(conc)])
        for _ in range(cell_args["reps"]):
            rep_reqs = []
            for pr in prompts:
                rep_reqs.extend(await asyncio.gather(*[
                    stream_one(s, base, _payload(cell_args, pr["text"], image_b64))
                    for _ in range(conc)]))
            per_rep.append(compute_run_metrics(rep_reqs))
    return per_rep


def main_extended():
    ap = argparse.ArgumentParser()
    ap.add_argument("base")
    ap.add_argument("--study", required=True)
    ap.add_argument("--endpoint", choices=["chat", "completions"], default="chat")
    ap.add_argument("--prompts", default="scripts/prompt-sets/muse-glimmer-diverse.json")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--np", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--temp", type=float, default=0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--top-k", type=int, default=0)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--reasoning-strength", default="high")
    ap.add_argument("--image", default=None)
    args = ap.parse_args()
    cell_args = vars(args)
    prompts = json.load(open(args.prompts))["prompts"]
    img = None
    if args.image:
        img = base64.b64encode(open(args.image, "rb").read()).decode()
    per_rep = asyncio.run(run_cell(args.base, cell_args, prompts, img))
    print(json.dumps(median_run_metrics(per_rep)))


def _legacy_main():
    """Legacy positional CLI: bench_client.py BASE C 512 — used by scripts/benchmark.sh.
    Emits {"concurrency","total_out_tokens","wall_s","agg_tok_s"}. Does its own
    asyncio.run + print, so call it directly (NOT asyncio.run(_legacy_main()))."""
    base = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000"
    concurrency = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    max_tokens = int(sys.argv[3]) if len(sys.argv) > 3 else 512
    model = os.environ.get("MODEL_NAME", "muse-glimmer")
    prompt = os.environ.get(
        "BENCH_PROMPT", "Summarize the plot of Hamlet in three sentences.")
    print(json.dumps(asyncio.run(
        main(base, concurrency, prompt, max_tokens, model))))


if __name__ == "__main__":
    # Legacy positional path "bench_client.py BASE C 512" -> scripts/benchmark.sh (unchanged).
    # Everything else routes to the extended streaming CLI.
    if len(sys.argv) > 2 and sys.argv[1].startswith("http") and sys.argv[2].isdigit():
        _legacy_main()           # wraps the ORIGINAL __main__ block (does its own asyncio.run)
    else:
        main_extended()
