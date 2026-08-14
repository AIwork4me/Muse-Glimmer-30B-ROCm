#!/usr/bin/env python3
"""Compare two GGUF benchmark matrices cell-by-cell (ROCm 7.2.1 vs 7.14.0).

Joins per-cell JSON records on (study, weight, np, dflash, vision) across two
matrix directories and renders a markdown table of the headline-metric deltas:
aggregate tok/s, TTFT p50/p90, TPOT median, VmPeak and DFlash acceptance.
Pathological (non-completing) and one-sided cells are reported, not dropped.
Malformed or duplicate cells fail closed.

Usage:
    scripts/compare_rocm.py [--a DIR] [--b DIR] [--label-a NAME] [--label-b NAME]
      DIR defaults: docs/results/matrix (a=7.2.1) and docs/results/matrix-714 (b=7.14.0)

Designed to be run after the 7.14.0 matrix exists; degrades gracefully when the
7.14 dir is empty or partially populated (reduced-scope first pass = 17 cells).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from statistics import mean

HERE = os.path.dirname(os.path.abspath(__file__))


def load_matrix(d: str) -> dict:
    """Load cells keyed by (study, weight, np, dflash, vision), failing closed."""
    out = {}
    for f in sorted(glob.glob(os.path.join(d, "cell-*.json"))):
        try:
            with open(f, encoding="utf-8") as handle:
                record = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"cannot parse benchmark cell {f}: {exc}") from exc
        if not isinstance(record, dict):
            raise ValueError(f"benchmark cell {f} must contain a JSON object")
        identity = ("study", "weight", "np", "dflash", "vision")
        missing = [name for name in identity if name not in record]
        if missing:
            raise ValueError(f"benchmark cell {f} lacks identity fields: {missing}")
        valid_types = (
            isinstance(record["study"], str),
            isinstance(record["weight"], str),
            isinstance(record["np"], int) and not isinstance(record["np"], bool),
            isinstance(record["dflash"], bool),
            isinstance(record["vision"], bool),
        )
        if not all(valid_types):
            raise ValueError(f"benchmark cell {f} has invalid identity field types")
        key = tuple(record[name] for name in identity)
        if key in out:
            raise ValueError(f"duplicate benchmark cell identity {key}: {f}")
        out[key] = record
    return out


def num(rec: dict, *path, default=None):
    """Drill into nested dict by path; return float or default."""
    cur = rec
    for p in path:
        if not isinstance(cur, dict) or p not in cur or cur[p] is None:
            return default
        cur = cur[p]
    try:
        return float(cur)
    except (TypeError, ValueError):
        return default


def pct(a, b):
    """Δ% of b relative to a: (b-a)/a*100. None if either missing or a==0."""
    if a is None or b is None or a == 0:
        return None
    return (b - a) / a * 100.0


def fmt(x, prec=2):
    return f"{x:.{prec}f}" if isinstance(x, (int, float)) else "—"


def fmtp(x, prec=1):
    """Format a percentage with explicit + sign."""
    if x is None:
        return "—"
    return f"{x:+.{prec}f}%"


def cell_label(rec) -> str:
    w = rec["weight"]
    tag = " +DFlash" if rec["dflash"] else " baseline"
    vis = " +vision" if rec["vision"] else ""
    return f"{w} np{rec['np']}{tag}{vis}"


def tpot_deltas_by_concurrency(a: dict, b: dict) -> dict[int, list[float]]:
    """Return comparable, non-pathological TPOT deltas grouped by concurrency."""
    grouped: dict[int, list[float]] = {}
    for key in sorted(set(a) & set(b)):
        before, after = a[key], b[key]
        if before.get("pathological") or after.get("pathological"):
            continue
        tpot_before = num(before, "metrics", "tpot_median")
        tpot_after = num(after, "metrics", "tpot_median")
        if tpot_before is None or tpot_after is None:
            continue
        delta = pct(tpot_before, tpot_after)
        if delta is not None:
            grouped.setdefault(key[2], []).append(delta)
    return grouped


METRICS = [
    # (header, JSON path, display precision, display scale)
    ("aggregate tok/s", ("metrics", "agg_tok_s"), 2, 1.0),
    ("TTFT p50 ms", ("metrics", "ttft_p50"), 1, 1000.0),
    ("TTFT p90 ms", ("metrics", "ttft_p90"), 1, 1000.0),
    ("TPOT ms", ("metrics", "tpot_median"), 1, 1000.0),
    ("VmPeak GiB", ("mem", "VmPeak_gib"), 2, 1.0),
]


def render(a: dict, b: dict, la: str, lb: str) -> str:
    keys = sorted(set(a) | set(b), key=lambda k: (k[0], k[1], k[2], int(k[3]), int(k[4])))
    lines = []
    lines.append(f"# ROCm {la} vs {lb} — cell-by-cell comparison\n")
    lines.append(f"- **{la}:** {len(a)} cells   **{lb}:** {len(b)} cells   **compared:** {len(set(a) & set(b))}\n")

    # TPOT is less confounded by sampling-length divergence than aggregate tok/s.
    d_tpot = tpot_deltas_by_concurrency(a, b)
    if d_tpot:
        lines.append("## Summary (TPOT, both arms measured)\n")
        lines.append("- TPOT is the primary, less length-confounded cross-version metric; "
                     "negative Δ means lower per-token decode latency.")
        for concurrency in sorted(d_tpot):
            deltas = d_tpot[concurrency]
            lines.append(f"- np={concurrency}: n={len(deltas)}, mean Δ "
                         f"**{fmtp(mean(deltas))}**, range "
                         f"{fmtp(min(deltas))} … {fmtp(max(deltas))}")
        lines.append("- Aggregate tok/s remains in the tables, but sampled Study 2/3 "
                     "comparisons can be generation-length-confounded.\n")

    by_study = {}
    for k in keys:
        by_study.setdefault(k[0], []).append(k)

    for st in sorted(by_study):
        lines.append(f"## {st}\n")
        hdr = ["cell"]
        for name, _path, _precision, _scale in METRICS:
            hdr += [f"{la} {name}", f"{lb} {name}", "Δ"]
        hdr.append(f"{lb} accept")
        lines.append("| " + " | ".join(hdr) + " |")
        lines.append("|" + "|".join(["---"] * len(hdr)) + "|")
        for k in by_study[st]:
            ra, rb = a.get(k), b.get(k)
            row = [cell_label(rb or ra)]
            for _name, path, precision, scale in METRICS:
                raw_a = num(ra, *path) if ra else None
                raw_b = num(rb, *path) if rb else None
                delta = pct(raw_a, raw_b)
                va = raw_a * scale if raw_a is not None else None
                vb = raw_b * scale if raw_b is not None else None
                row += [fmt(va, precision), fmt(vb, precision), fmtp(delta)]
            # acceptance (DFlash cells)
            acc = num(rb, "acceptance", "acceptance_rate") if rb and rb["dflash"] else None
            row.append(fmt(acc * 100, 1) + "%" if acc is not None else "—")
            # flag pathological / missing
            notes = []
            if ra is None:
                notes.append(f"⚠ no {la} cell")
            if rb is None:
                notes.append(f"⚠ no {lb} cell")
            if ra and ra.get("pathological"):
                notes.append(f"{la}: pathological")
            if rb and rb.get("pathological"):
                notes.append(f"{lb}: pathological")
            if notes:
                row[0] = row[0] + "  " + "; ".join(notes)
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")

    lines.append("## Notes\n")
    lines.append(f"- Δ = ({lb} − {la}) / {la} × 100. **Positive Δ on aggregate tok/s = {lb} is faster**; "
                 f"positive Δ on TTFT/TPOT/VmPeak = {lb} is slower/higher.")
    lines.append("- `—` = metric absent (baseline cells have no acceptance; pathological cells have no metrics).")
    lines.append("- Recorded invariants across arms: llama.cpp commit, flags, weights, prompt set and seeds; "
                 "the comparison intentionally changes the ROCm runtime.")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", default="docs/results/matrix", help="matrix dir A (default 7.2.1)")
    ap.add_argument("--b", default="docs/results/matrix-714", help="matrix dir B (default 7.14.0)")
    ap.add_argument("--label-a", default="7.2.1")
    ap.add_argument("--label-b", default="7.14.0")
    ap.add_argument("-o", "--out", default=None, help="also write markdown to this file")
    args = ap.parse_args()

    try:
        a = load_matrix(args.a)
        b = load_matrix(args.b)
    except ValueError as exc:
        ap.error(str(exc))
    if not a:
        sys.exit(f"no cells found in --a {args.a}")
    md = render(a, b, args.label_a, args.label_b)
    print(md)
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as handle:
            handle.write(md)
        print(f"\n(wrote {args.out})", file=sys.stderr)


if __name__ == "__main__":
    main()
