#!/usr/bin/env python3
"""Compare two GGUF benchmark matrices cell-by-cell (ROCm 7.2.1 vs 7.14.0).

Joins per-cell JSON records on (study, weight, np, dflash, vision) across two
matrix directories and renders a markdown table of the headline-metric deltas:
tok/s (median + min/max), TTFT p50/p90, TPOT median, VmPeak, DFlash acceptance,
temp. Pathological (non-completing) and one-sided cells are reported, not dropped.

Usage:
    scripts/compare_rocm.py [--a DIR] [--b DIR] [--label-a NAME] [--label-b NAME]
      DIR defaults: docs/results/matrix (a=7.2.1) and docs/results/matrix-714 (b=7.14.0)

Designed to be run after the 7.14.0 matrix exists; degrades gracefully when the
7.14 dir is empty or partially populated (reduced-scope first pass = 17 cells).
"""
from __future__ import annotations
import argparse, glob, json, os, sys
from statistics import mean

HERE = os.path.dirname(os.path.abspath(__file__))


def load_matrix(d: str) -> dict:
    """Return {key: record} for every cell-*.json in dir d. key = (study,weight,np,dflash,vision)."""
    out = {}
    for f in sorted(glob.glob(os.path.join(d, "cell-*.json"))):
        try:
            r = json.load(open(f))
        except Exception as e:
            print(f"warn: skip unparseable {f}: {e}", file=sys.stderr)
            continue
        if not all(k in r for k in ("study", "weight", "np", "dflash", "vision")):
            continue
        out[(r["study"], r["weight"], int(r["np"]), bool(r["dflash"]), bool(r["vision"]))] = r
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


METRICS = [
    # (header, path..., prec, is_speed_higher_better)
    ("tok/s", ("metrics", "agg_tok_s"), 2, True),
    ("TTFT p50 ms", ("metrics", "ttft_p50"), 1, False),
    ("TTFT p90 ms", ("metrics", "ttft_p90"), 1, False),
    ("TPOT ms", ("metrics", "tpot_median"), 1, False),
    ("VmPeak GiB", ("mem", "VmPeak_gib"), 2, False),
]


def render(a: dict, b: dict, la: str, lb: str) -> str:
    keys = sorted(set(a) | set(b), key=lambda k: (k[0], k[1], k[2], int(k[3]), int(k[4])))
    lines = []
    lines.append(f"# ROCm {la} vs {lb} — cell-by-cell comparison\n")
    lines.append(f"- **{la}:** {len(a)} cells   **{lb}:** {len(b)} cells   **compared:** {len(set(a) & set(b))}\n")

    # speedup-summary stats over cells present in both arms with real metrics
    d_tok = []
    for k in keys:
        ra, rb = a.get(k), b.get(k)
        if not (ra and rb):
            continue
        if ra.get("pathological") or rb.get("pathological"):
            continue
        ta = num(ra, "metrics", "agg_tok_s"); tb = num(rb, "metrics", "agg_tok_s")
        if ta and tb:
            d_tok.append(pct(ta, tb))

    if d_tok:
        faster = sum(1 for d in d_tok if d > 0)
        slower = sum(1 for d in d_tok if d < 0)
        lines.append("## Summary (tok/s, both arms measured)\n")
        lines.append(f"- cells **faster** on {lb}: **{faster}/{len(d_tok)}**, **slower**: {slower}")
        lines.append(f"- mean tok/s Δ: **{fmtp(mean(d_tok))}**   "
                     f"range: {fmtp(min(d_tok))} … {fmtp(max(d_tok))}\n")

    by_study = {}
    for k in keys:
        by_study.setdefault(k[0], []).append(k)

    for st in sorted(by_study):
        lines.append(f"## {st}\n")
        hdr = ["cell"]
        for name, _p, _pr, _h in METRICS:
            hdr += [f"{la} {name}", f"{lb} {name}", "Δ"]
        hdr.append(f"{lb} accept")
        lines.append("| " + " | ".join(hdr) + " |")
        lines.append("|" + "|".join(["---"] * len(hdr)) + "|")
        for k in by_study[st]:
            ra, rb = a.get(k), b.get(k)
            row = [cell_label(rb or ra)]
            for _name, p, pr, higher in METRICS:
                va = num(ra, *p) if ra else None
                vb = num(rb, *p) if rb else None
                d = pct(va, vb)
                # for latency/footprint (lower better) invert the Δ sign coloring semantics in text only
                row += [fmt(va, pr), fmt(vb, pr), fmtp(d)]
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
    lines.append(f"- Δ = ({lb} − {la}) / {la} × 100. **Positive Δ on tok/s = {lb} is faster**; "
                 f"positive Δ on TTFT/TPOT/VmPeak = {lb} is slower/higher.")
    lines.append("- `—` = metric absent (baseline cells have no acceptance; pathological cells have no metrics).")
    lines.append(f"- Only ROCm differs across arms: same llama.cpp build `0b1bad1`, flags, weights, prompt set, seeds.")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", default="docs/results/matrix", help="matrix dir A (default 7.2.1)")
    ap.add_argument("--b", default="docs/results/matrix-714", help="matrix dir B (default 7.14.0)")
    ap.add_argument("--label-a", default="7.2.1")
    ap.add_argument("--label-b", default="7.14.0")
    ap.add_argument("-o", "--out", default=None, help="also write markdown to this file")
    args = ap.parse_args()

    a = load_matrix(args.a)
    b = load_matrix(args.b)
    if not a:
        sys.exit(f"no cells found in --a {args.a}")
    md = render(a, b, args.label_a, args.label_b)
    print(md)
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        open(args.out, "w").write(md)
        print(f"\n(wrote {args.out})", file=sys.stderr)


if __name__ == "__main__":
    main()
