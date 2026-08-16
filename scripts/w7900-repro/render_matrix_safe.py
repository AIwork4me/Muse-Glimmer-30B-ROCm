#!/usr/bin/env python3
"""Robust Study-2 detail renderer for the W7900 matrix (repo render_matrix.py
crashes on the non-pathological probe DFlash@c=16 cells, which have agg_tok_s
but null ttft/tpot). Handles full / probe / pathological cells; never raises.
Usage: render_matrix_safe.py <matrix_dir>
"""
import glob, json, os, sys


def _num(v, fmt, dash="—"):
    return fmt.format(v) if isinstance(v, (int, float)) else dash


def render(matrix_dir):
    cells = []
    for p in sorted(glob.glob(os.path.join(matrix_dir, "cell-study2-*.json"))):
        try:
            cells.append(json.load(open(p)))
        except (OSError, json.JSONDecodeError):
            pass
    out = ["### Study 2 — Throughput under load (W7900 / gfx1100)\n",
           "| weight | np | mode | agg tok/s | TTFT p90 (s) | TPOT med (s) | accept | VRAM (GiB) | note |",
           "|---|---:|---:|---:|---:|---:|---:|---:|---|"]
    for c in sorted(cells, key=lambda x: (x.get("weight"), int(x.get("np", 0)), x.get("dflash"))):
        w, np_ = c.get("weight"), c.get("np")
        mode = "DFlash" if c.get("dflash") else "baseline"
        m = c.get("metrics") or {}
        acc = c.get("acceptance") or {}
        vr = (c.get("mem") or {}).get("vram_used_mib")
        vr = vr / 1024 if isinstance(vr, (int, float)) else None
        if c.get("pathological"):
            note, agg, ttft, tpot = "⚠ PATHOLOGICAL (did not complete)", "—", "—", "—"
        else:
            agg = _num(m.get("agg_tok_s"), "{:.2f}")
            ttft = _num(m.get("ttft_p90"), "{:.3f}")
            tpot = _num(m.get("tpot_median"), "{:.4f}")
            note = f"probe ({np_}×48, single prompt)" if m.get("reps") == "probe_only" else f"full cell (reps {m.get('reps')})"
        out.append(f"| {w} | {np_} | {mode} | {agg} | {ttft} | {tpot} | {_num(acc.get('acceptance_rate'), '{:.1%}')} | {_num(vr, '{:.1f}')} | {note} |")
    return "\n".join(out) + "\n"


if __name__ == "__main__":
    print(render(sys.argv[1] if len(sys.argv) > 1 else "results/matrix-w7900-gfx1100"))
