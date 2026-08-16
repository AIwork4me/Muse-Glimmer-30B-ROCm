#!/usr/bin/env python3
"""Render the Study-2 headline table from cell-study2-*.json.

Columns: c=1 & c=4 show baseline / DFlash; c=16 & c=32 show baseline only
(DFlash at high concurrency is a probe / not helpful -> reported in a footnote).
Stdlib only.  Usage: render_headline.py <matrix_dir>
"""
import glob
import json
import os
import sys

WEIGHT_LABEL = {"17gb": "17GB", "dynamic": "dynamic"}
PAIR_C = (1, 4)          # columns showing baseline / DFlash
BASE_C = (16, 32)        # columns showing baseline only


def load(matrix_dir):
    cells = []
    for p in sorted(glob.glob(os.path.join(matrix_dir, "cell-study2-*.json"))):
        try:
            cells.append(json.load(open(p)))
        except (OSError, json.JSONDecodeError):
            pass
    return cells


def _find(cells, weight, np_, dflash):
    for c in cells:
        if c.get("weight") == weight and int(c.get("np", -1)) == np_ and bool(c.get("dflash")) == dflash:
            return c
    return None


def _tok(cell):
    if cell is None:
        return "—"
    if cell.get("pathological"):
        return "⚠ pathological"
    v = (cell.get("metrics") or {}).get("agg_tok_s")
    return f"{v:.2f}" if isinstance(v, (int, float)) else "—"


def _pair(cells, weight, np_):
    return f"{_tok(_find(cells, weight, np_, False))} / {_tok(_find(cells, weight, np_, True))}"


def render(cells):
    header = "| Weight | " + " | ".join(f"c={c} baseline / DFlash" for c in PAIR_C) \
             + " | " + " | ".join(f"c={c} baseline" for c in BASE_C) + " |"
    sep = "|---|" + "---:|" * (len(PAIR_C) + len(BASE_C))
    lines = [header, sep]
    for w in ("17gb", "dynamic"):
        cols = [f"{_pair(cells, w, c)} tok/s" for c in PAIR_C]
        cols += [f"{_tok(_find(cells, w, c, False))} tok/s" for c in BASE_C]
        lines.append(f"| {WEIGHT_LABEL[w]} | " + " | ".join(cols) + " |")
    notes = []
    for c in BASE_C:
        for w in ("17gb", "dynamic"):
            cell = _find(cells, w, c, True)
            if cell is None:
                continue
            if cell.get("pathological"):
                notes.append(f"- DFlash @ c={c} ({WEIGHT_LABEL[w]}): **pathological — did not complete** (guarded).")
            else:
                notes.append(f"- DFlash @ c={c} ({WEIGHT_LABEL[w]}): {_tok(cell)} tok/s (探针；完成但慢于 baseline，故非表头列).")
    out = "\n".join(lines)
    if notes:
        out += "\n\n" + "\n".join(notes)
    return out + "\n"


if __name__ == "__main__":
    mdir = sys.argv[1] if len(sys.argv) > 1 else "results/matrix-w7900-gfx1100"
    cells = load(mdir)
    if not cells:
        sys.stderr.write(f"no cell-study2-*.json found in {mdir}\n")
        sys.exit(1)
    print(render(cells))
