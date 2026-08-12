#!/usr/bin/env python3
"""Render matrix cell-result JSONs into a markdown report (pure)."""
import glob
import json
import os


def _row(c, base=None):
    # Guard: a pathological study1/study3 cell (no metrics) would TypeError on
    # the f-string below. None exist by design today, but mirror _study2's
    # handling so a future one renders a warning instead of crashing.
    if c.get("pathological"):
        return (f'| {c["weight"]} | {"DFlash" if c["dflash"] else "baseline"} | '
                f'⚠ **PATHOLOGICAL — did not complete** | — | — | — | — | — |')
    m = c["metrics"]
    mem = c["mem"]
    toks = f'{m["agg_tok_s"]:.1f}'
    speedup = ""
    if base:
        speedup = f'{m["agg_tok_s"] / base:.2f}x'
    acc = c.get("acceptance") or {}
    acc_s = f'{int(round(acc["acceptance_rate"] * 100))}%' if acc.get("acceptance_rate") is not None else "—"
    return (f'| {c["weight"]} | {"DFlash" if c["dflash"] else "baseline"} | '
            f'{toks} | {m["ttft_p50"]:.2f} | {m["tpot_median"]:.4f} | '
            f'{mem["VmPeak_gib"]:.1f} | {speedup or "—"} | {acc_s} |')


def _study1(cells):
    out = ["### Study 1 — DFlash anchor (greedy, batch 1, diverse prompt set) — Meta-comparable\n",
           "| weight | mode | tok/s | TTFT p50 (s) | TPOT (s) | footprint VmPeak (GiB) | Speedup | draft acceptance |",
           "|---|---|---|---|---|---|---|---|"]
    for w in ("17gb", "dynamic"):
        base = next((c["metrics"]["agg_tok_s"] for c in cells
                     if c["weight"] == w and not c["dflash"] and c.get("study") == "study1"), None)
        for c in [x for x in cells if x["weight"] == w and x.get("study") == "study1"]:
            out.append(_row(c, base))
    return "\n".join(out)


def _study2(cells):
    out = ["### Study 2 — Throughput under load (temp 1.0) — NOT Meta-comparable\n",
           "| weight | np | mode | agg tok/s | TTFT p90 (s) | TPOT med (s) | footprint VmPeak (GiB) | acceptance |",
           "|---|---|---|---|---|---|---|---|"]
    for c in sorted([x for x in cells if x.get("study") == "study2"],
                    key=lambda x: (x["weight"], x["np"], x["dflash"])):
        if c.get("pathological"):
            out.append(f'| {c["weight"]} | {c["np"]} | DFlash | '
                       f'⚠ **PATHOLOGICAL — did not complete** (see c=16 warning) | — | — | — | — |')
            continue
        m = c["metrics"]
        acc = c.get("acceptance") or {}
        acc_s = f'{int(round(acc["acceptance_rate"] * 100))}%' if acc.get("acceptance_rate") is not None else "—"
        out.append(f'| {c["weight"]} | {c["np"]} | {"DFlash" if c["dflash"] else "baseline"} | '
                   f'{m["agg_tok_s"]:.1f} | {m["ttft_p90"]:.2f} | {m["tpot_median"]:.4f} | '
                   f'{c["mem"]["VmPeak_gib"]:.1f} | {acc_s} |')
    return "\n".join(out)


def _study3(cells):
    # Vision axis: memory footprint is the headline signal (mmproj adds ~+2 GB
    # per Meta). Report both GPU VRAM (the mmap'd+offloaded weight carveout, the
    # meaningful signal on Strix Halo) and host VmPeak (mmap'd model size) so the
    # delta vs the text-only baseline can be computed by the analysis layer.
    out = ["### Study 3 — Vision axis (temp 1.0, mmproj + test image) — memory delta vs text-only\n",
           "| weight | np | mode | agg tok/s | TTFT p90 (s) | TPOT med (s) | VRAM (MiB) | VmPeak (GiB) | acceptance |",
           "|---|---|---|---|---|---|---|---|---|"]
    for c in sorted([x for x in cells if x.get("study") == "study3"],
                    key=lambda x: (x["weight"], x["np"], x["dflash"])):
        m = c["metrics"]
        mem = c["mem"]
        acc = c.get("acceptance") or {}
        acc_s = f'{int(round(acc["acceptance_rate"] * 100))}%' if acc.get("acceptance_rate") is not None else "—"
        vram = f'{mem["vram_used_mib"]:.0f}' if mem.get("vram_used_mib") is not None else "—"
        vmpeak = f'{mem["VmPeak_gib"]:.1f}' if mem.get("VmPeak_gib") is not None else "—"
        out.append(f'| {c["weight"]} | {c["np"]} | {"DFlash" if c["dflash"] else "baseline"} | '
                   f'{m["agg_tok_s"]:.1f} | {m["ttft_p90"]:.2f} | {m["tpot_median"]:.4f} | '
                   f'{vram} | {vmpeak} | {acc_s} |')
    return "\n".join(out)


def render_studies(cells):
    if not cells:
        return "# Matrix report\n\n(no cells)\n"
    return "# llama.cpp benchmark matrix\n\n" + "\n\n".join(
        f for f in (_study1(cells), _study2(cells), _study3(cells)) if f.count("\n") > 2) + "\n"


if __name__ == "__main__":
    cells = [json.load(open(p)) for p in
             sorted(glob.glob(os.path.join(os.path.dirname(__file__), "..", "docs", "results", "matrix", "cell-*.json")))]
    print(render_studies(cells))
