#!/usr/bin/env python3
"""Derive exact llama-server flags from a cell config (pure).

Single source of truth for the matrix's flag logic. CI-tested; called by
gguf-bench-cell.sh. Usage:  python3 gguf_bench_args.py <cell.json> [server]
"""
import json
import sys

WEIGHTS = {
    "17gb": "models/muse-glimmer-30B-kquant-17gb.gguf",
    "dynamic": "models/muse-glimmer-30B-kquant-dynamic.gguf",
}
DEFAULT_DFLASH = "models/dflash-kquant.gguf"
DEFAULT_MMPROJ = "models/mmproj-kquant.gguf"

# Tuned request for `--spec-draft-n-max` on gfx1151 (Strix Halo):
# n_max=3 ->1.14x, n_max=8 ->1.51x, n_max=16 ->1.60x (16.7 tok/s vs 10.46
# baseline), flat past that elbow. Upstream DFlash drafts at most
# block_size - 1 = 15 tokens per round and silently clamps any higher request
# down to 15 with a warning line at every server start (the recorded sweep's
# 16 and 32 cells both ran at effective 15), so request 15 directly. Keep in
# sync with the acceptance-capture path in gguf-bench-cell.sh.
SPEC_DRAFT_N_MAX = 15

# Per-study client/sampling defaults (spec §5).
STUDY = {
    "study1": {"temp": 0, "top_p": 1.0, "top_k": 0, "max_tokens": 256,
               "reasoning_strength": "high", "endpoint": "chat"},
    "study2": {"temp": 1.0, "top_p": 0.95, "top_k": 64, "max_tokens": 512,
               "reasoning_strength": "high", "endpoint": "chat"},
    "study3": {"temp": 1.0, "top_p": 0.95, "top_k": 64, "max_tokens": 512,
               "reasoning_strength": "high", "endpoint": "chat"},
}


def _weight_path(cell):
    if cell["weight"] in WEIGHTS:
        return WEIGHTS[cell["weight"]]
    return cell["weight"]  # already a path


def build_server_args(cell):
    st = STUDY[cell["study"]]
    np_ = int(cell["np"])
    ctx = np_ * int(cell["per_slot_ctx"])
    args = [
        "llama-server",
        "-m", _weight_path(cell),
        "-ngl", "999",
        "-np", str(np_),
        "-c", str(ctx),
        "--jinja",
        "--temp", str(st["temp"]),
        "--seed", str(cell["seed"]),
        "--metrics",            # expose /metrics for draft-acceptance capture
        "--port", "8080",
        "--host", "127.0.0.1",
    ]
    if st["temp"] == 1.0:
        args += ["--top-p", str(st["top_p"]), "--top-k", str(st["top_k"])]
    if cell.get("dflash"):
        # `-md <dflash>` alone loads the draft model but does NOT enable
        # speculative decoding — `llama-server --spec-type` defaults to `none`,
        # which yields 0 drafts and a 1.00x speedup (verified on gfx1151).
        # Explicitly select the draft-dflash speculator and the tuned n_max.
        args += ["-md", cell.get("dflash_path", DEFAULT_DFLASH), "-ngld", "99",
                 "--spec-type", "draft-dflash",
                 "--spec-draft-n-max", str(SPEC_DRAFT_N_MAX)]
    if cell.get("vision"):
        args += ["--mmproj", cell.get("mmproj_path", DEFAULT_MMPROJ)]
    return args


if __name__ == "__main__":
    cell = json.loads(sys.argv[1])
    # Only the `server` path is used by the cell script (gguf-bench-cell.sh);
    # client params are sourced from the .conf files and passed to bench_client
    # via argparse, independently exercised by test_gguf_configs.py.
    print(json.dumps(build_server_args(cell)))
