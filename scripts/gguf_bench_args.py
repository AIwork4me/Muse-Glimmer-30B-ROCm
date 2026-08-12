#!/usr/bin/env python3
"""Derive exact llama-server flags + client params from a cell config (pure).

Single source of truth for the matrix's flag logic. CI-tested; called by
gguf-bench-cell.sh. Usage:  python3 gguf_bench_args.py <cell.json> server|client
"""
import json
import sys

WEIGHTS = {
    "17gb": "models/muse-glimmer-30B-kquant-17gb.gguf",
    "dynamic": "models/muse-glimmer-30B-kquant-dynamic.gguf",
}
DEFAULT_DFLASH = "models/dflash-kquant.gguf"
DEFAULT_MMPROJ = "models/mmproj-kquant.gguf"

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
        args += ["-md", cell.get("dflash_path", DEFAULT_DFLASH), "-ngld", "99"]
    if cell.get("vision"):
        args += ["--mmproj", cell.get("mmproj_path", DEFAULT_MMPROJ)]
    return args


def build_client_args(cell):
    st = STUDY[cell["study"]]
    return {
        "endpoint": st["endpoint"],
        "temp": st["temp"],
        "top_p": st["top_p"],
        "top_k": st["top_k"],
        "max_tokens": st["max_tokens"],
        "reasoning_strength": st["reasoning_strength"],
        "seed": cell["seed"],
        "np": cell["np"],
        "prompt_set": "muse-glimmer-diverse",
    }


if __name__ == "__main__":
    cell = json.loads(sys.argv[1])
    which = sys.argv[2] if len(sys.argv) > 2 else "server"
    out = build_server_args(cell) if which == "server" else build_client_args(cell)
    print(json.dumps(out))
