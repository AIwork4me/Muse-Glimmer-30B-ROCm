#!/usr/bin/env python3
"""Parse /proc/<pid>/status, rocm-smi JSON, and llama-server /metrics (pure parsers)
+ a small CLI to sample a live PID. Used by gguf-bench-cell.sh.
"""
import json
import re
import sys


def parse_proc_status(text):
    out = {}
    for key in ("VmHWM", "VmRSS", "RssShmem"):
        m = re.search(rf"^{key}:\s+(\d+)\s+kB", text, re.M)
        if m:
            kib = int(m.group(1))
            out[f"{key}_kib"] = kib
            out[f"{key}_gib"] = round(kib / 1024 / 1024, 2)
    return out


def parse_rocm_smi_power(json_text):
    d = json.loads(json_text)
    card = next(iter(d.values()))  # card0
    powr = card.get("Average Graphics Package Power (W)")
    temp = card.get("Temperature (Sensor edge) (C)")
    return {
        "power_w": float(powr) if powr else None,
        "temp_c": float(temp) if temp else None,
    }


def scrape_metrics(text):
    """Pull speculative-decoding counters from llama-server /metrics (Prometheus)."""
    vals = {}

    def grab(name):
        m = re.search(rf"^{name}\s+([0-9.eE+-]+)", text, re.M)
        return float(m.group(1)) if m else None

    accepted = grab("llama_speculative_accepted_draft_tokens_total")
    drafted = grab("llama_speculative_draft_tokens_total")
    avg = grab("llama_speculative_avg_accepted")
    out = {
        "accepted_draft_tokens": accepted,
        "draft_tokens": drafted,
        "avg_accepted_per_step": avg,
    }
    if accepted is not None and drafted:
        out["acceptance_rate"] = accepted / drafted
    return out


if __name__ == "__main__":
    # CLI modes:
    #   capture_proc.py status <pid>   -> /proc/<pid>/status memory fields
    #   capture_proc.py power          -> rocm-smi power/temp (stdin, or calls rocm-smi)
    #   capture_proc.py metrics        -> llama-server /metrics acceptance (stdin)
    import subprocess
    mode = sys.argv[1] if len(sys.argv) > 1 else "status"
    if mode == "status":
        try:
            print(json.dumps(parse_proc_status(open(f"/proc/{sys.argv[2]}/status").read())))
        except OSError as e:
            print(json.dumps({"error": str(e)}))
    elif mode == "power":
        txt = sys.stdin.read() or subprocess.run(
            ["rocm-smi", "--showpower", "--showtemp", "--json"],
            capture_output=True, text=True).stdout
        print(json.dumps(parse_rocm_smi_power(txt)))
    elif mode == "metrics":
        print(json.dumps(scrape_metrics(sys.stdin.read())))
