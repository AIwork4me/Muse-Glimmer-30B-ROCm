#!/usr/bin/env python3
"""Parse /proc/<pid>/status, rocm-smi JSON, and llama-server /metrics (pure parsers)
+ a small CLI to sample a live PID. Used by gguf-bench-cell.sh.
"""
import json
import re
import sys


def parse_proc_status(text):
    out = {}
    for key in ("VmPeak", "VmHWM", "VmRSS", "RssShmem"):
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


def parse_rocm_smi_vram(json_text):
    """GPU VRAM usage from `rocm-smi --showmeminfo vram --json`.

    Returns vram_used_mib/vram_total_mib. On APUs (gfx1151/Strix Halo) the GGUF
    is mmap'd and GPU-offloaded weights do NOT appear in host VmHWM, so this is
    the meaningful memory signal for the matrix. Note: on unified-memory APUs
    the driver's VRAM counter may underreport shared/system-resident weight
    pages; report both this and VmHWM and let the analysis layer interpret.
    """
    d = json.loads(json_text)
    card = next(iter(d.values()))  # card0
    used = card.get("VRAM Total Used Memory (B)") or card.get("VRAM Used Memory (B)")
    total = card.get("VRAM Total Memory (B)")
    out = {}
    if used is not None:
        out["vram_used_mib"] = round(int(used) / 1024 / 1024, 1)
    if total is not None:
        out["vram_total_mib"] = round(int(total) / 1024 / 1024, 1)
    return out


def scrape_metrics(text):
    """Pull speculative-decoding counters from llama-server /metrics (Prometheus).

    llama-server (build 0b1bad1+) exposes these as `llamacpp:spec_decode_*`
    (verified on 2026-08-12 against a live gfx1151 server — exact names:
    `llamacpp:spec_decode_num_{draft,accepted,drafts}_total`). Older builds used
    `llama_speculative_*`; we try the real names first and fall back to the
    legacy aliases so the parser stays robust across builds.

    There is NO published `avg_accepted` gauge: the per-STEP average must be
    DERIVED as `num_accepted_tokens_total / num_drafts_total` (drafts = the
    count of speculative verification steps, the correct per-step denominator —
    not the draft-TOKEN count). Returns null when num_drafts_total is 0/absent.

    NOTE (2026-08-12 DFlash fix): on the current build the /metrics spec
    counters stay 0 even when spec-decoding is ACTIVE — a build-instrumentation
    gap. The AUTHORITATIVE acceptance source is therefore the server log's
    per-slot print_timing line, parsed by `parse_draft_acceptance`. This
    function is retained as a secondary source for future builds that populate
    /metrics correctly.
    """
    def grab(*names):
        for name in names:
            m = re.search(rf"^{re.escape(name)}(?:\{{[^}}]*\}})?\s+([0-9.eE+-]+)", text, re.M)
            if m:
                return float(m.group(1))
        return None

    accepted = grab("llamacpp:spec_decode_num_accepted_tokens_total",
                    "llama_speculative_accepted_draft_tokens_total")
    drafted = grab("llamacpp:spec_decode_num_draft_tokens_total",
                   "llama_speculative_draft_tokens_total")
    drafts = grab("llamacpp:spec_decode_num_drafts_total",
                  "llama_speculative_drafts_total")
    avg = (accepted / drafts) if (accepted is not None and drafts) else None
    out = {
        "accepted_draft_tokens": accepted,
        "draft_tokens": drafted,
        "avg_accepted_per_step": avg,
    }
    if accepted is not None and drafted:
        out["acceptance_rate"] = accepted / drafted
    return out


# Matches the per-slot print_timing line emitted by llama-server, e.g.
#   slot print_timing: id  0 | task 0 | draft acceptance = 0.14996
#       (  175 accepted /  1167 generated), mean len =   3.19
# Tolerates variable whitespace and labels. `mean len` is the average number of
# accepted tokens per verification STEP for that slot.
_DRAFT_ACCEPT_RE = re.compile(
    r"draft acceptance\s*=\s*([0-9.]+)\s*\(\s*(\d+)\s*accepted\s*/\s*(\d+)\s*generated\s*\)"
    r",\s*mean len\s*=\s*([0-9.]+)"
)


def parse_draft_acceptance(log_text):
    """Sum speculative-draft acceptance across ALL print_timing lines in the
    llama-server log. Returns:
      {"accepted_draft_tokens": <sum accepted int>,
       "draft_tokens":          <sum generated int>,
       "acceptance_rate":       <sum accepted / sum generated or None>,
       "avg_accepted_per_step": <mean of per-line `mean len` or None>}
    On empty input or no match -> all values None, never raises. This is the
    AUTHORITATIVE acceptance source for DFlash cells (the /metrics spec counters
    are unpopulated on the current build).
    """
    matches = _DRAFT_ACCEPT_RE.findall(log_text or "")
    if not matches:
        return {"accepted_draft_tokens": None, "draft_tokens": None,
                "acceptance_rate": None, "avg_accepted_per_step": None}
    sum_accepted = sum(int(m[1]) for m in matches)
    sum_generated = sum(int(m[2]) for m in matches)
    mean_lens = [float(m[3]) for m in matches]
    rate = (sum_accepted / sum_generated) if sum_generated else None
    avg_per_step = sum(mean_lens) / len(mean_lens) if mean_lens else None
    return {
        "accepted_draft_tokens": sum_accepted,
        "draft_tokens": sum_generated,
        "acceptance_rate": rate,
        "avg_accepted_per_step": avg_per_step,
    }


if __name__ == "__main__":
    # CLI modes:
    #   capture_proc.py status <pid>   -> /proc/<pid>/status memory fields
    #   capture_proc.py power          -> rocm-smi power/temp (stdin, or calls rocm-smi)
    #   capture_proc.py vram           -> rocm-smi VRAM (stdin, or calls rocm-smi)
    #   capture_proc.py metrics        -> llama-server /metrics acceptance (stdin)
    #   capture_proc.py draft          -> llama-server LOG acceptance (stdin, PRIMARY)
    import subprocess
    mode = sys.argv[1] if len(sys.argv) > 1 else "status"
    if mode == "status":
        if len(sys.argv) < 3:
            print(json.dumps({"error": "missing pid argument"}))
        else:
            try:
                print(json.dumps(parse_proc_status(open(f"/proc/{sys.argv[2]}/status").read())))
            except OSError as e:
                print(json.dumps({"error": str(e)}))
    elif mode == "power":
        txt = sys.stdin.read() or subprocess.run(
            ["rocm-smi", "--showpower", "--showtemp", "--json"],
            capture_output=True, text=True).stdout
        print(json.dumps(parse_rocm_smi_power(txt)))
    elif mode == "vram":
        txt = sys.stdin.read() or subprocess.run(
            ["rocm-smi", "--showmeminfo", "vram", "--json"],
            capture_output=True, text=True).stdout
        print(json.dumps(parse_rocm_smi_vram(txt)))
    elif mode == "metrics":
        print(json.dumps(scrape_metrics(sys.stdin.read())))
    elif mode == "draft":
        print(json.dumps(parse_draft_acceptance(sys.stdin.read())))
