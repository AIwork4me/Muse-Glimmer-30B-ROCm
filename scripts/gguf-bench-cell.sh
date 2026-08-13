#!/usr/bin/env bash
# Run ONE matrix cell: launch llama-server with exact flags, warmup+measure via
# bench_client, capture RSS/power/acceptance, tear down. Emits one JSON record.
# Usage: gguf-bench-cell.sh <study> <weight:17gb|dynamic> <dflash:0|1> <vision:0|1> <np> <conf-file>
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"; cd "$HERE"
export PATH="$HOME/.local/bin:$PATH"
# Server binary + output dir are env-overridable so the ROCm 7.14.0 side-by-side
# run (scripts/run-gguf-matrix-714.sh) can point at its own build-714/ server and
# its own matrix-714/ output dir without touching this script's 7.2.1 defaults.
LLAMA="${LLAMA_BIN:-$HERE/third_party/llama.cpp/build/bin/llama-server}"

STUDY=$1; WEIGHT=$2; DFLASH=$3; VISION=$4; NP=$5; CONF=$6
# shellcheck source=/dev/null
. "$CONF"
PER_SLOT_CTX=${PER_SLOT_CTX:-8192}; SEED=${SEED:-0}; REPS=${REPS:-3}; WARMUP=${WARMUP:-2}
TEMP=${TEMP:-0}; TOP_P=${TOP_P:-1.0}; TOP_K=${TOP_K:-0}; MAX_TOKENS=${MAX_TOKENS:-256}
RS=${REASONING_STRENGTH:-high}

CELL=$(printf '{"weight":"%s","dflash":%s,"vision":%s,"np":%s,"per_slot_ctx":%s,"study":"%s","seed":%s}' \
  "$WEIGHT" "$DFLASH" "$VISION" "$NP" "$PER_SLOT_CTX" "$STUDY" "$SEED")
# gguf_bench_args.py returns a JSON argv list whose element 0 is the binary
# name. Convert the remaining elements to a NUL-delimited Bash array so paths
# with spaces and wildcard characters remain single literal arguments.
mapfile -d '' -t SRV_ARGS < <(
  python3 scripts/gguf_bench_args.py "$CELL" server |
    python3 -c 'import json, os, sys; [sys.stdout.buffer.write(os.fsencode(a) + b"\0") for a in json.load(sys.stdin)[1:]]'
)
printf -v SRV_FLAGS '%q ' "${SRV_ARGS[@]}"
SRV_FLAGS="${SRV_FLAGS% }"

LOG=$(mktemp)
"$LLAMA" "${SRV_ARGS[@]}" >"$LOG" 2>&1 &
SRV_PID=$!
# Kill + wait on exit so the server fully releases port 8080 and GPU memory
# before the next cell starts (the matrix driver runs cells back-to-back; a
# bare `kill` that doesn't reap can leave the port held and the next launch
# fails to bind). `wait` returns non-zero if the child was already gone, hence || true.
trap 'kill "$SRV_PID" 2>/dev/null; wait "$SRV_PID" 2>/dev/null || true' EXIT

# wait for health
for _ in $(seq 1 120); do curl -sf http://127.0.0.1:8080/health >/dev/null 2>&1 && break; sleep 1; done
curl -sf http://127.0.0.1:8080/health >/dev/null || { echo "server failed to start"; cat "$LOG"; exit 1; }

IMG_ARG=(); [ "$VISION" = "1" ] && IMG_ARG=(--image scripts/prompt-sets/test-image.png)
METRICS=$(uv run --no-sync python scripts/bench_client.py http://127.0.0.1:8080 \
  --study "$STUDY" --endpoint chat --np "$NP" --temp "$TEMP" --top-p "$TOP_P" --top-k "$TOP_K" \
  --max-tokens "$MAX_TOKENS" --reps "$REPS" --warmup "$WARMUP" --seed "$SEED" \
  --reasoning-strength "$RS" "${IMG_ARG[@]}")

MEM=$(python3 scripts/capture_proc.py status "$SRV_PID" 2>/dev/null || echo '{"error":"proc unreadable"}')
VRAM=$(rocm-smi --showmeminfo vram --json 2>/dev/null | python3 scripts/capture_proc.py vram 2>/dev/null || echo '{}')
POWER=$(rocm-smi --showpower --showtemp --json 2>/dev/null | python3 scripts/capture_proc.py power 2>/dev/null || echo '{}')

# Acceptance capture. PRIMARY source is the server LOG's per-slot print_timing
# line (parse_draft_acceptance): on build 0b1bad1 the /metrics spec counters
# stay 0 even when spec-decoding is ACTIVE (build-instrumentation gap, verified
# 2026-08-12 on gfx1151), so the log is authoritative. /metrics is retained as
# a SECONDARY fallback field (populated below, emitted only if non-empty).
# NB: parse $LOG before the `rm -f "$LOG"` at the end of the script.
ACC_LOG=$(python3 scripts/capture_proc.py draft < "$LOG" 2>/dev/null || echo 'null')
ACC_METRICS=$(curl -s http://127.0.0.1:8080/metrics 2>/dev/null | python3 scripts/capture_proc.py metrics 2>/dev/null || echo 'null')

# TOP-RISK probe (resolved 2026-08-12): /metrics DOES expose spec/draft
# counters, but they stay 0 even when spec-decoding is active (build
# instrumentation gap). Acceptance is therefore parsed from the SERVER LOG
# above (ACC_LOG). This probe is retained as a live diagnostic of which
# counters the running build emits.
echo "=== /metrics spec/draft counters (diagnostic; known to read 0 on 0b1bad1) ==="
curl -s http://127.0.0.1:8080/metrics 2>/dev/null | grep -iE 'spec|draft' | head || echo "  (none found)"
echo "=== end probe ==="

# Best-effort: capture any model-size line llama-server prints. Build 0b1bad1
# does NOT emit "model size: X MiB" (older builds did), so this is often empty;
# manifest.flags + mem.VmPeak already record the footprint. Keep it optional.
WGT=$(grep -oE 'model size[^0-9]*[0-9.]+ (MiB|GiB|MB|GB)' "$LOG" | tail -1 || true)

MATRIX_DIR="${MATRIX_OUTDIR:-docs/results/matrix}"
OUT="$MATRIX_DIR/cell-${STUDY}-${WEIGHT}-np${NP}-df${DFLASH}-vis${VISION}.json"
mkdir -p "$MATRIX_DIR"
python3 - "$OUT" "$STUDY" "$WEIGHT" "$DFLASH" "$VISION" "$NP" "$MEM" "$VRAM" "$METRICS" "$ACC_LOG" "$ACC_METRICS" "$POWER" "$WGT" "$SRV_FLAGS" "$SEED" "$REPS" <<'PY'
import json, sys, subprocess, datetime, os, re
(out, study, weight, df, vis, np_, mem, vram, metrics, acc_log, acc_metrics, power, wgt, flags, seed, reps) = sys.argv[1:]
rocm = subprocess.run("rocm-smi --showproductname --json".split(), capture_output=True, text=True).stdout.strip()
# Explicit ROCm version for provenance. The 7.2.1 cells have no such field
# (their `rocm` above is the GPU *product* JSON, not the ROCm version); this
# labels the 7.14.0 cells unambiguously without modifying the immutable 7.2.1 cells.
def _rocm_ver():
    import shutil
    # Prefer the active prefix's .info/version — the clean release label (e.g.
    # "7.14.0") — over the granular tool-reported HIP build number.
    try:
        hipcc = shutil.which("hipcc")
        if hipcc:
            base = os.path.dirname(os.path.dirname(os.path.realpath(hipcc)))
            v = open(os.path.join(base, ".info", "version")).read().strip()
            if v:
                return v
    except Exception:
        pass
    for cmd in (("rocm-smi", "--version"), ("hipcc", "--version"), ("hipconfig", "--version")):
        try:
            out = subprocess.run(cmd, capture_output=True, text=True, timeout=10).stdout
        except Exception:
            continue
        m = re.search(r"(?:ROCm|HIP)\s*version[:\s]*([0-9]+\.[0-9]+\.[0-9.]+)", out, re.I)
        if m:
            return m.group(1).strip().rstrip(".")
    return "unknown"
# mem merges /proc/<pid>/status (VmPeak/VmHWM/VmRSS/RssShmem — host-side) with
# GPU VRAM. On Strix Halo the GGUF is mmap'd + GPU-offloaded, so VmHWM (~1 GiB)
# is only the host working set; VmPeak records mapped address space and VRAM
# tracks the carve-out. Keep all three: none alone is resident physical memory.
mem_rec = json.loads(mem)
mem_rec.update(json.loads(vram) if vram else {})

# Acceptance source selection: the server log's print_timing line is PRIMARY
# (authoritative); /metrics is a secondary fallback for future builds that
# populate it. A record counts as "present" when it has a non-null
# accepted_draft_tokens count (baseline cells legitimately have none).
def _has_acc(d):
    return bool(d) and d.get("accepted_draft_tokens") is not None

acc_log_d = json.loads(acc_log) if acc_log != "null" else None
acc_metrics_d = json.loads(acc_metrics) if acc_metrics != "null" else None
if _has_acc(acc_log_d):
    acceptance, acceptance_source = acc_log_d, "log"
elif _has_acc(acc_metrics_d):
    acceptance, acceptance_source = acc_metrics_d, "metrics"
else:
    acceptance, acceptance_source = (acc_log_d or acc_metrics_d), None

rec = {"study": study, "weight": weight, "dflash": df=="1", "vision": vis=="1", "np": int(np_),
       "metrics": json.loads(metrics), "mem": mem_rec, "acceptance": acceptance,
       "acceptance_source": acceptance_source,
       "power_temp": json.loads(power) if power else {}, "llama_log_mem": wgt,
       "manifest": {"flags": flags, "seed": int(seed), "reps": int(reps), "build": "0b1bad1",
                    "rocm_version": _rocm_ver(), "rocm": rocm, "kernel": os.popen("uname -r").read().strip(),
                    "date": datetime.date.today().isoformat()}}
# Secondary /metrics-derived acceptance: emit only when it carries non-null
# data AND differs from the chosen primary (avoids a redundant duplicate).
if _has_acc(acc_metrics_d) and acc_metrics_d != acceptance:
    rec["acceptance_metrics_secondary"] = acc_metrics_d
json.dump(rec, open(out, "w"), indent=2)
print("wrote", out)
PY
rm -f "$LOG"
