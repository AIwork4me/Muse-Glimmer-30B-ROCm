#!/usr/bin/env bash
# Run ONE matrix cell: launch llama-server with exact flags, warmup+measure via
# bench_client, capture RSS/power/acceptance, tear down. Emits one JSON record.
# Usage: gguf-bench-cell.sh <study> <weight:17gb|dynamic> <dflash:0|1> <vision:0|1> <np> <conf-file>
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"; cd "$HERE"
export PATH="$HOME/.local/bin:$PATH"
LLAMA="$HERE/third_party/llama.cpp/build/bin/llama-server"

STUDY=$1; WEIGHT=$2; DFLASH=$3; VISION=$4; NP=$5; CONF=$6
# shellcheck source=/dev/null
. "$CONF"
PER_SLOT_CTX=${PER_SLOT_CTX:-8192}; SEED=${SEED:-0}; REPS=${REPS:-3}; WARMUP=${WARMUP:-2}
TEMP=${TEMP:-0}; TOP_P=${TOP_P:-1.0}; TOP_K=${TOP_K:-0}; MAX_TOKENS=${MAX_TOKENS:-256}
RS=${REASONING_STRENGTH:-high}

CELL=$(printf '{"weight":"%s","dflash":%s,"vision":%s,"np":%s,"per_slot_ctx":%s,"study":"%s","seed":%s}' \
  "$WEIGHT" "$DFLASH" "$VISION" "$NP" "$PER_SLOT_CTX" "$STUDY" "$SEED")
# gguf_bench_args.py returns a JSON list whose element 0 is the argv[0] binary
# name ("llama-server"); we drop it and join the rest with spaces. (The brief's
# `tr -d '[]"' | sed 's/,/ /g' left argv[0] as a spurious positional arg, which
# llama-server rejects with "error: invalid argument: llama-server". Doing the
# conversion in python is robust to that and to any future spaces in paths.)
SRV_ARGS=$(python3 scripts/gguf_bench_args.py "$CELL" server \
  | python3 -c "import json,sys; print(' '.join(json.load(sys.stdin)[1:]))")

LOG=$(mktemp)
"$LLAMA" $SRV_ARGS >"$LOG" 2>&1 &
SRV_PID=$!
trap 'kill $SRV_PID 2>/dev/null || true' EXIT

# wait for health
for _ in $(seq 1 120); do curl -sf http://127.0.0.1:8080/health >/dev/null 2>&1 && break; sleep 1; done
curl -sf http://127.0.0.1:8080/health >/dev/null || { echo "server failed to start"; cat "$LOG"; exit 1; }

IMG_ARG=(); [ "$VISION" = "1" ] && IMG_ARG=(--image scripts/prompt-sets/test-image.png)
METRICS=$(uv run --no-sync python scripts/bench_client.py http://127.0.0.1:8080 \
  --study "$STUDY" --endpoint chat --np "$NP" --temp "$TEMP" --top-p "$TOP_P" --top-k "$TOP_K" \
  --max-tokens "$MAX_TOKENS" --reps "$REPS" --warmup "$WARMUP" --seed "$SEED" \
  --reasoning-strength "$RS" "${IMG_ARG[@]}")

MEM=$(python3 scripts/capture_proc.py status "$SRV_PID")
VRAM=$(rocm-smi --showmeminfo vram --json 2>/dev/null | python3 scripts/capture_proc.py vram 2>/dev/null || echo '{}')
POWER=$(rocm-smi --showpower --showtemp --json 2>/dev/null | python3 scripts/capture_proc.py power 2>/dev/null || echo '{}')
ACC=$(curl -s http://127.0.0.1:8080/metrics 2>/dev/null | python3 scripts/capture_proc.py metrics 2>/dev/null || echo 'null')

# TOP-RISK probe (spec): does /metrics expose the speculative/draft counters
# needed for DFlash cells in Task 11? Report whatever is present.
echo "=== /metrics spec/draft counters (TOP-RISK probe) ==="
curl -s http://127.0.0.1:8080/metrics 2>/dev/null | grep -iE 'spec|draft' | head || echo "  (none found)"
echo "=== end probe ==="

# Best-effort: capture any model-size line llama-server prints. Build 0b1bad1
# does NOT emit "model size: X MiB" (older builds did), so this is often empty;
# manifest.flags + mem.VmPeak already record the footprint. Keep it optional.
WGT=$(grep -oE 'model size[^0-9]*[0-9.]+ (MiB|GiB|MB|GB)' "$LOG" | tail -1 || true)

OUT="docs/results/matrix/cell-${STUDY}-${WEIGHT}-np${NP}-df${DFLASH}-vis${VISION}.json"
mkdir -p docs/results/matrix
python3 - "$OUT" "$STUDY" "$WEIGHT" "$DFLASH" "$VISION" "$NP" "$MEM" "$VRAM" "$METRICS" "$ACC" "$POWER" "$WGT" "$SRV_ARGS" "$SEED" "$REPS" <<'PY'
import json, sys, subprocess, datetime, os
(out, study, weight, df, vis, np_, mem, vram, metrics, acc, power, wgt, flags, seed, reps) = sys.argv[1:]
rocm = subprocess.run("rocm-smi --showproductname --json".split(), capture_output=True, text=True).stdout.strip()
# mem merges /proc/<pid>/status (VmPeak/VmHWM/VmRSS/RssShmem — host-side) with
# GPU VRAM. On Strix Halo the GGUF is mmap'd + GPU-offloaded, so VmHWM (~1 GiB)
# is only the host working set; VmPeak reflects the mmap'd model and VRAM tracks
# the GPU carveout. All three together describe the real footprint.
mem_rec = json.loads(mem)
mem_rec.update(json.loads(vram) if vram else {})
rec = {"study": study, "weight": weight, "dflash": df=="1", "vision": vis=="1", "np": int(np_),
       "metrics": json.loads(metrics), "mem": mem_rec, "acceptance": json.loads(acc) if acc!="null" else None,
       "power_temp": json.loads(power) if power else {}, "llama_log_mem": wgt,
       "manifest": {"flags": flags, "seed": int(seed), "reps": int(reps), "build": "0b1bad1",
                    "rocm": rocm, "kernel": os.popen("uname -r").read().strip(),
                    "date": datetime.date.today().isoformat()}}
json.dump(rec, open(out, "w"), indent=2)
print("wrote", out)
PY
rm -f "$LOG"
