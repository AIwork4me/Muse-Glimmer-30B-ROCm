#!/usr/bin/env bash
# W7900 Study-2 reproduction driver (runs in the container OR on the host).
# Reproduces every W7900 number in docs/results/w7900-gfx1100.md:
#   {17gb,dynamic} x baseline{c=1,4,16,32} + DFlash{c=1,4}  = 12 full 5-rep cells.
# Reuses the committed harness (scripts/gguf-bench-cell.sh / bench_client.py /
# gguf_bench_args.py / capture_proc.py) UNMODIFIED; adds only this glue + renders.
set -uo pipefail
REPO="${REPO_DIR:-/workspace}"
GLUE="${GLUE_DIR:-$REPO/scripts/w7900-repro}"
MATRIX="${MATRIX_OUTDIR:-$GLUE/_out/matrix-w7900-gfx1100}"
LLAMA="${LLAMA_BIN:-/llamacpp_workspace/bin/llama-server}"
CONF="${STUDY_CONF:-configs/gguf-bench/study2.conf}"
IMG="${IMAGE_TAG:-unknown}"
export PATH="$GLUE/_uvshim:$PATH" LLAMA_BIN="$LLAMA" MATRIX_OUTDIR="$MATRIX"
cd "$REPO" || { echo "repo not found: $REPO"; exit 1; }
mkdir -p "$MATRIX"
log(){ echo "[repro $(date +%T)] $*"; }

# --- prerequisites (models resolve under <repo>/models; see 00_prepare.sh) ---
[ -f models/muse-glimmer-30B-kquant-17gb.gguf ]    || { echo "17gb model missing under $REPO/models (run 00_prepare.sh)"; exit 1; }
[ -f models/muse-glimmer-30B-kquant-dynamic.gguf ] || { echo "dynamic model missing under $REPO/models (run 00_prepare.sh)"; exit 1; }
python3 -c 'import aiohttp' 2>/dev/null || pip install -q --disable-pip-version-check aiohttp || { echo "aiohttp install failed"; exit 1; }
"$LLAMA" --version > "$MATRIX/llama-server-version.txt" 2>&1 || true
VER="$(head -1 "$MATRIX/llama-server-version.txt" 2>/dev/null || echo unknown)"
log "harness ok; llama-server=$LLAMA ; uv=$(command -v uv)"

# --- one full 5-rep cell (retry once; dynamic@high-c can crash transiently) ---
run_full(){  # weight dflash np
  local W=$1 D=$2 N=$3
  local OUT="$MATRIX/cell-study2-${W}-np${N}-df${D}-vis0.json"
  if [ "${RESUME:-1}" = "1" ] && [ -f "$OUT" ]; then log "FULL $W df$D c$N -> SKIP (resume: exists)"; return 0; fi
  log "FULL cell weight=$W dflash=$D np=$N"
  local a
  for a in 1 2; do
    bash scripts/gguf-bench-cell.sh study2 "$W" "$D" 0 "$N" "$CONF" && return 0
    log "  attempt $a FAILED; retry"; pkill -x llama-server 2>/dev/null; sleep 3
  done
  log "  CELL FAILED weight=$W dflash=$D np=$N"; return 1
}

log "=== REPRODUCTION: 12 cells (Study-2 matrix + c=32) -> $MATRIX ==="
START=$(date +%s)
# baseline @ c=1,4,16,32 and DFlash @ c=1,4 (per weight)
for W in 17gb dynamic; do
  run_full "$W" 0 1;  run_full "$W" 1 1
  run_full "$W" 0 4;  run_full "$W" 1 4
  run_full "$W" 0 16
  run_full "$W" 0 32
done
END=$(date +%s); log "wall time: $(( (END-START)/60 )) min"

# --- provenance + render artifacts ---
python3 - "$MATRIX" "$VER" "$IMG" <<'PY'
import glob,json,os,sys
outdir,ver,img=sys.argv[1:]; n=0
for p in glob.glob(os.path.join(outdir,"cell-study2-*.json")):
    d=json.load(open(p)); m=d.setdefault("manifest",{})
    m["build"]=f"image:{img} (llama.cpp {ver})"; m["host"]="Radeon PRO W7900 (gfx1100)"
    m["gfx"]="gfx1100"; m["image"]=img; m["gpu_index"]=0
    json.dump(d,open(p,"w"),indent=2); n+=1
print("provenance fixed for",n,"cells")
PY
python3 "$GLUE/render_matrix_safe.py" "$MATRIX" > "$MATRIX/matrix.md"
python3 "$GLUE/render_headline.py"    "$MATRIX" > "$MATRIX/headline.md"
log "DONE. Evidence in $MATRIX (headline.md + matrix.md + 12 cells)"
echo; echo "================= HEADLINE ================="; cat "$MATRIX/headline.md"
