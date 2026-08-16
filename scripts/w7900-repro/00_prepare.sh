#!/usr/bin/env bash
# Shared prerequisite for BOTH reproduction methods:
#   1. download the 4 Muse-Glimmer-30B GGUFs from OFFICIAL Hugging Face -> $MODELS_HOST
#   2. make sure <repo>/models resolves to them (gitignored)
#   3. verify size + GGUF magic
# Network access to huggingface.co is YOUR responsibility. Behind a firewall you
# may export a mirror yourself, e.g.  HF_ENDPOINT=https://hf-mirror.com bash 00_prepare.sh
# Safe to re-run: skips anything already correct.
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"; . "$HERE/config.env"
mkdir -p "$MODELS_HOST"
FILES=(muse-glimmer-30B-kquant-17gb.gguf muse-glimmer-30B-kquant-dynamic.gguf dflash-kquant.gguf mmproj-kquant.gguf)
declare -A SZ=(
  [muse-glimmer-30B-kquant-17gb.gguf]=$SZ_17GB
  [muse-glimmer-30B-kquant-dynamic.gguf]=$SZ_DYNAMIC
  [dflash-kquant.gguf]=$SZ_DFLASH
  [mmproj-kquant.gguf]=$SZ_MMPROJ)
ok_file(){ [ -f "$MODELS_HOST/$1" ] && [ "$(stat -c %s "$MODELS_HOST/$1" 2>/dev/null)" = "${SZ[$1]}" ] && [ "$(head -c4 "$MODELS_HOST/$1" 2>/dev/null)" = "GGUF" ]; }

echo "=== 1/3 download from Hugging Face ($HF_REPO) -> $MODELS_HOST ==="
need=(); for f in "${FILES[@]}"; do if ok_file "$f"; then echo "  present: $f"; else need+=("$f"); fi; done
if [ "${#need[@]}" -gt 0 ]; then
  command -v python3 >/dev/null || { echo "  ERROR: python3 required"; exit 1; }
  python3 -c 'import huggingface_hub' 2>/dev/null \
    || python3 -m pip install --user -q huggingface_hub \
    || { echo "  ERROR: cannot import/install huggingface_hub"; exit 1; }
  echo "  downloading: ${need[*]}  (endpoint: ${HF_ENDPOINT:-official huggingface.co})"
  # Export (not an inline `VAR=x cmd` built from a parameter expansion: bash
  # treats the expanded word as the command name and fails with "No such file
  # or directory" whenever HF_ENDPOINT is set). huggingface_hub reads it.
  export HF_ENDPOINT
  python3 - "$HF_REPO" "$MODELS_HOST" "${need[@]}" <<'PY'
import os, sys
from huggingface_hub import hf_hub_download
repo, outdir = sys.argv[1], sys.argv[2]
for f in sys.argv[3:]:
    print("   ->", f, flush=True)
    hf_hub_download(repo_id=repo, filename=f, local_dir=outdir,
                    token=os.environ.get("HF_TOKEN") or None)
PY
fi

echo "=== 2/3 make <repo>/models resolve to the weights ==="
if [ "$MODELS_HOST" = "$REPO_DIR/models" ]; then
  echo "  models live at $REPO_DIR/models (nothing to link)"
elif [ -L "$REPO_DIR/models" ] || [ ! -e "$REPO_DIR/models" ]; then
  ln -sfn "$MODELS_HOST" "$REPO_DIR/models"; echo "  $REPO_DIR/models -> $MODELS_HOST"
else
  echo "  WARN: $REPO_DIR/models exists and is not a symlink; left as-is."
fi

echo "=== 3/3 verify (size + GGUF magic) ==="
rc=0; for f in "${FILES[@]}"; do if ok_file "$f"; then echo "  OK   $f"; else echo "  FAIL $f (expected ${SZ[$f]} B, GGUF magic)"; rc=1; fi; done
[ $rc -eq 0 ] && echo "PREP COMPLETE — ready for run_all.sh (Method 1) or run_host.sh (Method 2)" || { echo "PREP INCOMPLETE"; exit 1; }
