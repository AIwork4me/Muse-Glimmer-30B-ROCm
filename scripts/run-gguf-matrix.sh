#!/usr/bin/env bash
# Drive a full study (or all three): enumerate cells from the per-study configs,
# randomize their order (to control for thermal drift across the long run), run
# each via gguf-bench-cell.sh, then render the markdown report.
#
# Usage: run-gguf-matrix.sh [--dry-run] [study1|study2|study3|all]
#   (default study: all)
#   --dry-run: enumerate + shuffle, print the cell list, per-study counts, and
#              the order, then exit 0. No cells run, no report rendered.
#              Use to verify the matrix shape (study1=4, study2=12, study3=5).
#
# SAFETY: NEVER use `pkill -f llama-server` (or any pattern that matches this
# driver's own shell) in this file — it self-terminates with exit 144. Each
# cell's llama-server is launched, health-checked, and killed BY PID inside
# gguf-bench-cell.sh; this driver only invokes that script per cell.
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"; cd "$HERE"

DRY_RUN=0
WHICH="all"
for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY_RUN=1 ;;
    study1|study2|study3|all) WHICH="$arg" ;;
    *) echo "usage: $0 [--dry-run] [study1|study2|study3|all]" >&2; exit 2 ;;
  esac
done
STUDIES=("$WHICH"); [ "$WHICH" = "all" ] && STUDIES=(study1 study2 study3)

# --- Enumerate cells -------------------------------------------------------
# Each cell is a 6-field record read back by the run loop below:
#   "<study> <weight> <dflash:0|1> <vision:0|1> <np> <conf-path>"
# study3 filter: keep the {17gb,dynamic}x{c1,c4} baseline (4 cells) plus the
# single 17gbxDFlashxc1 probe (1 cell) = 5 total; skip the other dflash cells
# (dynamic/dflash and 17gb/dflash/c4 are out of scope for the vision axis).
CELLS=()
COUNTS=()
for ST in "${STUDIES[@]}"; do
  CONF="configs/gguf-bench/$ST.conf"
  # shellcheck source=/dev/null
  . "$CONF"
  count=0
  for W in $WEIGHTS; do for D in $DFS; do for N in $NPS; do
    if [ "$ST" = "study3" ] && [ "$D" = "1" ] && { [ "$W" != "17gb" ] || [ "$N" != "1" ]; }; then
      continue
    fi
    CELLS+=("$ST $W $D ${VISION:-0} $N $CONF")
    count=$((count+1))
  done; done; done
  COUNTS+=("$count")
done

# --- Randomize order (controls for thermal drift across the long run) ------
# Record the order to /tmp/matrix-order.txt so a post-mortem can correlate any
# failures or thermal throttling to a cell's position in the sequence.
ORDER_FILE=/tmp/matrix-order.txt
if [ "${#CELLS[@]}" -gt 0 ]; then
  printf '%s\n' "${CELLS[@]}" | shuf > "$ORDER_FILE"
else
  : > "$ORDER_FILE"
fi

echo "Running ${#CELLS[@]} cells in randomized order (see $ORDER_FILE):"
cat "$ORDER_FILE" >&2

if [ "$DRY_RUN" -eq 1 ]; then
  echo
  echo "Per-study cell counts:"
  for i in "${!STUDIES[@]}"; do
    echo "  ${STUDIES[$i]}: ${COUNTS[$i]}"
  done
  echo
  echo "(--dry-run: not executing cells, not rendering report.)"
  exit 0
fi

# --- Run each cell ---------------------------------------------------------
# gguf-bench-cell.sh owns all server lifecycle (launch/health/teardown by PID);
# a failing cell is logged but does not abort the matrix (render what succeeded).
while read -r ST W D V N CONF; do
  [ -z "$ST" ] && continue
  echo "=== cell: $ST weight=$W dflash=$D vision=$V np=$N ==="
  bash scripts/gguf-bench-cell.sh "$ST" "$W" "$D" "$V" "$N" "$CONF" \
    || echo "  CELL FAILED (logged above)"
done < "$ORDER_FILE"

# --- Render the markdown report -------------------------------------------
# render_matrix.py globs docs/results/matrix/cell-*.json and prints markdown.
mkdir -p docs/results/matrix
uv run --no-sync python scripts/render_matrix.py > docs/results/matrix/matrix.md
echo "rendered docs/results/matrix/matrix.md"
