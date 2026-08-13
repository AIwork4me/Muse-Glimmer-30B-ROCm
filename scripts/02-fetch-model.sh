#!/usr/bin/env bash
# Fetch the validated Muse-Glimmer-30B BF16 artifact set.
#
# Defaults are reproducible: official Hugging Face, the revision recorded in
# configs/artifact-manifest.json, resumable downloads, and post-download SHA256
# verification. Expert overrides:
#   HF_ENDPOINT=https://hf-mirror.com   optional regional mirror
#   MODEL_REVISION=main                 latest/experimental, not validated here
#   USE_HF_DOWNLOAD=1                   stock single-stream weight download
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PATH="$HOME/.local/bin:$PATH"

for cmd in curl python3 uv; do
    command -v "$cmd" >/dev/null 2>&1 || {
        echo "ERROR: required command not found: $cmd" >&2
        exit 1
    }
done

read_artifact_field() {
    python3 - "$ROOT/configs/artifact-manifest.json" "$1" <<'PY'
import json, sys
data = json.load(open(sys.argv[1]))["sets"]["bf16"]
print(data[sys.argv[2]])
PY
}

MODEL_ID="$(read_artifact_field repository)"
VALIDATED_MODEL_REVISION="$(read_artifact_field revision)"
MODEL_REVISION="${MODEL_REVISION:-$VALIDATED_MODEL_REVISION}"
DEST="${MODEL_DEST:-models/Muse-Glimmer-30B}"
SHARDS=(model-00001-of-00002.safetensors model-00002-of-00002.safetensors)

export HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}"
export HF_HUB_DISABLE_XET=1
export NCONNS="${NCONNS:-24}"

echo "Fetching $MODEL_ID -> $DEST"
echo "  endpoint : $HF_ENDPOINT"
echo "  revision : $MODEL_REVISION"
if [ "$MODEL_REVISION" = "$VALIDATED_MODEL_REVISION" ]; then
    echo "  track    : validated reference (size + SHA256 verification enabled)"
else
    echo "  track    : latest/experimental override (not the published benchmark revision)"
fi

resolve_url="$HF_ENDPOINT/$MODEL_ID/resolve/$MODEL_REVISION/config.json"
if ! curl -fsS --retry 3 --retry-all-errors --connect-timeout 10      --max-time 30 -o /dev/null "$resolve_url"; then
    echo "ERROR: cannot resolve $MODEL_ID@$MODEL_REVISION via $HF_ENDPOINT" >&2
    echo "       Optional mirror example: HF_ENDPOINT=https://hf-mirror.com" >&2
    exit 1
fi

run_with_retries() {
    local attempts="$1"
    shift
    local attempt=1
    until "$@"; do
        if [ "$attempt" -ge "$attempts" ]; then
            echo "ERROR: command failed after $attempts attempts" >&2
            return 1
        fi
        echo "  attempt $attempt failed; retrying in 10s (downloads are resumable)..." >&2
        attempt=$((attempt + 1))
        sleep 10
    done
}

mkdir -p "$DEST"
started_at="$(date +%s)"

run_with_retries 10     uv run --no-sync hf download "$MODEL_ID" --revision "$MODEL_REVISION"         --local-dir "$DEST" --exclude "*.safetensors"

if [ "${USE_HF_DOWNLOAD:-0}" = "1" ]; then
    echo "  weights  : hf download (single stream)"
    run_with_retries 10         uv run --no-sync hf download "$MODEL_ID" --revision "$MODEL_REVISION"             --local-dir "$DEST" --include "*.safetensors"
else
    echo "  weights  : hf_parallel_get.py ($NCONNS range connections)"
    run_with_retries 10         uv run --no-sync python scripts/hf_parallel_get.py "$MODEL_ID"             "${SHARDS[@]}" --revision "$MODEL_REVISION" --local-dir "$DEST"             --concurrency "$NCONNS"
fi

if [ "$MODEL_REVISION" = "$VALIDATED_MODEL_REVISION" ]; then
    echo "Verifying validated BF16 artifacts ..."
    verify_failed=0
    mapfile -t VALIDATED_FILES < <(python3 - "$ROOT/configs/artifact-manifest.json" <<'PY_FILES'
import json
import sys
for item in json.load(open(sys.argv[1]))["sets"]["bf16"]["files"]:
    print(item["path"])
PY_FILES
)
    for artifact in "${VALIDATED_FILES[@]}"; do
        if ! python3 scripts/verify_artifacts.py bf16 "$DEST" "$artifact"; then
            verify_failed=1
            suffix="$(date +%s)"
            if [ -e "$DEST/$artifact" ]; then
                quarantine="$DEST/$artifact.corrupt.$suffix"
                mv "$DEST/$artifact" "$quarantine"
                echo "quarantined invalid artifact: $quarantine" >&2
            fi
            [ ! -e "$DEST/$artifact.parts.json" ] ||
                mv "$DEST/$artifact.parts.json" "$DEST/$artifact.parts.json.invalid.$suffix"
        fi
    done
    [ "$verify_failed" -eq 0 ] || {
        echo "ERROR: invalid artifacts were quarantined; rerun to download clean copies." >&2
        exit 1
    }
else
    echo "WARNING: override revision downloaded; validated-reference hashes were not applied." >&2
fi

elapsed=$(( $(date +%s) - started_at ))
bytes="$(du -sb "$DEST" | cut -f1)"
gib="$(awk -v b="$bytes" 'BEGIN { printf "%.2f", b / 1024 / 1024 / 1024 }')"
echo "OK: model at $DEST"
echo "Summary: ${gib} GiB, ${elapsed}s elapsed"
