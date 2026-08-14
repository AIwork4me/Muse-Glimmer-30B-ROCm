#!/usr/bin/env bash
# Compare baseline and actively drafting DFlash output across the deterministic
# six-prompt Study 1 corpus, plus the original 17x23 arithmetic smoke prompt.
set -euo pipefail

HERE="$(cd "$(dirname "$0")/.." && pwd)"
cd "$HERE"

LLAMA="${LLAMA_BIN:-$HERE/third_party/llama.cpp/build/bin/llama-server}"
PROMPTS="$HERE/scripts/prompt-sets/muse-glimmer-diverse.json"
TMP_DIR="$(mktemp -d)"
CURRENT_PID=""

cleanup() {
    if [ -n "$CURRENT_PID" ]; then
        kill "$CURRENT_PID" 2>/dev/null || true
        wait "$CURRENT_PID" 2>/dev/null || true
    fi
    rm -rf "$TMP_DIR"
}
trap cleanup EXIT INT TERM

[ -x "$LLAMA" ] || {
    echo "ERROR: llama-server missing: $LLAMA" >&2
    exit 1
}
for artifact in models/muse-glimmer-30B-kquant-17gb.gguf models/dflash-kquant.gguf; do
    [ -f "$artifact" ] || {
        echo "ERROR: required artifact missing: $artifact" >&2
        exit 1
    }
done
if curl -fsS --connect-timeout 1 http://127.0.0.1:8090/health >/dev/null 2>&1; then
    echo "ERROR: port 8090 is already serving; stop that process first." >&2
    exit 1
fi

run_mode() {
    local output="$1"
    local log="$2"
    shift 2
    local extra=("$@")

    "$LLAMA" -m models/muse-glimmer-30B-kquant-17gb.gguf         -ngl 999 -np 1 -c 8192 --jinja --temp 0 --seed 0         --port 8090 "${extra[@]}" >"$log" 2>&1 &
    CURRENT_PID=$!

    local ready=0
    for _ in $(seq 1 120); do
        if curl -fsS http://127.0.0.1:8090/health >/dev/null 2>&1; then
            ready=1
            break
        fi
        sleep 1
    done
    if [ "$ready" -ne 1 ]; then
        echo "ERROR: server failed to start; log follows" >&2
        tail -100 "$log" >&2
        exit 1
    fi

    python3 - "$PROMPTS" "$output" <<'PY'
import json
import sys
import urllib.request

prompt_set = json.load(open(sys.argv[1]))
prompts = list(prompt_set["prompts"])
prompts.append({
    "category": "arithmetic-smoke",
    "text": "What is 17 * 23? Reply with just the number.",
})
results = []
for prompt in prompts:
    body = json.dumps({
        "model": "muse-glimmer-30B",
        "messages": [{"role": "user", "content": prompt["text"]}],
        "max_tokens": 256,
        "temperature": 0,
        "seed": 0,
        "chat_template_kwargs": {"reasoning_strength": "high"},
    }).encode()
    request = urllib.request.Request(
        "http://127.0.0.1:8090/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=600) as response:
        message = json.load(response)["choices"][0]["message"]
    results.append({
        "category": prompt["category"],
        "message": message,
    })
json.dump(results, open(sys.argv[2], "w"), ensure_ascii=False, indent=2)
PY

    kill "$CURRENT_PID" 2>/dev/null || true
    wait "$CURRENT_PID" 2>/dev/null || true
    CURRENT_PID=""
}

run_mode "$TMP_DIR/baseline.json" "$TMP_DIR/baseline.log"
run_mode "$TMP_DIR/dflash.json" "$TMP_DIR/dflash.log"     -md models/dflash-kquant.gguf -ngld 99     --spec-type draft-dflash --spec-draft-n-max 15

python3 scripts/capture_proc.py draft <"$TMP_DIR/dflash.log" >"$TMP_DIR/acceptance.json"
python3 - "$TMP_DIR/baseline.json" "$TMP_DIR/dflash.json" "$TMP_DIR/acceptance.json" <<'PY'
import json
import sys

baseline = json.load(open(sys.argv[1]))
dflash = json.load(open(sys.argv[2]))
acceptance = json.load(open(sys.argv[3]))
if not acceptance.get("draft_tokens") or not acceptance.get("accepted_draft_tokens"):
    raise SystemExit("FAIL: DFlash produced no recorded drafts; equivalence would be trivial")

def canonical_bytes(message):
    return json.dumps(
        message, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


study_matches = 0
failures = []
for base, draft in zip(baseline, dflash, strict=True):
    category = base["category"]
    if canonical_bytes(base["message"]) != canonical_bytes(draft["message"]):
        failures.append(category)
        print(f"MISMATCH: {category}")
    else:
        print(f"byte-identical: {category}")
        if category != "arithmetic-smoke":
            study_matches += 1

smoke = next(item for item in baseline if item["category"] == "arithmetic-smoke")
if (smoke["message"].get("content") or "").strip() != "391":
    failures.append("arithmetic-smoke-answer")
if failures:
    raise SystemExit("FAIL: " + ", ".join(failures))
print(f"PASS: {study_matches}/6 Study 1 prompts byte-identical")
print("PASS: arithmetic smoke output = 391")
print(
    "DFlash engaged: "
    f"{acceptance['accepted_draft_tokens']}/{acceptance['draft_tokens']} "
    "draft tokens accepted"
)
PY
