#!/usr/bin/env bash
# Assert DFlash greedy output is byte-identical to baseline for a fixed prompt.
#
# Greedy spec-decode (temp=0, seed fixed) must produce the exact same token
# sequence as the non-speculative baseline — speculative decoding is an exact
# equivalence under greedy sampling, so any divergence is a correctness bug
# worth documenting. We compare the `content` field of the chat-completions
# response from two servers (baseline vs +DFlash) started with identical flags.
#
# IMPORTANT: the DFlash server MUST launch with `--spec-type draft-dflash` so
# spec-decoding is actually engaged (llama-server's --spec-type defaults to
# `none`, which would load the draft model but never draft — making the
# equivalence check trivially pass). --spec-draft-n-max 16 = DFlash block_size.
#
# Uses port 8090 (cells use 8080) so it can coexist with an unrelated run, but
# in practice it should be invoked when NO cell server is holding VRAM, since
# two 17gb models side-by-side on a 32 GiB card can OOM.
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"; cd "$HERE"
LLAMA="$HERE/third_party/llama.cpp/build/bin/llama-server"
P='{"model":"muse-glimmer-30B","messages":[{"role":"user","content":"What is 17 * 23? Reply with just the number."}],"max_tokens":256,"temperature":0,"seed":0,"chat_template_kwargs":{"reasoning_strength":"high"}}'

get_content() { # $1 = extra server args
  "$LLAMA" -m models/muse-glimmer-30B-kquant-17gb.gguf -ngl 999 -np 1 -c 8192 --jinja --temp 0 --seed 0 --port 8090 $1 >/dev/null 2>&1 &
  local p=$!; trap "kill $p 2>/dev/null" RETURN
  for _ in $(seq 1 120); do curl -sf http://127.0.0.1:8090/health >/dev/null 2>&1 && break; sleep 1; done
  curl -sf http://127.0.0.1:8090/health >/dev/null || { echo "server (args: $1) failed to start" >&2; exit 1; }
  curl -s http://127.0.0.1:8090/v1/chat/completions -H 'Content-Type: application/json' -d "$P" \
    | python3 -c "import sys,json;print(json.load(sys.stdin)['choices'][0]['message']['content'])"
}
BASE=$(get_content "")
DF=$(get_content "-md models/dflash-kquant.gguf -ngld 99 --spec-type draft-dflash --spec-draft-n-max 16")
python3 - "$BASE" "$DF" <<'PY'
import sys
b, d = sys.argv[1], sys.argv[2]
print("baseline:", repr(b)); print("dflash  :", repr(d))
sys.exit(0 if b.strip() == d.strip() else 1)
PY
