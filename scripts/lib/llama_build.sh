#!/usr/bin/env bash
# Toolchain-aware llama.cpp build identity helpers.

canonical_rocm_prefix() {
    python3 - "$1" <<'PY'
from pathlib import Path
import sys

print(Path(sys.argv[1]).expanduser().resolve(strict=False))
PY
}

llama_build_dir() {
    local llama_dir="$1"
    local rocm_prefix="$2"
    local rocm_version="$3"
    local override="${4:-}"
    local canonical_prefix home_714 opt_rocm version_slug prefix_key

    if [ -n "$override" ]; then
        printf '%s\n' "$override"
        return 0
    fi

    canonical_prefix="$(canonical_rocm_prefix "$rocm_prefix")"
    home_714="$(canonical_rocm_prefix "$HOME/rocm-7.14.0")"
    opt_rocm="$(canonical_rocm_prefix /opt/rocm)"
    if [ "$canonical_prefix" = "$home_714" ]; then
        printf '%s\n' "$llama_dir/build-714"
    elif [ "$canonical_prefix" = "$opt_rocm" ]; then
        printf '%s\n' "$llama_dir/build"
    else
        version_slug="$(printf '%s' "$rocm_version" | tr -c '[:alnum:].-' '-')"
        prefix_key="$(python3 - "$canonical_prefix" <<'PY'
import hashlib
import sys

print(hashlib.sha256(sys.argv[1].encode()).hexdigest()[:12])
PY
)"
        printf '%s/build-rocm-%s-%s\n' "$llama_dir" "$version_slug" "$prefix_key"
    fi
}

write_llama_build_fingerprint() {
    local output="$1"
    local llama_commit="$2"
    local rocm_prefix="$3"
    local rocm_version="$4"
    local amdgpu_target="$5"
    local canonical_prefix hipcc_identity

    canonical_prefix="$(canonical_rocm_prefix "$rocm_prefix")"
    hipcc_identity="$("$rocm_prefix/bin/hipcc" --version 2>&1)"
    python3 - "$output" "$llama_commit" "$canonical_prefix" "$rocm_version" \
        "$hipcc_identity" "$amdgpu_target" <<'PY'
import json
from pathlib import Path
import sys

output, commit, prefix, version, hipcc, target = sys.argv[1:]
fingerprint = {
    "schema_version": 1,
    "llama_cpp_commit": commit,
    "rocm_prefix": prefix,
    "rocm_version": version,
    "hipcc": hipcc,
    "amdgpu_targets": [target],
    "cmake": {
        "GGML_HIP": True,
        "CMAKE_BUILD_TYPE": "Release",
        "AMDGPU_TARGETS": target,
        "ROCM_PATH": prefix,
        "hip_DIR": f"{prefix}/lib/cmake/hip",
    },
}
with Path(output).open("w", encoding="utf-8") as stream:
    json.dump(fingerprint, stream, indent=2, sort_keys=True)
    stream.write("\n")
PY
}

llama_build_fingerprint_matches() {
    local expected="$1"
    local recorded="$2"

    [ -f "$recorded" ] && cmp -s "$expected" "$recorded"
}

# True when $1 is the state gguf-quickstart.sh's own clone command creates:
# `git clone --filter=blob:none --no-checkout` leaves an empty worktree and no
# index file, which git's diff machinery misreads as every tracked path
# staged-deleted. That state cannot contain user work, so guards must not
# treat it as dirty (F-04).
llama_is_indexless_empty_clone() {
    local llama_dir="$1"

    [ -d "$llama_dir/.git" ] && [ ! -f "$llama_dir/.git/index" ] &&
        [ -z "$(find "$llama_dir" -mindepth 1 -maxdepth 1 \
            -not -name .git -print -quit 2>/dev/null)" ]
}

# Exit 0 when $1 has uncommitted tracked changes (worktree or index) that the
# quickstart must refuse to touch; exit 1 when it is safe to switch commits.
llama_has_tracked_changes() {
    local llama_dir="$1"

    if llama_is_indexless_empty_clone "$llama_dir"; then
        return 1
    fi
    ! git -C "$llama_dir" diff --quiet --ignore-submodules HEAD -- ||
        ! git -C "$llama_dir" diff --cached --quiet
}

# F-05: actionable dirty-checkout refusal. Names the situation, excerpts what
# git actually sees, and hands the user recovery verbs plus a docs anchor
# instead of dead-ending.
llama_refuse_dirty_checkout() {
    local llama_dir="$1"
    local situation="$2"
    local excerpt

    excerpt="$(git -C "$llama_dir" status --porcelain 2>/dev/null | head -n 10 || true)"
    {
        echo "ERROR: $llama_dir has uncommitted tracked changes;"
        echo "refusing to $situation (your changes are never discarded automatically)."
        echo "git status --porcelain says (first 10 lines):"
        if [ -n "$excerpt" ]; then
            printf '%s\n' "$excerpt" | sed 's/^/    /'
        else
            echo "    (git reported no changes; inspect the checkout manually)"
        fi
        echo "To keep your changes:   git -C '$llama_dir' stash, then rerun this script"
        echo "                        (git -C '$llama_dir' stash pop restores them later)."
        echo "To discard them:        git -C '$llama_dir' checkout -- . && rerun this script."
        echo "Details: docs/troubleshooting.md#dirty-llama-cpp-checkout"
    } >&2
}
