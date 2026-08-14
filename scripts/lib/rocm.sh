#!/usr/bin/env bash
# Shared ROCm toolchain selection. Sourcing this file has no side effects.

rocm_prefix_is_valid() {
    local prefix="$1"
    [ -x "$prefix/bin/hipcc" ] && [ -x "$prefix/bin/rocminfo" ]
}

resolve_rocm_prefix() {
    local recommended="${1:-${HOME:?HOME is required}/rocm-7.14.0}"
    local fallback="${2:-/opt/rocm}"
    local selected source

    if [ -n "${ROCM_PREFIX:-}" ]; then
        selected="$ROCM_PREFIX"
        source="ROCM_PREFIX override"
    elif [ -n "${ROCM_PATH:-}" ]; then
        selected="$ROCM_PATH"
        source="ROCM_PATH override"
    elif rocm_prefix_is_valid "$recommended"; then
        selected="$recommended"
        source="recommended default"
    elif rocm_prefix_is_valid "$fallback"; then
        selected="$fallback"
        source="historical fallback"
    else
        echo "ERROR: no usable ROCm installation found." >&2
        echo "       Checked $recommended (recommended ROCm 7.14) and $fallback (historical fallback)." >&2
        echo "       Install 7.14 with: bash scripts/install-rocm-7.14.sh" >&2
        echo "       Or set ROCM_PREFIX=/path/to/rocm explicitly." >&2
        return 1
    fi

    if ! rocm_prefix_is_valid "$selected"; then
        echo "ERROR: selected ROCm prefix is incomplete: $selected" >&2
        echo "       Expected executable bin/hipcc and bin/rocminfo." >&2
        return 1
    fi

    ROCM_PREFIX="$selected"
    ROCM_PATH="$selected"
    ROCM_SELECTION_SOURCE="$source"
    export ROCM_PREFIX ROCM_PATH ROCM_SELECTION_SOURCE
}

detect_rocm_version() {
    local prefix="$1"
    local version=""

    if [ -r "$prefix/.info/version" ]; then
        IFS= read -r version < "$prefix/.info/version" || true
    fi
    if [ -z "$version" ]; then
        version="$("$prefix/bin/hipcc" --version 2>/dev/null | sed -n 's/^HIP version: *//p' | head -n 1)"
    fi
    [ -n "$version" ] || {
        echo "ERROR: could not detect the ROCm version at $prefix." >&2
        return 1
    }
    printf '%s\n' "$version"
}

rocm_track() {
    case "$1" in
        7.14.*|7.14) printf '%s\n' "recommended" ;;
        7.2.*|7.2) printf '%s\n' "historical reference" ;;
        *) printf '%s\n' "custom / unvalidated" ;;
    esac
}

print_selected_rocm() {
    local version="$1"
    echo "Selected ROCm:"
    echo "  prefix: $ROCM_PREFIX"
    echo "  version: $version"
    echo "  track: $(rocm_track "$version")"
    echo "  source: $ROCM_SELECTION_SOURCE"
}
