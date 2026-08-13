#!/usr/bin/env bash
# Pure numeric dotted-version comparison. Sourcing this file has no side effects.

version_at_least() {
    local actual_core required_core
    local actual_major actual_minor actual_patch
    local required_major required_minor required_patch

    actual_core="${1%%-*}"
    required_core="${2%%-*}"
    IFS=. read -r actual_major actual_minor actual_patch _ <<<"$actual_core"
    IFS=. read -r required_major required_minor required_patch _ <<<"$required_core"
    actual_minor="${actual_minor:-0}"
    actual_patch="${actual_patch:-0}"
    required_minor="${required_minor:-0}"
    required_patch="${required_patch:-0}"

    [[ "$actual_major" =~ ^[0-9]+$ && "$actual_minor" =~ ^[0-9]+$ && "$actual_patch" =~ ^[0-9]+$ ]] || return 1
    [[ "$required_major" =~ ^[0-9]+$ && "$required_minor" =~ ^[0-9]+$ && "$required_patch" =~ ^[0-9]+$ ]] || return 1

    (( 10#$actual_major > 10#$required_major )) && return 0
    (( 10#$actual_major < 10#$required_major )) && return 1
    (( 10#$actual_minor > 10#$required_minor )) && return 0
    (( 10#$actual_minor < 10#$required_minor )) && return 1
    (( 10#$actual_patch >= 10#$required_patch ))
}
