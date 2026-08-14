#!/usr/bin/env bash
# Stop and remove the benchmark container. Results persist in ./results (mounted).
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"; . "$HERE/config.env"
docker rm -f "$CONTAINER" 2>/dev/null && echo "removed $CONTAINER" || echo "no container $CONTAINER"
