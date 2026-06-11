#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE="$(cd "$SCRIPT_DIR/../.." && pwd)"

python "$SCRIPT_DIR/edgepilot.py" \
  --workspace "$WORKSPACE" \
  demo \
  --output "$WORKSPACE/edgepilot_demo_run"
