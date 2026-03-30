#!/usr/bin/env bash
set -euo pipefail
# Install cortex-memory package and configure hooks
pip install -e "$(dirname "$0")/.." 2>/dev/null || pip3 install -e "$(dirname "$0")/.."
python3 -m cortex.scripts.install "$@"
