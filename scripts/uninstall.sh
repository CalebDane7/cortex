#!/usr/bin/env bash
set -euo pipefail
python3 -m cortex.scripts.install --uninstall
echo "Cortex hooks removed from settings.json"
echo "Memory data preserved at ~/.cortex/"
echo "To remove data: rm -rf ~/.cortex/"
