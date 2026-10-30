#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python="${root}/.venv/bin/python"
[[ -x "$python" ]] || python=python3

export PYTHONPATH="$root"
export PIPEDRAM_OUTPUT_DIR="${PIPEDRAM_OUTPUT_DIR:-$root/generated/figures}"
mkdir -p "$PIPEDRAM_OUTPUT_DIR"

"$python" "$root/tools/verify_area.py"
"$python" "$root/plotting/plot_area_comparison.py"
