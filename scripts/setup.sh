#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python3 -m venv "$root/.venv"
"$root/.venv/bin/python" -m pip install --upgrade pip
"$root/.venv/bin/python" -m pip install -r "$root/requirements.txt"

for command in gcc g++ make pdfcrop pdfinfo pdftoppm yosys iverilog vvp; do
    command -v "$command" >/dev/null || {
        echo "missing system prerequisite: $command" >&2
        exit 1
    }
done

echo "PipeDRAM environment is ready: $root/.venv"
