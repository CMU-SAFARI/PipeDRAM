#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mode="${1:-}"
case "$mode" in
    quick|full) ;;
    *)
        echo "usage: $0 quick|full" >&2
        exit 2
        ;;
esac

[[ "$(cat "$root/generated/mode.txt" 2>/dev/null)" == "$mode" ]] || {
    echo "generated output is not from a $mode reproduction" >&2
    exit 1
}

(cd "$root" && sha256sum -c reference_checksums.sha256 >/dev/null)
"$root/.venv/bin/python" "$root/tools/validate.py" --mode "$mode"
echo "$mode validation passed"
