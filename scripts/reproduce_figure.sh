#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
figure="${1:-}"
mode="${2:-}"
shift 2 2>/dev/null || true
heartwall_input=""
reuse_results=false

while (($#)); do
    case "$1" in
        --heartwall-input)
            heartwall_input="${2:?--heartwall-input requires a path}"
            shift 2
            ;;
        --reuse-results)
            reuse_results=true
            shift
            ;;
        *)
            echo "unknown argument: $1" >&2
            exit 2
            ;;
    esac
done

case "$mode" in
    replot|full) ;;
    *)
        echo "usage: $0 4|9|10|11|12|13|14|area replot|full [--reuse-results] [--heartwall-input PATH]" >&2
        exit 2
        ;;
esac

case "$figure" in
    4) script="plotting/plot_transposition_reuse.py"; needs_results=true ;;
    9) script="plotting/plot_combined_scalability.py"; needs_results=false ;;
    10) script="plotting/plot_speedup.py"; needs_results=true ;;
    11) script="plotting/plot_energy_efficiency.py"; needs_results=true ;;
    12) script="plotting/plot_mat_utilization.py"; needs_results=false ;;
    13) script="plotting/plot_precision_datasize_tradeoff.py"; needs_results=false ;;
    14) script="plotting/plot_exposed_carry.py"; needs_results=false ;;
    area) script="plotting/plot_area_comparison.py"; needs_results=false ;;
    *)
        echo "unknown figure: $figure" >&2
        exit 2
        ;;
esac

python="$root/.venv/bin/python"
[[ -x "$python" ]] || {
    echo "run ./scripts/setup.sh first" >&2
    exit 1
}

if [[ "$mode" == replot ]]; then
    results="$root/reference/results"
elif [[ "$needs_results" == true ]]; then
    if [[ "$reuse_results" != true ]] || [[ "$(cat "$root/generated/results_mode.txt" 2>/dev/null)" != full ]]; then
        args=()
        [[ -z "$heartwall_input" ]] || args+=(--heartwall-input "$heartwall_input")
        "$root/scripts/prepare_results.sh" "${args[@]}"
    fi
    results="$root/generated/results"
else
    results="$root/reference/results"
fi

export PYTHONPATH="$root"
export PIPEDRAM_RESULTS_DIR="$results"
export PIPEDRAM_OUTPUT_DIR="$root/generated/figures"
mkdir -p "$PIPEDRAM_OUTPUT_DIR" "$root/generated/logs"

if [[ "$figure" == area ]]; then
    "$python" "$root/tools/verify_area.py"
fi

name="$(basename "${script%.py}")"
"$python" "$root/$script" | tee "$root/generated/logs/$name.log"
echo "Figure $figure complete ($mode mode)"
