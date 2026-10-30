#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mode="${1:-}"
shift || true
heartwall_input=""

while (($#)); do
    case "$1" in
        --heartwall-input)
            heartwall_input="${2:?--heartwall-input requires a path}"
            shift 2
            ;;
        *)
            echo "unknown argument: $1" >&2
            exit 2
            ;;
    esac
done

case "$mode" in
    quick|full) ;;
    *)
        echo "usage: $0 quick|full [--heartwall-input PATH]" >&2
        exit 2
        ;;
esac

python="$root/.venv/bin/python"
[[ -x "$python" ]] || {
    echo "run ./scripts/setup.sh first" >&2
    exit 1
}

out="$root/generated"
figures="$out/figures"
results="$out/results"
logs="$out/logs"
mkdir -p "$figures" "$results" "$logs"
find "$figures" -maxdepth 1 -type f -delete
find "$logs" -maxdepth 1 -type f -delete

if [[ "$mode" == quick ]]; then
    results="$root/reference/results"
else
    args=()
    [[ -z "$heartwall_input" ]] || args+=(--heartwall-input "$heartwall_input")
    "$root/scripts/prepare_results.sh" "${args[@]}"
fi

export PYTHONPATH="$root"
export PIPEDRAM_RESULTS_DIR="$results"
export PIPEDRAM_OUTPUT_DIR="$figures"

plots=(
    plotting/plot_transposition_reuse.py
    plotting/plot_combined_scalability.py
    plotting/plot_speedup.py
    plotting/plot_energy_efficiency.py
    plotting/plot_precision_datasize_tradeoff.py
    plotting/plot_mat_utilization.py
    plotting/plot_exposed_carry.py
)

for plot in "${plots[@]}"; do
    name="$(basename "${plot%.py}")"
    echo "Generating $name"
    "$python" "$root/$plot" >"$logs/$name.log" 2>&1
done

"$root/scripts/reproduce_area.sh"
"$python" "$root/tools/collect_metrics.py" \
    --results "$results" --output "$out/metrics.json"
echo "$mode" >"$out/mode.txt"
echo "Reproduction complete: $figures"
