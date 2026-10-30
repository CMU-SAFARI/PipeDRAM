#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
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

results="$root/generated/results"
mkdir -p "$results/baseline"
find "$results" -maxdepth 2 -type f -delete
cp "$root"/reference/results/*_bbop_statistics.csv "$results/"
cp "$root"/reference/results/*_transposition.csv "$results/"
cp "$root/reference/results/cpu_gpu_results.csv" "$results/"
cp "$root/reference/results/baseline/summary.csv" "$results/baseline/"

"$root/scripts/run_workloads.sh" "$results" "$heartwall_input"
printf 'full\n' >"$root/generated/results_mode.txt"
echo "Full workload results prepared in $results"
