#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
results="${1:?results directory required}"
heartwall_input="${2:-}"
mkdir -p "$results"

if [[ -n "$heartwall_input" ]]; then
    expected="c51d82f46b5c95c4257d906ab1f399209ee511c73e87eea32340c7547a0f0d81"
    actual="$(sha256sum "$heartwall_input" | awk '{print $1}')"
    [[ "$actual" == "$expected" ]] || {
        echo "heartwall input checksum mismatch" >&2
        exit 1
    }
fi

dataset="$root/generated/datasets/10000_16384.txt"
dataset_checksum="$dataset.sha256"
mkdir -p "$(dirname "$dataset")"
if [[ ! -f "$dataset" ]]; then
    "$root/scripts/generate_kmeans_input.sh" "$dataset"
elif [[ ! -f "$dataset_checksum" ]] || ! (
    cd "$(dirname "$dataset")"
    sha256sum -c "$(basename "$dataset_checksum")"
) >/dev/null 2>&1; then
    echo "kmeans dataset checksum mismatch" >&2
    exit 1
fi
expected_kmeans="c4d6982efb1ea5354c5de841bd1192f1a6824f1cc21277b37e61221e83c9d899"
actual_kmeans="$(sha256sum "$dataset" | awk '{print $1}')"
[[ "$actual_kmeans" == "$expected_kmeans" ]] || {
    echo "kmeans dataset does not match the paper-reproduction seed" >&2
    exit 1
}

run_one() {
    local name="$1"
    local target="$2"
    shift 2
    local dir="$root/workloads/$name"
    make -C "$dir" "$target"
    (
        cd "$dir"
        "./$target" "$@"
    )
    cp "$dir/bbop_statistics.csv" "$results/${name}_bbop_statistics.csv"
    cp "$dir/transposition.csv" "$results/${name}_transposition.csv"
}

run_one 2mm 2mm_pim_fast.exe
run_one 3mm 3mm_pim_fast.exe
run_one covariance covariance_pim_fast.exe
run_one doitgen doitgen_pim_fast.exe
run_one fdtd-apml fdtd-apml_pim_fast.exe
run_one gemm gemm_pim_fast.exe
run_one gramschmidt gramschmidt_pim_fast.exe
run_one backprop backprop_pim_fast.exe 134217728
run_one pca pca_pim_fast.exe -r 16000 -c 16000

run_one kmeans kmeans_pim_fast.exe -n 1 -i "$dataset"

if [[ -n "$heartwall_input" ]]; then
    run_one heartwall heartwall_pim_fast.exe "$heartwall_input" 20 8
else
    make -C "$root/workloads/heartwall" heartwall_pim_fast.exe
    echo "heartwall: no verified input supplied; using the paper result"
fi

echo "x264: using the paper result (SPEC source and media are not redistributed)"
