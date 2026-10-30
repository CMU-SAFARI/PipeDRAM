#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
output="${1:-$root/generated/datasets/10000_16384.txt}"
output="$(realpath -m "$output")"
mkdir -p "$(dirname "$output")"

make -C "$root/workloads/kmeans" datagen.exe
(
    cd "$(dirname "$output")"
    "$root/workloads/kmeans/datagen.exe" 10000 16384 11
)

generated="$(dirname "$output")/10000_16384.txt"
if [[ "$generated" != "$output" ]]; then
    mv "$generated" "$output"
fi

lines="$(wc -l <"$output")"
[[ "$lines" -eq 10000 ]] || {
    echo "unexpected kmeans object count: $lines" >&2
    exit 1
}

checksum="$(sha256sum "$output" | awk '{print $1}')"
expected="c4d6982efb1ea5354c5de841bd1192f1a6824f1cc21277b37e61221e83c9d899"
[[ "$checksum" == "$expected" ]] || {
    echo "generated kmeans checksum differs from the Ubuntu 24.04 reference" >&2
    exit 1
}
printf '%s  %s\n' "$checksum" "$(basename "$output")" >"$output.sha256"
echo "Generated $output"
echo "SHA-256: $checksum"
