#!/bin/bash
# Run 2mm at 3 increasing dataset sizes (3 runs each) and use regression
# to predict PAPER_DATASET (32K). All sizes chosen to exceed L3 cache
# so scaling behavior is consistent.

set -e

NTHREADS=32
SIZES=(4000 8000 12000)
NRUNS=3
TARGET=32000
RESULTS_FILE="extrapolation_results.csv"

echo "size,run,elapsed_ms,energy_j" > "$RESULTS_FILE"

for N in "${SIZES[@]}"; do
    echo "=== Building N=$N ==="
    gcc 2mm.c polybench.c ../util/bbop_manager.c \
        -DNI=$N -DNJ=$N -DNK=$N -DNL=$N \
        -DGET_TIME -DRUN_BASELINE -DTHREADS=${NTHREADS} \
        -fopenmp -lm -Ofast -march=native -funroll-loops -w \
        -o 2mm_extrap_${N}.exe

    for RUN in $(seq 1 $NRUNS); do
        echo "  Run $RUN/$NRUNS (N=$N)..."
        OUTPUT=$(./2mm_extrap_${N}.exe 2>/dev/null)

        ELAPSED=$(echo "$OUTPUT" | grep "Elapsed time:" | awk '{print $3}')
        ENERGY=$(echo "$OUTPUT" | grep "Energy:" | awk '{print $2}')

        echo "    Time: ${ELAPSED} ms, Energy: ${ENERGY} J"
        echo "$N,$RUN,$ELAPSED,$ENERGY" >> "$RESULTS_FILE"
    done

    rm -f 2mm_extrap_${N}.exe
done

echo ""
echo "=== Raw results ==="
cat "$RESULTS_FILE"
echo ""

# Use Python to average runs, fit, and extrapolate
python3 - "$RESULTS_FILE" "$TARGET" << 'PYEOF'
import sys, csv
import numpy as np

results_file = sys.argv[1]
target = int(sys.argv[2])

# Read all runs
data = {}
with open(results_file) as f:
    reader = csv.DictReader(f)
    for row in reader:
        size = int(row["size"])
        if size not in data:
            data[size] = {"times": [], "energies": []}
        data[size]["times"].append(float(row["elapsed_ms"]))
        data[size]["energies"].append(float(row["energy_j"]))

# Compute averages and std devs
sizes = sorted(data.keys())
avg_times, avg_energies = [], []
print("=== Per-size statistics ===")
print(f"{'Size':>8s}  {'Avg time (ms)':>15s}  {'Std (ms)':>10s}  {'CV%':>6s}  {'Avg energy (J)':>16s}  {'Std (J)':>10s}  {'CV%':>6s}")
for n in sizes:
    t = np.array(data[n]["times"])
    e = np.array(data[n]["energies"])
    avg_t, std_t = t.mean(), t.std()
    avg_e, std_e = e.mean(), e.std()
    cv_t = 100 * std_t / avg_t if avg_t > 0 else 0
    cv_e = 100 * std_e / avg_e if avg_e > 0 else 0
    print(f"{n:>8d}  {avg_t:>15.1f}  {std_t:>10.1f}  {cv_t:>5.1f}%  {avg_e:>16.1f}  {std_e:>10.1f}  {cv_e:>5.1f}%")
    avg_times.append(avg_t)
    avg_energies.append(avg_e)

sizes = np.array(sizes, dtype=float)
avg_times = np.array(avg_times)
avg_energies = np.array(avg_energies)

# Fit in log-log space: log(y) = a * log(n) + b  =>  y = e^b * n^a
log_sizes = np.log(sizes)

a_t, b_t = np.polyfit(log_sizes, np.log(avg_times), 1)
pred_time = np.exp(b_t) * target**a_t

a_e, b_e = np.polyfit(log_sizes, np.log(avg_energies), 1)
pred_energy = np.exp(b_e) * target**a_e

print()
print(f"=== Log-log regression ===")
print(f"Time exponent:   {a_t:.3f} (expected ~3.0 for O(n³))")
print(f"Energy exponent: {a_e:.3f}")
print()

# Show fit quality
print(f"=== Fit vs measured (averages) ===")
print(f"{'Size':>8s}  {'Meas time (ms)':>15s}  {'Pred time (ms)':>15s}  {'Err%':>6s}  {'Meas energy (J)':>16s}  {'Pred energy (J)':>16s}  {'Err%':>6s}")
for i, n in enumerate(sizes):
    pt = np.exp(b_t) * n**a_t
    pe = np.exp(b_e) * n**a_e
    et = 100 * abs(avg_times[i] - pt) / avg_times[i]
    ee = 100 * abs(avg_energies[i] - pe) / avg_energies[i]
    print(f"{int(n):>8d}  {avg_times[i]:>15.1f}  {pt:>15.1f}  {et:>5.1f}%  {avg_energies[i]:>16.1f}  {pe:>16.1f}  {ee:>5.1f}%")

print()
print(f"=== Prediction for N={target} ===")
print(f"Predicted elapsed time: {pred_time:.1f} ms ({pred_time/1000:.1f} s)")
print(f"Predicted energy:       {pred_energy:.1f} J")
print(f"Predicted avg power:    {pred_energy / (pred_time/1000):.1f} W")
PYEOF
