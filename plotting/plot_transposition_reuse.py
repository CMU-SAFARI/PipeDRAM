#!/usr/bin/env python3
"""
Transposition overhead vs. PUD operational intensity for micro-benchmarks and
the 12 applications (Fig. 4).

X-axis: operational intensity (element-ops per transposed byte)
Y-axis: transposition overhead (% of total execution time or energy)

Micro-benchmarks: 16 bit-serial ops at 32-bit, sweeping reuse 1..10^7.
Applications: *_bbop_statistics.csv and *_transposition.csv (worst case,
16 banks).
"""

import math
import os
import csv
import glob
import subprocess

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from adjustText import adjust_text

from pud_simulator.operations import get_operation
from pud_simulator.simulator import SIMDRAMSimulator
from pud_simulator.presets import pipedram_config

ARTIFACT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ggplot-like style
plt.rcParams.update({
    "figure.dpi":          150,
    "figure.facecolor":    "white",
    "font.family":         "sans-serif",
    "font.sans-serif":     ["Liberation Sans", "Arial", "Helvetica", "DejaVu Sans"],
    "font.size":           9,
    "axes.titlesize":      10,
    "axes.labelsize":      10,
    "xtick.labelsize":     9,
    "ytick.labelsize":     9,
    "legend.fontsize":     8,
    "axes.facecolor":      "white",
    "axes.edgecolor":      "black",
    "axes.linewidth":      0.8,
    "axes.grid":           True,
    "axes.grid.axis":      "y",
    "axes.axisbelow":      True,
    "grid.color":          "#f0f0f0",
    "grid.alpha":          1.0,
    "grid.linestyle":      "-",
    "grid.linewidth":      0.6,
    "xtick.color":         "black",
    "ytick.color":         "black",
    "xtick.direction":     "out",
    "ytick.direction":     "out",
})

# Parameters
BIT_PRECISION = 32
NUM_ELEMENTS = 67_108_864  # 64M
NUM_SUBARRAYS = 1
NUM_BANKS = 1

OP_CLASSES = {
    "Logarithmic": {
        "ops": ["and_reduction", "or_reduction", "xor_reduction", "bitcount"],
        "color": "#72A2D5",   # blue (from workload_normalized_combined)
    },
    "Linear": {
        "ops": ["abs", "addition", "subtraction", "equal",
                "greater", "greater_equal", "if_then_else", "max",
                "min", "relu"],
        "color": "#A296BF",   # purple
    },
    "Quadratic": {
        "ops": ["multiplication", "division"],
        "color": "#D28989",   # red
    },
}

OPERATIONS = []
OP_TO_CLASS = {}
for cls_name, cls_info in OP_CLASSES.items():
    for op in cls_info["ops"]:
        OPERATIONS.append(op)
        OP_TO_CLASS[op] = cls_name

# DDR4-2400 transposition cost model
T_RAS = 32.0; T_RP = 15.0; T_RC = T_RAS + T_RP
T_CK = 0.833; T_BURST = 8 * T_CK / 2
CACHE_LINE_BYTES = 64; DATATYPE_BYTES = 4
E_ACT_PRE_NJ = 10.74; E_COL_RD_NJ = 4.50; E_COL_WR_NJ = 3.99; E_IO_NJ = 5.79
E_L3_HIT_NJ = 0.945; E_L3_MISS_NJ = 1.904
DRAM_ENERGY_RD_NJ = E_ACT_PRE_NJ + E_COL_RD_NJ + E_IO_NJ
DRAM_ENERGY_WR_NJ = E_ACT_PRE_NJ + E_COL_WR_NJ + E_IO_NJ
ENERGY_PER_CL_WORST_NJ = BIT_PRECISION * (E_L3_MISS_NJ + DRAM_ENERGY_RD_NJ + E_L3_MISS_NJ + DRAM_ENERGY_WR_NJ)
BANK_TIME_PER_CL = BIT_PRECISION * T_RC
BUS_TIME_PER_CL = BIT_PRECISION * T_BURST
T_FLUSH_PER_CL = 80.0


def transposition_cost_worst(num_operands, num_elements, num_banks=1):
    """Worst-case transposition latency (ns) and energy (nJ)."""
    total_bytes = num_operands * num_elements * DATATYPE_BYTES
    total_cls = math.ceil(total_bytes / CACHE_LINE_BYTES)
    bank_lim = (total_cls / num_banks) * BANK_TIME_PER_CL
    bus_lim = total_cls * BUS_TIME_PER_CL
    flush_ns = total_cls * T_FLUSH_PER_CL
    return max(bank_lim, bus_lim) + flush_ns, total_cls * ENERGY_PER_CL_WORST_NJ


# Micro-benchmark simulations
cfg = pipedram_config()
sim = SIMDRAMSimulator(cfg)

op_info = {}  # op_name -> {compute_ns, compute_nj, num_operands, transp_ns, transp_nj, transp_bytes}
for op_name in OPERATIONS:
    gen_fn = get_operation(op_name)
    uprog = gen_fn(BIT_PRECISION)
    r = sim.simulate(uprog, num_elements=NUM_ELEMENTS,
                     num_compute_subarrays=NUM_SUBARRAYS,
                     num_compute_banks=NUM_BANKS, transposition_mode="none")
    num_ops = uprog.num_inputs + uprog.num_outputs
    t_lat, t_eng = transposition_cost_worst(num_ops, NUM_ELEMENTS)
    transp_bytes = num_ops * NUM_ELEMENTS * DATATYPE_BYTES
    op_info[op_name] = {
        "compute_ns": r.compute_latency_ns,
        "compute_nj": r.compute_energy_nj,
        "num_operands": num_ops,
        "transp_ns": t_lat,
        "transp_nj": t_eng,
        "transp_bytes": transp_bytes,
    }
    print(f"{op_name:20s}  compute={r.compute_latency_ns/1e6:.2f} ms  "
          f"transp={t_lat/1e6:.2f} ms  ops={num_ops}")

# Application results
RESULTS_DIR = os.environ.get(
    "PIPEDRAM_RESULTS_DIR",
    os.path.join(ARTIFACT_ROOT, "reference", "results"),
)

workloads = {}
for bbop_file in sorted(glob.glob(os.path.join(RESULTS_DIR, "*_bbop_statistics.csv"))):
    name = os.path.basename(bbop_file).replace("_bbop_statistics.csv", "")
    transp_file = os.path.join(RESULTS_DIR, f"{name}_transposition.csv")
    if not os.path.exists(transp_file):
        continue

    # Parse compute (SIMDRAM_64 summary)
    compute_lat = compute_eng = 0
    with open(bbop_file) as f:
        for line in f:
            if "Summary" in line and "SIMDRAM_64," in line:
                parts = [p.strip() for p in line.split(",")]
                compute_lat = float(parts[5])  # ms
                compute_eng = float(parts[6])  # mJ
                break

    # Parse transposition (worst, 16 banks)
    transp_lat = transp_eng = 0
    transp_in = transp_out = 0
    with open(transp_file) as f:
        for line in f:
            line = line.strip()
            if line.startswith("worst") and ", 16," in line:
                parts = [p.strip() for p in line.split(",")]
                transp_lat = float(parts[2])  # ms
                transp_eng = float(parts[3])  # mJ
                transp_in = int(parts[4])
                transp_out = int(parts[5])
                break

    if compute_lat > 0:
        transp_bytes = transp_in + transp_out
        workloads[name] = {
            "compute_lat_ms": compute_lat,
            "compute_eng_mj": compute_eng,
            "transp_lat_ms": transp_lat,
            "transp_eng_mj": transp_eng,
            "transp_bytes": transp_bytes,
            "lat_overhead_pct": 100 * transp_lat / (compute_lat + transp_lat),
            "eng_overhead_pct": 100 * transp_eng / (compute_eng + transp_eng),
            # Intensity proxy (not plotted): compute ms per GB transposed
            "ai_ms_per_gb": compute_lat / (transp_bytes / 1e9),
        }

# Micro-benchmark curves
# Sweep reuse (effective number of times each transposed element is used)
# Operational intensity (ai) = reuse * NUM_ELEMENTS / transp_bytes (element-ops per byte)
reuse_range = np.logspace(0, 7, 300)  # 1 to 10,000,000

micro_curves = {}  # op_name -> {ai: [], lat_pct: [], eng_pct: []}
for op_name in OPERATIONS:
    info = op_info[op_name]
    ais = []
    lat_pcts = []
    eng_pcts = []
    for reuse in reuse_range:
        total_compute_ns = reuse * info["compute_ns"]
        total_compute_nj = reuse * info["compute_nj"]
        # AI = total element-ops / transp_bytes
        total_ops = reuse * NUM_ELEMENTS
        ai = total_ops / info["transp_bytes"]  # element-ops per byte

        lat_pct = 100 * info["transp_ns"] / (total_compute_ns + info["transp_ns"])
        eng_pct = 100 * info["transp_nj"] / (total_compute_nj + info["transp_nj"])

        ais.append(ai)
        lat_pcts.append(lat_pct)
        eng_pcts.append(eng_pct)

    micro_curves[op_name] = {"ai": ais, "lat_pct": lat_pcts, "eng_pct": eng_pcts}

# Workload short names
WL_SHORT = {
    "2mm": "2mm", "3mm": "3mm", "gemm": "gmm", "gramschmidt": "gs",
    "covariance": "cov", "pca": "pca", "doitgen": "dg", "heartwall": "hw",
    "fdtd-apml": "fdtd", "kmeans": "km", "backprop": "bp", "x264": "x264",
}

# Application operational intensity (computed in the plotting loop): total
# element-ops are estimated as SIMDRAM compute time / median per-element
# compute time of the micro-benchmarks, then divided by the transposed bytes.
ns_per_elem = sorted([op_info[op]["compute_ns"] / NUM_ELEMENTS for op in OPERATIONS])
median_ns_per_elem = ns_per_elem[len(ns_per_elem) // 2]

# Plot (two panels, stacked vertically)
PANEL_CFG = [
    ("lat", "lat_pct", "Performance\nOverhead (%)",   1e-2, 2e4),
    ("eng", "eng_pct", "Energy\nOverhead (%)",   1e-2, 2e6),
]

LABEL_POS_LAT = {
    "bp":   ( 10,  -3, "left",   "center"),
    "fdtd": ( 10,  -6, "left",   "center"),
    "km":   ( 10,  -3, "left",   "center"),
    "cov":  (-14,   8, "right",  "bottom"),
    "3mm":  ( 8,    7, "left",   "bottom"),
    "2mm":  ( 8,    7, "left",   "bottom"),
    "hw":   (-14,  -2, "right",  "center"),
    "dg":   ( 8,    7, "left",   "bottom"),
    "gs":   ( 8,    7, "left",   "bottom"),
    "pca":  (-14,  14, "right",  "bottom"),
    "gmm":  ( 10,   7, "left",   "bottom"),
    "x264": ( 10,   5, "left",   "bottom"),
}

LABEL_POS_ENG = {
    "bp":   ( 12,  -4, "left",   "top"),
    "fdtd": ( 12,  -6, "left",   "top"),
    "km":   ( 12,  -3, "left",   "center"),
    "cov":  (-20,  18, "right",  "bottom"),
    "3mm":  ( 12,  10, "left",   "bottom"),
    "2mm":  ( 12,  10, "left",   "bottom"),
    "hw":   (-20,   6, "right",  "bottom"),
    "dg":   ( 6,   14, "left",   "bottom"),
    "gs":   ( 12,  10, "left",   "bottom"),
    "pca":  (-14,  22, "right",  "bottom"),
    "gmm":  ( 14,  10, "left",   "bottom"),
    "x264": ( 14,   3, "left",   "bottom"),
}

LABEL_POS_MAP = {"lat": LABEL_POS_LAT, "eng": LABEL_POS_ENG}

fig, axes = plt.subplots(2, 1, figsize=(5, 2.8))

for idx, (metric, m_key, ylabel, xlim_lo, xlim_hi) in enumerate(PANEL_CFG):
    ax = axes[idx]
    LABEL_POS = LABEL_POS_MAP[metric]

    # Plot micro-benchmark envelopes per class
    for cls_name, cls_info in OP_CLASSES.items():
        color = cls_info["color"]
        all_ais = micro_curves[cls_info["ops"][0]]["ai"]
        mins = np.full(len(all_ais), np.inf)
        maxs = np.full(len(all_ais), -np.inf)
        for op_name in cls_info["ops"]:
            vals = np.array(micro_curves[op_name][m_key])
            mins = np.minimum(mins, vals)
            maxs = np.maximum(maxs, vals)
        ax.fill_between(all_ais, mins, maxs, alpha=0.55, color=color,
                        label=cls_name, linewidth=0)
        ax.plot(all_ais, mins, color=color, alpha=0.7, linewidth=0.8)
        ax.plot(all_ais, maxs, color=color, alpha=0.7, linewidth=0.8)

    # Applications as labeled points
    wl_key = "lat_overhead_pct" if metric == "lat" else "eng_overhead_pct"

    for wl_name, wl in workloads.items():
        total_elem_ops = wl["compute_lat_ms"] * 1e6 / median_ns_per_elem
        ai = total_elem_ops / wl["transp_bytes"]
        overhead = wl[wl_key]
        short = WL_SHORT.get(wl_name, wl_name)

        ax.scatter(ai, overhead, color='#F8DB6F', alpha=0.7, s=28, zorder=10,
                   edgecolors='black', linewidths=0.6)

        dx, dy, ha, va = LABEL_POS.get(short, (6, 2, "left", "bottom"))
        ax.annotate(short, (ai, overhead), fontsize=7,
                    xytext=(dx, dy), textcoords='offset points',
                    ha=ha, va=va, clip_on=True,
                    arrowprops=dict(arrowstyle='-', color='#888888', lw=0.3,
                                    shrinkA=1, shrinkB=1))

    ax.set_ylabel(ylabel)
    ax.set_xscale('log')
    ax.set_xlim(xlim_lo, xlim_hi)
    ax.set_ylim(-2, 105)

    # Only bottom panel gets x-axis label
    if idx == 1:
        ax.set_xlabel("PUD Operational Intensity (PUD Operations per Transposed Byte)")
    else:
        ax.set_xlabel("")
        ax.tick_params(axis='x', labelbottom=False)

    # Horizontal grid only
    ax.grid(True, which='major', axis='y', color='#f0f0f0', linewidth=0.6)
    ax.grid(False, which='major', axis='x')
    ax.grid(False, which='minor')

    # Black border
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('black')
        spine.set_linewidth(0.8)

    # Legend only on the energy panel, bottom left
    if metric == "eng":
        leg = ax.legend(fontsize=7, loc='lower left', ncol=1,
                        framealpha=0.95, edgecolor='black',
                        facecolor='white', fancybox=False,
                        handlelength=1.0, handleheight=0.7,
                        columnspacing=0.8, handletextpad=0.4,
                        borderpad=0.3)
        for patch in leg.legend_handles:
            patch.set_edgecolor('black')
            patch.set_linewidth(0.6)
    else:
        ax.legend().remove()

fig.subplots_adjust(left=0.14, right=0.97, top=0.97, bottom=0.16, hspace=0.15)
outdir = os.environ.get(
    "PIPEDRAM_OUTPUT_DIR",
    os.path.join(ARTIFACT_ROOT, "generated", "figures"),
)
os.makedirs(outdir, exist_ok=True)
outfile = os.path.join(outdir, "transposition_reuse_combined.pdf")
fig.savefig(outfile, dpi=150)
subprocess.run(["pdfcrop", outfile, outfile], check=True, capture_output=True)
print(f"\nSaved: {outfile}")
plt.close()

# Summary
print(f"\n{'='*80}")
print(f"  Real workload overhead (worst-case, 16 banks)")
print(f"{'='*80}")
print(f"{'Workload':>14s}  {'Comp(ms)':>12s}  {'Transp(ms)':>12s}  {'Lat OH%':>8s}  {'Eng OH%':>8s}")
print(f"{'-'*14}  {'-'*12}  {'-'*12}  {'-'*8}  {'-'*8}")
for name in sorted(workloads, key=lambda n: workloads[n]["lat_overhead_pct"]):
    w = workloads[name]
    print(f"{name:>14s}  {w['compute_lat_ms']:>12.1f}  {w['transp_lat_ms']:>12.1f}  "
          f"{w['lat_overhead_pct']:>7.1f}%  {w['eng_overhead_pct']:>7.1f}%")

print("\nDone.")
