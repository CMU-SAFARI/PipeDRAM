#!/usr/bin/env python3
"""
CPU-normalized energy efficiency (performance per Watt), Fig. 11.

EE = 1/Energy, so CPU-normalized EE = CPU_energy / System_energy.
Values above 1 are more energy-efficient than the CPU.

Systems: GPU, SIMDRAM, MIMDRAM, Proteus, PipeDRAM-Seq, PipeDRAM-Pipe,
PipeDRAM-Hier. SIMDRAM, MIMDRAM, and Proteus include common-case
transposition energy (no breakdown).
"""

import os
import glob
import csv
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# Style (matches workload_normalized_combined.pdf)
plt.rcParams.update({
    "figure.dpi":          150,
    "figure.facecolor":    "white",
    "font.family":         "sans-serif",
    "font.sans-serif":     ["Liberation Sans", "Arial", "Helvetica", "DejaVu Sans"],
    "font.size":           9,
    "axes.titlesize":      10,
    "axes.labelsize":      9,
    "xtick.labelsize":     8,
    "ytick.labelsize":     8,
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


def gmean(vals):
    if not vals or any(v <= 0 for v in vals):
        return 0.0
    return math.exp(sum(math.log(v) for v in vals) / len(vals))


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.environ.get(
    "PIPEDRAM_RESULTS_DIR", os.path.join(BASE_DIR, "reference", "results")
)
OUTDIR = os.environ.get(
    "PIPEDRAM_OUTPUT_DIR", os.path.join(BASE_DIR, "generated", "figures")
)
os.makedirs(OUTDIR, exist_ok=True)

# PUD results from *_bbop_statistics.csv
workloads = {}
for bbop_file in sorted(glob.glob(os.path.join(RESULTS_DIR, "*_bbop_statistics.csv"))):
    name = os.path.basename(bbop_file).replace("_bbop_statistics.csv", "")
    mechs = {}
    with open(bbop_file) as f:
        for line in f:
            if "Summary" not in line:
                continue
            parts = [p.strip() for p in line.split(",")]
            mech = parts[4]
            eng = float(parts[6])  # mJ
            mechs[mech] = eng
    workloads[name] = mechs

# CPU baseline energy (J -> mJ)
cpu_energy = {}  # mJ
summary_file = os.path.join(RESULTS_DIR, "baseline", "summary.csv")
with open(summary_file) as f:
    reader = csv.DictReader(f)
    for row in reader:
        name = row["workload"].strip()
        cpu_energy[name] = float(row["energy_j"]) * 1000.0  # J -> mJ

# GPU energy (mJ) measured at EXTRALARGE_DATASET, scaled to PAPER_DATASET
GPU_EXTRALARGE_ENERGY = {
    "2mm":         48_473,
    "3mm":         69_114,
    "gemm":        20_161,
    "gramschmidt": 853_247,
    "covariance":  760_490,
    "doitgen":     465_217,      # same total FLOPs (equal NQ*NR*NP^2)
    "heartwall":   12_597,       # test.avi: same PUD workload as test_4k.avi
    "kmeans":      2_985,        # same dataset (10K x 16384)
}

# Scaling factors from EXTRALARGE to PAPER dataset size
# O(n^3) kernels: (32000/4000)^3 = 512
# doitgen: PAPER NQ*NR*NP^2 = EXTRALARGE NQ*NR*NP^2 -> scale=1
# heartwall: same tracked points (51), same frames -> scale=1
# kmeans: same dataset -> scale=1
GPU_SCALE = {
    "2mm": 512, "3mm": 512, "gemm": 512,
    "gramschmidt": 512, "covariance": 512,
    "doitgen": 1, "heartwall": 1, "kmeans": 1,
}

gpu_energy = {}  # mJ at PAPER_DATASET
for wl, raw_e in GPU_EXTRALARGE_ENERGY.items():
    gpu_energy[wl] = raw_e * GPU_SCALE[wl]

# Configurations
# (label, key in bbop_statistics or "GPU", color)
CONFIGS = [
    ("GPU",           "GPU",                "#8C8C8CB2"),  # gray
    ("SIMDRAM",       "SIMDRAM_64_COMMON",  "#F8DB6FB2"),  # yellow
    ("MIMDRAM",       "MIMDRAM_64_COMMON",  "#eabc94B2"),  # light peach
    ("Proteus",       "PROTEUS_PIPE_COMMON","#A6C680B2"),  # green
    ("PipeDRAM-Seq",  "PIPEDRAM_SEQ",       "#D28989B2"),  # red
    ("PipeDRAM-Pipe", "PIPEDRAM_PIPE",      "#A296BFB2"),  # purple
    ("PipeDRAM-Hier", "PIPEDRAM_HIER",      "#72A2D5B2"),  # blue
]

WL_ORDER = ["pca", "2mm", "3mm", "covariance", "doitgen", "fdtd-apml",
            "gemm", "gramschmidt", "backprop", "heartwall", "kmeans", "x264"]

WL_SHORT = {
    "2mm": "2mm", "3mm": "3mm", "gemm": "gmm", "gramschmidt": "gs",
    "covariance": "cov", "pca": "pca", "doitgen": "dg", "heartwall": "hw",
    "fdtd-apml": "fdtd", "kmeans": "km", "backprop": "bp", "x264": "x264",
}

# CPU-normalized energy efficiency
n_configs = len(CONFIGS)
n_wl = len(WL_ORDER)
bar_width = 0.14
group_width = n_configs * bar_width + 0.18
# Two geomean groups: GMean-8 (applications with GPU measurements) and
# GMean-12 (all applications), separated by extra gaps for their labels
x_positions = np.arange(n_wl + 2, dtype=float) * group_width
x_positions[-2] += 0.40 * group_width
x_positions[-1] += 0.85 * group_width
GPU_WLS = [wl for wl in WL_ORDER if wl in gpu_energy]

fig, ax = plt.subplots(1, 1, figsize=(6.5, 2.1))

all_bar_vals = []
all_data = []

for ci, (label, key, color) in enumerate(CONFIGS):
    ee_vals = []
    for wl in WL_ORDER:
        cpu_e = cpu_energy.get(wl, 0)
        if key == "GPU":
            sys_e = gpu_energy.get(wl, 0)
        else:
            sys_e = workloads.get(wl, {}).get(key, 0)

        if cpu_e > 0 and sys_e > 0:
            ee = cpu_e / sys_e  # >1 means more efficient than CPU
            ee_vals.append(ee)
        else:
            ee_vals.append(0)

    # GMean-12 has no GPU bar: four applications lack GPU measurements
    valid8 = [v for wl, v in zip(WL_ORDER, ee_vals) if wl in GPU_WLS and v > 0]
    gm8 = gmean(valid8) if valid8 else 0
    valid12 = [v for v in ee_vals if v > 0]
    gm12 = gmean(valid12) if (key != "GPU" and len(valid12) == n_wl) else 0
    ee_vals.extend([gm8, gm12])

    all_bar_vals.extend(v for v in ee_vals if v > 0)
    all_data.append((ci, label, color, ee_vals))

# Y-axis limits (log scale)
if all_bar_vals:
    sorted_vals = sorted(v for v in all_bar_vals if v > 0)
    q75 = sorted_vals[int(len(sorted_vals) * 0.75)]
    ymax_cap = q75 * 10
    ymin = 10 ** (math.floor(math.log10(min(sorted_vals))) - 0.3)
    ymax = 10 ** (math.ceil(math.log10(ymax_cap)) + 0.3)
    ax.set_ylim(ymin, ymax)

# Plot bars
for ci, label, color, ee_vals in all_data:
    x = x_positions + ci * bar_width

    clip_vals = []
    for v in ee_vals:
        clip_vals.append(min(v, ymax) if v > 0 else 0)

    # Replace 0s with NaN to avoid plotting
    plot_vals = [v if v > 0 else np.nan for v in clip_vals]

    ax.bar(x, plot_vals, bar_width, color=color,
           edgecolor='black', linewidth=0.4, zorder=3, label=label)

    # Annotate clipped bars
    for i, v in enumerate(ee_vals):
        if v > ymax:
            if v >= 10000:
                txt = f"{v/1000:.0f}k"
            elif v >= 1000:
                txt = f"{v/1000:.1f}k"
            elif v >= 100:
                txt = f"{v:.0f}"
            else:
                txt = f"{v:.1f}"
            ax.text(x[i], ymax * 0.60, txt, ha='center', va='top',
                    fontsize=5.5, fontweight='bold', rotation=90, zorder=5,
                    color='black', bbox=dict(boxstyle='round,pad=0.1',
                    facecolor='white', edgecolor='none', alpha=0.7))

# Reference line at y=1 (CPU baseline)
ax.axhline(1.0, color='black', linewidth=0.8, linestyle="-.", alpha=0.6)

# Separator between the applications and the geomean groups
gm_x = (x_positions[-3] + (n_configs - 1) * bar_width + x_positions[-2]) / 2
ax.axvline(gm_x, color='gray', linestyle='--', linewidth=0.8, zorder=2)

# Spines
for spine in ax.spines.values():
    spine.set_visible(True)
    spine.set_color("black")
    spine.set_linewidth(0.8)

ax.set_yscale('log')

# X-axis
x_left = x_positions[0] - bar_width
x_right = x_positions[-1] + n_configs * bar_width + bar_width * 0.5
ax.set_xlim(x_left, x_right)

tick_labels = [WL_SHORT[w] for w in WL_ORDER] + ["GMean-8", "GMean-12"]
ax.set_xticks(x_positions + (n_configs - 1) * bar_width / 2)
ax.set_xticklabels(tick_labels, fontsize=10)
for lbl in ax.get_xticklabels():
    if lbl.get_text().startswith("GMean"):
        lbl.set_fontweight('bold')
        lbl.set_fontsize(7.5)  # fit "GMean-12" within one group width

ax.set_ylabel("CPU-Normalized\nPerformance/Watt (log)", fontsize=11)

# Legend
handles, labels = ax.get_legend_handles_labels()
fig.tight_layout(rect=[0.0, 0.0, 1, 0.92])

ax_bbox = ax.get_position()
fig.legend(handles=handles, labels=labels, loc="upper left",
           ncol=n_configs, fontsize=9, frameon=False,
           bbox_to_anchor=(0.0, 0.998, ax_bbox.x0 + ax_bbox.width, 0.02),
           mode="expand", borderaxespad=0,
           handletextpad=0.3, handlelength=0.7, handleheight=0.7)

outfile = os.path.join(OUTDIR, "energy_efficiency_cpu_normalized.pdf")
fig.savefig(outfile, bbox_inches="tight")
print(f"Saved: {outfile}")
plt.close()

# Text table
print(f"\n{'='*110}")
print(f"  CPU-Normalized Energy Efficiency (CPU_energy / System_energy)")
print(f"  >1 = more efficient than CPU, <1 = less efficient")
print(f"{'='*110}")

header = f"{'Workload':>14s}"
for label, _, _ in CONFIGS:
    header += f"  {label:>14s}"
print(header)
print("-" * 110)

gm_data = {label: [] for label, _, _ in CONFIGS}
gm8_data = {label: [] for label, _, _ in CONFIGS}

for wl in WL_ORDER:
    row = f"{wl:>14s}"
    cpu_e = cpu_energy.get(wl, 0)
    for label, key, _ in CONFIGS:
        if key == "GPU":
            sys_e = gpu_energy.get(wl, 0)
        else:
            sys_e = workloads.get(wl, {}).get(key, 0)

        if cpu_e > 0 and sys_e > 0:
            ee = cpu_e / sys_e
            gm_data[label].append(ee)
            if wl in GPU_WLS:
                gm8_data[label].append(ee)
            row += f"  {ee:>13.1f}×"
        else:
            row += f"  {'N/A':>13s}"
    print(row)

print("-" * 110)
for name, data, need in (("GMean-8", gm8_data, len(GPU_WLS)), ("GMean-12", gm_data, n_wl)):
    row = f"{name:>14s}"
    for label, _, _ in CONFIGS:
        vals = data[label]
        gm = gmean(vals) if len(vals) == need else 0
        row += f"  {gm:>13.1f}×" if gm > 0 else f"  {'N/A':>13s}"
    print(row)
