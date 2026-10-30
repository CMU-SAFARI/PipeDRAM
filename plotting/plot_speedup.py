#!/usr/bin/env python3
"""
Normalized execution time and energy of PipeDRAM vs. SIMDRAM, MIMDRAM, and
Proteus for the 12 applications (Fig. 10).

All bars are normalized to SIMDRAM without transposition (compute-only).
Bars are split into compute (solid) and transposition (hatched) components.

Output: workload_normalized_combined.pdf
"""

import os
import glob
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np


# ggplot-like style
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


ARTIFACT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.environ.get(
    "PIPEDRAM_RESULTS_DIR",
    os.path.join(ARTIFACT_ROOT, "reference", "results"),
)
OUTDIR = os.environ.get(
    "PIPEDRAM_OUTPUT_DIR",
    os.path.join(ARTIFACT_ROOT, "generated", "figures"),
)
os.makedirs(OUTDIR, exist_ok=True)

# Workload summaries from *_bbop_statistics.csv
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
            lat = float(parts[5])  # ms
            eng = float(parts[6])  # mJ
            mechs[mech] = {"lat": lat, "eng": eng}
    workloads[name] = mechs

# Configurations: (label, total key, compute-only key, color)
BASELINE_KEY = "SIMDRAM_64"

ALL_CONFIGS = [
    ("SIMDRAM",       "SIMDRAM_64_WORST",   "SIMDRAM_64",   "#F8DB6FB2"),  # yellow
    ("MIMDRAM",       "MIMDRAM_64_WORST",   "MIMDRAM_64",   "#eabc94B2"),  # light peach
    ("Proteus",       "PROTEUS_PIPE_WORST", "PROTEUS_PIPE", "#A6C680B2"),  # green
    ("PipeDRAM-Seq",  "PIPEDRAM_SEQ",       "PIPEDRAM_SEQ", "#D28989B2"),  # red
    ("PipeDRAM-Pipe", "PIPEDRAM_PIPE",      "PIPEDRAM_PIPE","#A296BFB2"),  # purple
    ("PipeDRAM-Hier", "PIPEDRAM_HIER",      "PIPEDRAM_HIER","#72A2D5B2"),  # blue
]

# Workload order on the x-axis
WL_ORDER = ["pca", "2mm", "3mm", "covariance", "doitgen", "fdtd-apml",
            "gemm", "gramschmidt", "backprop", "heartwall", "kmeans", "x264"]

# Short display names
WL_SHORT = {
    "2mm": "2mm", "3mm": "3mm", "gemm": "gmm", "gramschmidt": "gs",
    "covariance": "cov", "pca": "pca", "doitgen": "dg", "heartwall": "hw",
    "fdtd-apml": "fdtd", "kmeans": "km", "backprop": "bp", "x264": "x264",
}

# Panel layout
n_configs = len(ALL_CONFIGS)
n_wl = len(WL_ORDER)
bar_width = 0.16
group_width = n_configs * bar_width + 0.18
x_positions = np.arange(n_wl + 1) * group_width  # +1 for geomean


def plot_stacked_panel(ax, workloads, metric, ylabel):
    """Normalized stacked bars (compute + transposition) for a given metric."""
    all_data = []
    all_bar_vals = []

    for ci, (label, total_key, compute_key, color) in enumerate(ALL_CONFIGS):
        compute_vals = []
        transp_vals = []
        for wl_name in WL_ORDER:
            mechs = workloads.get(wl_name, {})
            baseline = mechs.get(BASELINE_KEY, {}).get(metric, 0)
            total = mechs.get(total_key, {}).get(metric, 0)
            compute = mechs.get(compute_key, {}).get(metric, 0)
            transp = max(0, total - compute)
            if baseline > 0:
                compute_vals.append(compute / baseline)
                transp_vals.append(transp / baseline)
            else:
                compute_vals.append(0)
                transp_vals.append(0)

        # Geomean
        norm_totals = [c + t for c, t in zip(compute_vals, transp_vals)]
        valid_totals = [v for v in norm_totals if v > 0]
        gm_total = gmean(valid_totals) if valid_totals else 0
        valid_compute = [c for c in compute_vals if c > 0]
        gm_compute = gmean(valid_compute) if valid_compute else 0
        gm_transp = max(0, gm_total - gm_compute)

        compute_vals.append(gm_compute)
        transp_vals.append(gm_transp)

        all_bar_vals.extend(c + t for c, t in zip(compute_vals, transp_vals) if c > 0)
        all_data.append((ci, label, color, compute_vals, transp_vals))

    # Y-axis limits
    if all_bar_vals:
        sorted_vals = sorted(v for v in all_bar_vals if v > 0)
        q75 = sorted_vals[int(len(sorted_vals) * 0.75)]
        ymax_cap = q75 * 10
        ymin = 10 ** (math.floor(math.log10(min(sorted_vals))) - 0.3)
        ymax = 10 ** (math.ceil(math.log10(ymax_cap)) + 0.3)
        ax.set_ylim(ymin, ymax)

    # Plot bars and annotate clipped ones
    for ci, label, color, compute_vals, transp_vals in all_data:
        x = x_positions + ci * bar_width

        clip_compute = []
        clip_transp = []
        for c, t in zip(compute_vals, transp_vals):
            total = c + t
            if total > ymax:
                clip_compute.append(min(c, ymax))
                clip_transp.append(max(0, ymax - c))
            else:
                clip_compute.append(c)
                clip_transp.append(t)

        ax.bar(x, clip_compute, bar_width, color=color,
               edgecolor='black', linewidth=0.4, zorder=3, label=label)
        ax.bar(x, clip_transp, bar_width, bottom=clip_compute, color=color,
               edgecolor='black', linewidth=0.4, zorder=3,
               hatch='///', alpha=0.5)

        for i, (c, t) in enumerate(zip(compute_vals, transp_vals)):
            total = c + t
            if total > ymax:
                if total >= 10000:
                    txt = f"{total/1000:.0f}k×"
                elif total >= 1000:
                    txt = f"{total/1000:.1f}k×"
                elif total >= 100:
                    txt = f"{total:.0f}×"
                else:
                    txt = f"{total:.1f}×"
                ax.text(x[i], ymax * 0.60, txt, ha='center', va='top',
                        fontsize=5.5, fontweight='bold', rotation=90, zorder=5,
                        color='black', bbox=dict(boxstyle='round,pad=0.1',
                        facecolor='white', edgecolor='none', alpha=0.7))

    ax.set_ylabel(ylabel, fontsize=11)


# Combined plot
fig, axes = plt.subplots(2, 1, figsize=(7.0, 4.2))

plot_stacked_panel(axes[0], workloads, "lat", "Normalized Exec. Time\n(log scale)")
plot_stacked_panel(axes[1], workloads, "eng", "Normalized Energy\n(log scale)")

for ax_idx, ax in enumerate(axes):
    # Baseline reference line at y=1.0
    ax.axhline(1.0, color='black', linewidth=0.8, linestyle="-.", alpha=0.6)

    # Geomean separator
    gm_x = (x_positions[-2] + (n_configs - 1) * bar_width + x_positions[-1]) / 2
    ax.axvline(gm_x, color='gray', linestyle='--', linewidth=0.8, zorder=2)

    # Black spines
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(0.8)

    ax.set_yscale('log')

    # Tight x-axis limits: remove dead space before first and after last group
    x_left = x_positions[0] - bar_width
    x_right = x_positions[-1] + (n_configs) * bar_width + bar_width * 0.5
    ax.set_xlim(x_left, x_right)

    # X-axis labels
    tick_labels = [WL_SHORT[w] for w in WL_ORDER] + ["GMean"]
    ax.set_xticks(x_positions + (n_configs - 1) * bar_width / 2)
    ax.set_xticklabels(tick_labels, fontsize=10)
    for lbl in ax.get_xticklabels():
        if lbl.get_text() == "GMean":
            lbl.set_fontweight('bold')

    # Panel label
    panel_label = "(a)" if ax_idx == 0 else "(b)"
    ax.text(0.008, 0.965, panel_label, transform=ax.transAxes,
            fontsize=12, fontweight='bold', va='top', ha='left', zorder=10)

# Legend
handles, labels = axes[0].get_legend_handles_labels()
handles.append(Patch(facecolor='white', edgecolor='black', linewidth=0.5))
labels.append('Comp.')
handles.append(Patch(facecolor='gray', edgecolor='black', linewidth=0.5,
                     hatch='///', alpha=0.5))
labels.append('Trans.')

fig.tight_layout(rect=[0.0, 0.0, 1, 0.96], h_pad=1.2)

# Place legend spanning the full width of the top axes (single row)
ax0_bbox = axes[0].get_position()
fig.legend(handles=handles, labels=labels, loc="upper left",
           ncol=len(ALL_CONFIGS) + 2, fontsize=9, frameon=False,
           bbox_to_anchor=(0.0, 0.998, ax0_bbox.x0 + ax0_bbox.width, 0.02),
           mode="expand", borderaxespad=0,
           handletextpad=0.3, handlelength=0.7, handleheight=0.7)

outfile = os.path.join(OUTDIR, "workload_normalized_combined.pdf")
fig.savefig(outfile, bbox_inches="tight")
print(f"Saved: {outfile}")
plt.close()

# Speedup table
BASELINES = [
    ("SIMDRAM",  "SIMDRAM_64_WORST",   "#e6a030"),
    ("MIMDRAM",  "MIMDRAM_64_WORST",   "#8bc34a"),
    ("Proteus",  "PROTEUS_PIPE_WORST", "#ff7043"),
]

print(f"\n{'='*100}")
print(f"  PipeDRAM-Hier speedup over baselines")
print(f"{'='*100}")
print(f"{'Workload':>14s}  {'vs SIMDRAM':>12s}  {'vs MIMDRAM':>12s}  {'vs Proteus':>12s}")
print("-" * 100)

speedups_lat = {"SIMDRAM": [], "MIMDRAM": [], "Proteus": []}

for wl_name in WL_ORDER:
    mechs = workloads.get(wl_name, {})
    pipe_hier = mechs.get("PIPEDRAM_HIER", {})
    if not pipe_hier:
        continue
    pipe_lat = pipe_hier["lat"]

    results = []
    for base_label, base_key, _ in BASELINES:
        base = mechs.get(base_key, {})
        if base and pipe_lat > 0:
            su = base["lat"] / pipe_lat
            speedups_lat[base_label].append(su)
            results.append(su)
        else:
            results.append(0)

    print(f"{wl_name:>14s}  {results[0]:>11.1f}×  {results[1]:>11.1f}×  {results[2]:>11.1f}×")

print("-" * 100)
print(f"{'Geomean':>14s}  "
      f"{gmean(speedups_lat['SIMDRAM']):>11.1f}×  "
      f"{gmean(speedups_lat['MIMDRAM']):>11.1f}×  "
      f"{gmean(speedups_lat['Proteus']):>11.1f}×")
