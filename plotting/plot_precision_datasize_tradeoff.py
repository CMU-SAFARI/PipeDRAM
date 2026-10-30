#!/usr/bin/env python3
"""
PipeDRAM-Hier vs. SIMDRAM (compute-only) trade-off over both bit-precision (N)
and data size (M) (Fig. 13). Fig. 9 fixes M=64M (steady state); this figure
shows the full crossover surface, exposing two effects that single slice hides:

  - Pipeline fill: PipeDRAM amortizes its (n-1)*II fill only at large M.
  - SIMD-width utilization: SIMDRAM's 32x-wider batch (65,536 vs 2,048 lanes
    x32 subarrays) is under-filled at small M, erasing its lane advantage.

Color = log2(PipeDRAM-Hier throughput / SIMDRAM throughput), compute-only,
32 subarrays, 1 bank. Blue = PipeDRAM faster, red = SIMDRAM faster; the 1x
level is the crossover (speedup = 1). Dashed lines mark the SIMD width of one
mat (512) and one DRAM row (64K).
"""
import os
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np

from pud_simulator.presets import pipedram_config
from pud_simulator.operations import get_operation
from pud_simulator.simulator import SIMDRAMSimulator
from pud_simulator.pipedram_simulator import PipeDRAMSimulator

# ggplot-like style
plt.rcParams.update({
    "figure.dpi": 150, "figure.facecolor": "white", "font.family": "sans-serif",
    "font.sans-serif": ["Liberation Sans", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 8, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.5,
    "axes.facecolor": "white", "axes.edgecolor": "black", "axes.linewidth": 0.8,
    "xtick.color": "black", "ytick.color": "black",
    "xtick.direction": "out", "ytick.direction": "out",
})

CFG = pipedram_config()
SIM = SIMDRAMSimulator(CFG)
HIER = PipeDRAMSimulator(CFG, "pipelined", "hierarchical")

# Application cost model (workloads/util/bbop_manager.c, used for Fig. 10).
# Per-bbop latency for `size` elements:
#   SIMDRAM    = lat_pass[N] * ceil(size/65536) / SA / banks
#   PipeDRAM-H = (fill[N] + batch[N] * ceil(size/2048)) / SA / banks
# The tables (lat_pass, fill, batch) come from the same simulators (per pass,
# 1 element, 1 SA). The SA/banks divisor cancels in the SIMDRAM-vs-Hier ratio,
# so the heatmap is consistent with Fig. 10 without needing SA/bank counts.
ROW_W = 65536.0                                       # SIMDRAM SIMD width (full row)
MAT_W = 2048.0                                        # PipeDRAM SIMD width (mat*chips)

N_GRID = list(range(2, 41, 2))                        # bit-precision
M_GRID = np.logspace(2, 9, 110)                       # data elements per bbop (size)

# 12 workloads: (name, N_fixed, N_dynamic, M = per-bbop element count `size`).
#  - N_fixed : precision PipeDRAM actually uses (BBOP_BIT_PRECISION=32 all; x264=8).
#  - N_dynamic : per-workload precision if dynamic bit-precision (Proteus) were on.
#  - M : per-bbop `size` (loop dimension passed to bbop_op), from the run data
#        (MIMDRAM_DEPS elems= in reference/results/*_bbop_statistics.csv).
WORKLOADS = [
    ("pca", 32, 8, 16000), ("2mm", 32, 25, 32000), ("3mm", 32, 12, 32000),
    ("cov", 32, 23, 32000), ("dg", 32, 11, 10000), ("fdtd", 32, 13, 10000),
    ("gmm", 32, 24, 32000), ("gs", 32, 13, 32000), ("bp", 32, 13, 134217729),
    ("hw", 32, 17, 10201), ("km", 32, 17, 16384), ("x264", 8, 8, 320),
]


def per_n_params(op, N):
    """Cost-model table entries: SIMDRAM per-pass latency, Hier fill and per-batch
    time (per pass, 1 element, 1 subarray; as stored by tools/generate_tables.py)."""
    u = get_operation(op)(N)
    rs = SIM.simulate(u, 1, 1, 1, transposition_mode="none")
    rh = HIER.simulate(u, 1, 1, 1)
    sim_pass = rs.latency_per_pass_ns
    pd_fill = rh.pipeline_fill_ns or 0.0
    pd_batch = rh.per_batch_time_ns or (rh.latency_per_pass_ns)
    return sim_pass, pd_fill, pd_batch


def cm_ratio(sim_pass, pd_fill, pd_batch, M, smooth=False):
    """Cost-model throughput ratio Hier/SIM = lat_SIM / lat_Hier (SA/banks cancel).
    smooth=True uses continuous rep = max(1, M/width) (no ceil staircase) for a
    readable surface; smooth=False uses the exact ceil() of the cost model."""
    if smooth:
        rep_s = max(1.0, M / ROW_W)
        rep_h = max(1.0, M / MAT_W)
    else:
        rep_s = 1.0 if ROW_W > M else math.ceil(M / ROW_W)
        rep_h = 1.0 if MAT_W > M else math.ceil(M / MAT_W)
    return (sim_pass * rep_s) / (pd_fill + pd_batch * rep_h)


def speedup_grid(op):
    Z = np.zeros((len(M_GRID), len(N_GRID)))
    for j, N in enumerate(N_GRID):
        sp, pf, pb = per_n_params(op, N)
        for i, M in enumerate(M_GRID):
            Z[i, j] = math.log2(cm_ratio(sp, pf, pb, M, smooth=True))
    return Z


# Single panel: multiplication, the dominant PUD operation of 9 of the 12
# real-world applications.
fig, ax = plt.subplots(figsize=(5.0, 2.0))
OP = "multiplication"
X, Y = np.meshgrid(N_GRID, M_GRID)
Z = speedup_grid(OP)
LEVELS = [-3, -2, -1, 0, 1, 2, 3]
pcm = ax.contourf(X, Y, Z, levels=LEVELS, cmap="RdBu", extend="both")

# SIMD-width reference lines: one mat (512 cols) and one DRAM row (64K).
for yv, lab in [(512, "1 mat (512)"), (65536, "1 row (64K)")]:
    ax.axhline(yv, color="black", ls="--", lw=0.7, alpha=0.75)
    ax.text(2.7, yv * 1.3, lab, fontsize=5.5, ha="left", va="bottom")

# Workloads at their fixed precision (BBOP_BIT_PRECISION=32; x264=8) and
# per-bbop element count. Coincident points share one dot + grouped label.
DOTS = {  # label: (N, M)
    "x264": (8, 320),
    "dg, fdtd, hw": (32, 10000),
    "pca, km": (32, 16000),
    "2mm, 3mm, gmm, gs, cov": (32, 32000),
    "bp": (32, 134217729),
}
import matplotlib.patheffects as pe
for lab, (n, m) in DOTS.items():
    ax.scatter(n, m, s=26, c="#F8DB6F", edgecolors="black", linewidths=0.9,
               marker="o", zorder=6)  # SIMDRAM yellow from plot_speedup.py
HALO = [pe.withStroke(linewidth=1.6, foreground="white")]
# isolated points: label adjacent to the dot
for lab, dx, dy, ha in [("x264", 5, -2, "left"), ("bp", -6, -2, "right")]:
    ax.annotate(lab, DOTS[lab], textcoords="offset points", xytext=(dx, dy),
                fontsize=6, fontweight="bold", zorder=7, ha=ha,
                path_effects=HALO)
# clustered points: labels moved into empty map regions, thin leader lines
LEADERS = [  # label, text position (N, M)
    ("2mm, 3mm, gmm, gs, cov", (23.0, 1.2e6)),
    ("pca, km",                (20.0, 1.1e5)),
    ("dg, fdtd, hw",           (24.0, 2.0e3)),
]
for lab, (tx, ty) in LEADERS:
    ax.annotate(lab, DOTS[lab], xytext=(tx, ty), textcoords="data",
                fontsize=6, fontweight="bold", zorder=7,
                ha="center", va="center", path_effects=HALO,
                arrowprops=dict(arrowstyle="-", lw=0.5, color="#888888",
                                shrinkA=2, shrinkB=2.5))

ax.set_yscale("log")
ax.set_xlabel("Bit-Precision")
ax.set_ylabel("Data Elements per PUD Op.")
ax.set_xlim(N_GRID[0], N_GRID[-1])
ax.set_ylim(M_GRID[0], M_GRID[-1])
ax.set_xticks([2, 8, 16, 24, 32, 40])
ax.set_yticks([1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9])
ax.set_yticklabels(["100", "1K", "10K", "100K", "1M", "10M", "100M", "1B"])
ax.grid(False)

cb = fig.colorbar(pcm, ax=ax, fraction=0.05, pad=0.02, ticks=LEVELS)
cb.ax.set_yticklabels(["1/8x", "1/4x", "1/2x", "1x", "2x", "4x", "8x"])
cb.ax.tick_params(labelsize=6)
cb.set_label("PipeDRAM-Hier / SIMDRAM\nThroughput (compute-only)", fontsize=6)

fig.tight_layout(pad=0.3)
ARTIFACT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
outdir = os.environ.get(
    "PIPEDRAM_OUTPUT_DIR",
    os.path.join(ARTIFACT_ROOT, "generated", "figures"),
)
os.makedirs(outdir, exist_ok=True)
out = os.path.join(outdir, "precision_datasize_tradeoff.pdf")
fig.savefig(out, bbox_inches="tight")
fig.savefig(out.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
print(f"wrote {out}")
# Cross-check: the map's Hier/SIM latency ratio at each workload point vs. the
# compute-only Hier/SIM ratio of Fig. 10 (Summary lines of
# reference/results/*_bbop_statistics.csv). Workloads mix several operations,
# so the values are approximate but should agree in sign and magnitude.
FIG10 = {"pca": 0.72, "2mm": 0.89, "3mm": 0.89, "cov": 0.89, "dg": 0.65,
         "fdtd": 0.48, "gmm": 0.89, "gs": 0.89, "bp": 0.71, "hw": 0.58,
         "km": 0.72, "x264": 0.78}
print(f"  {'wl':6s} {'N':>3s} {'M':>11s} {'map lat Hier/SIM':>16s} {'Fig.10':>7s}  "
      f"(cost-model, mult op, fixed precision)")
for name, n_fix, n_dyn, m in WORKLOADS:
    sp, pf, pb = per_n_params("multiplication", n_fix)
    lat_ratio = 1.0 / cm_ratio(sp, pf, pb, m)   # latency ratio, comparable to Fig. 10
    print(f"  {name:6s} {n_fix:3d} {m:11.0f} {lat_ratio:16.2f} {FIG10.get(name, float('nan')):7.2f}")
