#!/usr/bin/env python3
"""
Exposed carry-propagation overhead vs. number of in-flight batches B (Fig. 14).

The hierarchical carry round-trip is hidden by overlapping work: bit-independent
primitives of the same operation and, more importantly, subsequent data batches.
With B pipeline-width batches, the exposed carry is:

    exposed_frac(B) = (fill_carry + B * lisa_block) / (fill + B * per_batch)

  - fill_carry = (n_bitpos - 1) * round_trip   (full round-trip, one-time fill)
  - lisa_block = the per-batch carry hand-off that blocks the computing subarray

The fraction decays from the B=1 worst case toward the steady-state floor
lisa_block / per_batch.

Bands cover representative operations; the 12 workloads are overlaid at their
actual B = ceil(per-op elements / mat-level SIMD width) and dominant operation.
"""
import os
import math
from functools import lru_cache

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pud_simulator.presets import pipedram_config
from pud_simulator.operations import get_operation
from pud_simulator.ddg import build_ddg
from pud_simulator.pipedram_simulator import PipeDRAMSimulator

# ggplot-like style (same rcParams as plot_combined_scalability.py and
# plot_speedup.py)
plt.rcParams.update({
    "figure.dpi": 150, "figure.facecolor": "white",
    "font.family": "sans-serif",
    "font.sans-serif": ["Liberation Sans", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 8, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.5,
    "axes.facecolor": "white", "axes.edgecolor": "black", "axes.linewidth": 0.8,
    "axes.grid": True, "axes.grid.axis": "y", "axes.axisbelow": True,
    "grid.color": "#f0f0f0", "grid.linestyle": "-", "grid.linewidth": 0.6,
    "xtick.color": "black", "ytick.color": "black",
    "xtick.direction": "out", "ytick.direction": "out",
})

CFG = pipedram_config()
H = PipeDRAMSimulator(CFG, "pipelined", "hierarchical")
RT = CFG.t_hier_round_trip       # full carry round-trip
LO = CFG.t_LISA_sensed           # per-batch hand-off that blocks the computing SA
LANES = CFG.pipedram_simd_lanes  # mat-level SIMD width


@lru_cache(maxsize=None)
def params(op, n):
    u = get_operation(op)(n)
    nbp = build_ddg(u).num_bit_positions
    r = H.simulate(u, LANES, 1, 1)
    fill = r.pipeline_fill_ns or 0.0
    pb = r.per_batch_time_ns or r.latency_per_pass_ns
    return fill, pb, max(0, nbp - 1) * RT


def frac(op, n, B):
    fill, pb, fc = params(op, n)
    return (fc + B * LO) / (fill + B * pb)


B = np.logspace(0, 6, 200)
# Operation classes (same scaling classes as the throughput analysis); each
# class is drawn as a shaded min-max range across its member operations.
# Colors follow plot_combined_scalability.py.
CLASSES = [
    ("linear",      ["addition", "subtraction", "greater", "equal", "min", "max"],
     "#7bafd6"),
    ("logarithmic", ["and_reduction", "or_reduction", "xor_reduction"],
     "#e6a030"),
    ("quadratic",   ["multiplication", "division"],
     "#6ab87a"),
]

# 12 workloads: (name, N, per-op size, dominant op)
WL = {"pca": (32, 16000, "multiplication"), "2mm": (32, 32000, "multiplication"),
      "3mm": (32, 32000, "multiplication"), "cov": (32, 32000, "division"),
      "dg": (32, 10000, "multiplication"), "fdtd": (32, 10000, "division"),
      "gmm": (32, 32000, "multiplication"), "gs": (32, 32000, "multiplication"),
      "bp": (32, 134217729, "multiplication"), "hw": (32, 10201, "multiplication"),
      "km": (32, 16384, "multiplication"), "x264": (8, 320, "addition")}

fig, ax = plt.subplots(figsize=(5.0, 2.0))
for lab, ops, col in CLASSES:
    curves = np.array([[100 * frac(op, 32, b) for b in B] for op in ops])
    lo, hi = curves.min(axis=0), curves.max(axis=0)
    ax.fill_between(B, lo, hi, color=col, alpha=0.30, lw=0, zorder=2)
    ax.plot(B, lo, color=col, lw=1.0, zorder=3)
    ax.plot(B, hi, color=col, lw=1.0, zorder=3)

ax.set_xscale("log")                       # B spans 1..1M; y stays linear (%)
ax.set_xlabel("Number of In-Flight Batches")
ax.set_ylabel("Carry Prop. Overhead (%)")
# small negative margin so near-zero values (quadratic band, bp dot)
# are not clipped by the bottom spine
ax.set_xlim(0.85, 1e6); ax.set_ylim(-2.5, 75)
ax.set_xticks([1, 10, 100, 1e3, 1e4, 1e5, 1e6])
ax.set_xticklabels(["1", "10", "100", "1K", "10K", "100K", "1M"])
ax.set_yticks([0, 10, 20, 30, 40, 50, 60, 70])

# Overlay the 12 workloads (white dots) at their actual B and dominant op;
# coincident dots (same dominant op + same B) are jittered horizontally and
# share one label, connected by thin leader lines.
pts = {}
for name, (n, sz, op) in WL.items():
    b = max(1, math.ceil(sz / LANES))
    pts[name] = (b, 100 * frac(op, n, b))

XJIT = {"dg": 4.4, "hw": 5.9, "pca": 7.3, "km": 8.8,
        "2mm": 12.5, "3mm": 15.0, "gmm": 18.0, "gs": 21.5, "cov": 26.0}
for name, (b, f) in pts.items():
    ax.scatter(XJIT.get(name, b), f, s=14, c="white", edgecolors="black",
               linewidths=0.8, zorder=6)

# direct labels for the isolated points
for name, dx, dy, ha in [("x264", 4, -1, "left"), ("bp", 0, 4, "center"),
                         ("cov", 4, -1, "left")]:
    ax.annotate(name, (XJIT.get(name, pts[name][0]), pts[name][1]),
                textcoords="offset points", xytext=(dx, dy),
                fontsize=6, fontweight="bold", zorder=7, ha=ha)
# leader-line labels for the B=5-16 cluster (text placed in the empty wedge
# between the quadratic and linear bands)
LEADERS = [  # label, text (x, y), anchor dots
    ("fdtd",              (2.2, 11.0), [(5.0,  pts["fdtd"][1])]),
    ("dg, hw",            (4.5, 18.0), [(4.4,  pts["dg"][1]), (5.9, pts["dg"][1])]),
    ("pca, km",           (9.0, 13.0), [(7.3,  pts["pca"][1]), (8.8, pts["pca"][1])]),
    ("2mm, 3mm,\ngmm, gs", (48., 5.5), [(12.5, pts["2mm"][1]), (21.5, pts["2mm"][1])]),
]
for lab, (tx, ty), anchors in LEADERS:
    ax.text(tx, ty, lab, fontsize=5.5, fontweight="bold", ha="center",
            va="bottom", zorder=7, linespacing=1.1)
    for axy in anchors:
        ax.annotate("", xy=axy, xytext=(tx, ty),
                    arrowprops=dict(arrowstyle="-", lw=0.5, color="#999999"),
                    zorder=5)

# legend with black-edged color keys
from matplotlib.colors import to_rgba
from matplotlib.patches import Patch
handles = [Patch(facecolor=to_rgba(col, 0.45), edgecolor="black",
                 linewidth=0.6, label=f"{lab} ops")
           for lab, ops, col in CLASSES]
ax.legend(handles=handles, loc="upper right", frameon=False, borderpad=0.3,
          handlelength=1.6, labelspacing=0.3)

fig.tight_layout(pad=0.3)
ARTIFACT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
outdir = os.environ.get(
    "PIPEDRAM_OUTPUT_DIR",
    os.path.join(ARTIFACT_ROOT, "generated", "figures"),
)
os.makedirs(outdir, exist_ok=True)
out = os.path.join(outdir, "exposed_carry.pdf")
fig.savefig(out, bbox_inches="tight")
fig.savefig(out.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
print(f"wrote {out}")
print("workload exposed carry (dominant op):")
for name, (n, sz, op) in WL.items():
    b = max(1, math.ceil(sz / LANES))
    print(f"  {name:6s} B={b:6d}  {100*frac(op,n,b):5.2f}%  ({op}, N={n})")
