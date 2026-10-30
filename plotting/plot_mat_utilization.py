#!/usr/bin/env python3
"""
Utilization breakdown across mats and chips for the 12 real-world
applications (Fig. 12).

MABM localizes each scalar to one chip (i mod 4) and maps bit j of that
scalar to mat (j mod 32) within that chip. Every chip therefore runs the
same 32-mat pipeline on its quarter of the elements.

Left heatmap : per-mat utilization (busy time / busiest mat) for each
               application, across the 32 mats of a chip (identical for
               all four chips). Computed from each application's
               instrumented bbop mix (counts from
               reference/results/*_bbop_statistics.csv) and the
               per-bit-position work of each uProgram at the fixed 32-bit
               precision (x264: 8-bit), with bit-positions >= 32 folded
               onto mat (j mod 32). The fold balances multiplication:
               its symmetric 62-position tent folds to a flat profile
               (mean/peak 0.52 -> 1.00). Division (exactly 32 positions,
               no wrap) keeps its algorithmic ramp (0.61).
Right heatmap: per-chip utilization = the average busy-fraction over that
               chip's 32 mats. Every chip runs the identical pipeline on an
               equal quarter of the elements, so the four columns match
               (a final partial batch skews element counts by <= 1, i.e.
               <= 0.04%, worst case heartwall 2551 vs 2550).
"""
import os
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pud_simulator.presets import pipedram_config
from pud_simulator.operations import get_operation
from pud_simulator.ddg import build_ddg
from pud_simulator.primitives import OpType

# ggplot-like style (same rcParams as plot_speedup.py and
# plot_combined_scalability.py)
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

CFG = pipedram_config()
NUM_MATS = 32           # mats per chip (a scalar's bits stay in its chip)
NUM_CHIPS = 4

BBOP_TO_OP = {"BBOP_MUL": "multiplication", "BBOP_MUL_32": "multiplication",
              "BBOP_ADD": "addition", "BBOP_SUB": "subtraction",
              "BBOP_DIV": "division", "BBOP_REDUCE": "addition",
              "BBOP_CPY": None}

# (bbop counts from *_bbop_statistics.csv PIPEDRAM_HIER rows,
#  per-op element count M, bit-precision N)
WL = {
    "pca":  ({"BBOP_MUL": 2, "BBOP_REDUCE": 4, "BBOP_DIV": 2, "BBOP_SUB": 4},
             16000, 32),
    "2mm":  ({"BBOP_MUL": 8, "BBOP_REDUCE": 4}, 32000, 32),
    "3mm":  ({"BBOP_MUL": 6, "BBOP_REDUCE": 6}, 32000, 32),
    "cov":  ({"BBOP_MUL": 2, "BBOP_REDUCE": 4, "BBOP_DIV": 2, "BBOP_SUB": 2},
             32000, 32),
    "dg":   ({"BBOP_MUL_32": 2, "BBOP_REDUCE": 2, "BBOP_CPY": 2}, 10000, 32),
    "fdtd": ({"BBOP_MUL": 12, "BBOP_DIV": 12, "BBOP_ADD": 4, "BBOP_SUB": 4},
             10000, 32),
    "gmm":  ({"BBOP_MUL": 6, "BBOP_REDUCE": 2}, 32000, 32),
    "gs":   ({"BBOP_MUL": 6, "BBOP_REDUCE": 4, "BBOP_DIV": 2, "BBOP_SUB": 2},
             32000, 32),
    "bp":   ({"BBOP_MUL": 8, "BBOP_REDUCE": 2, "BBOP_ADD": 4, "BBOP_CPY": 2},
             134217729, 32),
    "hw":   ({"BBOP_MUL": 2, "BBOP_REDUCE": 4}, 10201, 32),
    "km":   ({"BBOP_MUL": 2, "BBOP_REDUCE": 2, "BBOP_SUB": 2}, 16384, 32),
    "x264": ({"BBOP_ADD": 8}, 320, 8),
}

_prof_cache = {}


def per_mat_work(op, n):
    """Per-bit-position (= per-mat) busy time (ns) of one uProgram pass."""
    if (op, n) not in _prof_cache:
        u = get_operation(op)(n)
        d = build_ddg(u)
        _prof_cache[(op, n)] = [
            sum(CFG.t_AAP if u.primitives[i].op == OpType.AAP else CFG.t_AP
                for i in d.bit_groups[b])
            for b in sorted(k for k in d.bit_groups if k >= 0)]
    return _prof_cache[(op, n)]


mat_util = np.zeros((len(WL), NUM_MATS))
chip_util = np.zeros((len(WL), NUM_CHIPS))
for r, (name, (mix, m, n)) in enumerate(WL.items()):
    busy = np.zeros(NUM_MATS)
    for bbop, cnt in mix.items():
        op = BBOP_TO_OP[bbop]
        if op is None:
            continue
        for j, w in enumerate(per_mat_work(op, n)):
            busy[j % NUM_MATS] += cnt * w       # MABM fold: bit j -> mat j mod 32
    mat_util[r] = 100.0 * busy / busy.max()
    per_chip = [(m + NUM_CHIPS - 1 - c) // NUM_CHIPS for c in range(NUM_CHIPS)]
    occ = mat_util[r].mean()        # chip utilization = busy fraction of its 32 mats
    chip_util[r] = [occ * c / max(per_chip) for c in per_chip]

# Colormap built from the palette blue (#7bafd6, as in plot_speedup.py)
from matplotlib.colors import LinearSegmentedColormap
PAPER_BLUES = LinearSegmentedColormap.from_list(
    "paper_blues", ["#ffffff", "#7bafd6", "#2c5d8f"])

fig, (axM, axC) = plt.subplots(
    1, 2, figsize=(5.0, 2.0),
    gridspec_kw={"width_ratios": [4.2, 1.0], "wspace": 0.08})

# No gridlines on the heatmaps (the y-grid would draw over the imshow cells)
axM.grid(False)
axC.grid(False)

names = list(WL.keys())
im = axM.imshow(mat_util, aspect="auto", cmap=PAPER_BLUES, vmin=0, vmax=100,
                interpolation="nearest")
axM.set_xlabel("Mat within Chip")
axM.set_xticks([0, 8, 16, 24, 31])
axM.set_yticks(range(len(names)))
axM.set_yticklabels(names)
axM.tick_params(length=2)
for y in np.arange(0.5, len(names) - 0.5):
    axM.axhline(y, color="white", lw=0.4)
    axC.axhline(y, color="white", lw=0.4)

axC.imshow(chip_util, aspect="auto", cmap=PAPER_BLUES, vmin=0, vmax=100,
           interpolation="nearest")
axC.set_xlabel("Chip")
axC.set_xticks(range(NUM_CHIPS))
axC.set_yticks([])
axC.tick_params(length=2)

cb = fig.colorbar(im, ax=[axM, axC], fraction=0.05, pad=0.03)
cb.set_label("Utilization (%)", fontsize=7)
cb.ax.tick_params(labelsize=6.5)

ARTIFACT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
outdir = os.environ.get(
    "PIPEDRAM_OUTPUT_DIR",
    os.path.join(ARTIFACT_ROOT, "generated", "figures"),
)
os.makedirs(outdir, exist_ok=True)
out = os.path.join(outdir, "mat_utilization.pdf")
fig.savefig(out, bbox_inches="tight")
fig.savefig(out.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
print(f"wrote {out}")
for r, name in enumerate(names):
    used = int((mat_util[r] > 0).sum())
    print(f"  {name:5s} mats used = {used:2d}/32, mean util (all 32 mats) = "
          f"{mat_util[r].mean():5.1f}%, min = {mat_util[r].min():5.1f}%, "
          f"chip util = {chip_util[r].mean():6.2f}% (min {chip_util[r].min():6.2f}%)")
