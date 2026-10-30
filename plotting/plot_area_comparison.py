#!/usr/bin/env python3
"""
DRAM-die area overhead of PipeDRAM's modifications vs. in-die logic that DRAM
vendors have already standardized or fabricated.

Bars are typed by mechanism class (passive structural / RowHammer counter /
active compute) and provenance-coded (solid = vendor silicon or JEDEC standard;
hatched = self-reported CACTI/Design-Compiler model).

Values and sources (read from area/area_metrics.json):
  in-DRAM MAC compute (Newton/AiM) ~20%  : Newton, MICRO'20 (SK hynix); GDDR6-AiM
                                           ISSCC'22 fabricated. 25% = vendor-
                                           stated acceptable ceiling (Newton).
  DDR5 per-row counters (PRAC)     ~9%   : SK hynix 16Gb DDR5, ISSCC'23 (via MINT,
                                           MICRO'24); PRAC standardized JESD79-5C'24.
  PipeDRAM (LISA+mat+SALP)         1.86% : this work (MIMDRAM 1.11 + LISA 0.6 + SALP 0.15).
  Sectored DRAM                    1.72% : Olgun et al., TACO'24.
  fine-grained mat (MIMDRAM)       1.11% : Oliveira et al., HPCA'24.
"""
import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

plt.rcParams.update({
    "figure.dpi": 150, "figure.facecolor": "white",
    "font.family": "sans-serif",
    "font.sans-serif": ["Liberation Sans", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 8, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6,
    "axes.facecolor": "white", "axes.edgecolor": "black", "axes.linewidth": 0.8,
    "axes.grid": True, "axes.grid.axis": "x", "axes.axisbelow": True,
    "grid.color": "#f0f0f0", "grid.linewidth": 0.6,
    "xtick.color": "black", "ytick.color": "black",
    "xtick.direction": "out", "ytick.direction": "out",
})

COMPUTE = "#e07070"
COUNTER = "#e6a030"
PASSIVE = "#7bafd6"

ARTIFACT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
with open(os.path.join(ARTIFACT_ROOT, "area", "area_metrics.json")) as stream:
    metrics = json.load(stream)

colors = {"active_compute": COMPUTE, "counter": COUNTER, "passive": PASSIVE}
BARS = [
    (
        item["label"].replace("\\n", "\n"),
        item["area_percent"],
        colors[item["mechanism_class"]],
        item["display_value"],
        item["vendor_evidence"],
        item["name"] == "pipedram",
    )
    for item in metrics["comparison"]
]

fig, ax = plt.subplots(figsize=(3.4, 2.15))
ys = list(range(len(BARS)))[::-1]          # first entry on top
for y, (lab, v, col, vt, vendor, ours) in zip(ys, BARS):
    ax.barh(y, v, height=0.62, color=col,
            edgecolor="#10314e" if ours else "black",
            linewidth=1.8 if ours else 0.5,
            hatch=None if vendor else "////", zorder=3)
    ax.text(v + 0.4, y, f"{vt}", va="center", ha="left",
            fontsize=7 if ours else 6.6,
            fontweight="bold", color="#10314e" if ours else "black")

ax.set_yticks(ys)
ax.set_yticklabels([b[0] for b in BARS])
ax.get_yticklabels()[ys.index(2)].set_fontweight("bold")   # PipeDRAM row
ax.set_xlabel("DRAM Area Overhead per Die (%)")
ax.set_xlim(0, 27)
ax.set_ylim(-0.7, len(BARS) - 0.4)

budget = metrics["vendor_compute_budget_percent"]
ax.axvline(budget, ls="--", color="#555", lw=1.0, zorder=2)
ax.text(budget + 0.6, len(BARS) / 2 - 0.5, f"SK hynix AiM Compute Budget ({budget:g}%)",
        fontsize=5.7, color="#555", ha="center", va="center", style="italic",
        rotation=90)

fig.tight_layout(pad=0.3)
outdir = os.environ.get(
    "PIPEDRAM_OUTPUT_DIR",
    os.path.join(ARTIFACT_ROOT, "generated", "figures"),
)
os.makedirs(outdir, exist_ok=True)
out = os.path.join(outdir, "area_comparison.pdf")
fig.savefig(out, bbox_inches="tight")
fig.savefig(out.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
print(f"wrote {out}")
