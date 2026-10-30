#!/usr/bin/env python3
"""
Combined scalability plot: (a) bit-serial and (b) bit-parallel normalized throughput.

Single figure with shared legend, Liberation Sans font (ggplot-like style).
"""

import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

from pud_simulator.operations import get_operation
from pud_simulator.simulator import SIMDRAMSimulator
from pud_simulator.pipedram_simulator import PipeDRAMSimulator
from pud_simulator.presets import proteus_config, pipedram_config
from pud_simulator.bit_parallel import (
    SubstrateConfig, OPERATIONS as BP_OPERATIONS, _next_pow2,
)

# Parameters
PRECISIONS = [4, 8, 16, 32]
NUM_ELEMENTS = 67_108_864  # 64M

# (a) Bit-serial: 16 operations, 1 bank, 1 SA
BS_OPERATIONS = [
    "and_reduction", "or_reduction", "xor_reduction", "bitcount",
    "abs", "addition", "subtraction", "equal",
    "greater", "greater_equal", "if_then_else", "max",
    "min", "relu",
    "multiplication", "division",
]

BS_SHORT = {
    "abs": "abs", "addition": "add", "and_reduction": "AND red.",
    "bitcount": "bitcount", "division": "div", "equal": "equal",
    "greater": "greater", "greater_equal": "greater_eq",
    "if_then_else": "if-then-else", "max": "max", "min": "min",
    "multiplication": "mul", "or_reduction": "OR red.",
    "relu": "relu", "subtraction": "sub", "xor_reduction": "XOR red.",
}

BP_OP_LIST = ["addition", "subtraction", "multiplication", "division"]
BP_SHORT = {"addition": "add", "subtraction": "sub",
            "multiplication": "mul", "division": "div"}

# Configurations
cfg = pipedram_config()
cfg_p = proteus_config()

# Bit-serial simulators
BS_CONFIGS = [
    ("SIMDRAM/MIMDRAM",            SIMDRAMSimulator(cfg), "none"),
    ("SIMDRAM/MIMDRAM + transp.",  SIMDRAMSimulator(cfg), "worst"),
    ("PipeDRAM-Seq",    PipeDRAMSimulator(cfg, scheduling_mode="sequential"),                                "none"),
    ("PipeDRAM-Pipe",   PipeDRAMSimulator(cfg, scheduling_mode="pipelined", carry_copy_mode="direct"),       "none"),
    ("PipeDRAM-Hier",   PipeDRAMSimulator(cfg, scheduling_mode="pipelined", carry_copy_mode="hierarchical"), "none"),
]

# Bit-parallel substrates
e_mat = cfg.e_AAP / cfg.num_mats_per_subarray
e_mat_ap = cfg.e_AP / cfg.num_mats_per_subarray
mpc = cfg.num_mats_per_subarray // cfg.num_chips_per_rank
hier_t = cfg.t_hier_round_trip
hier_e = 2 * cfg.e_LISA_mat + e_mat

BP_NUM_BANKS = 1  # single bank for bit-parallel comparison
BP_SUBSTRATES = {
    "Proteus": SubstrateConfig(
        name="Proteus",
        t_hop=cfg_p.t_LISA, e_hop=cfg_p.e_LISA,
        e_aap=cfg_p.e_AAP, e_ap=cfg_p.e_AP,
        max_concurrent=1, simd_width=cfg_p.simd_lanes,
        sa_per_bank=32, mats_per_chip=0,
        num_banks=BP_NUM_BANKS, is_proteus=True,
    ),
    "PipeDRAM-Seq": SubstrateConfig(
        name="PipeDRAM-Seq",
        t_hop=cfg.t_inter_mat_copy, e_hop=e_mat,
        e_aap=e_mat, e_ap=e_mat_ap,
        max_concurrent=1, simd_width=cfg.pipedram_simd_lanes,
        sa_per_bank=32, mats_per_chip=mpc,
        num_banks=BP_NUM_BANKS, is_proteus=False,
    ),
    "PipeDRAM-Pipe": SubstrateConfig(
        name="PipeDRAM-Pipe",
        t_hop=cfg.t_inter_mat_copy, e_hop=e_mat,
        e_aap=e_mat, e_ap=e_mat_ap,
        max_concurrent=0, simd_width=cfg.pipedram_simd_lanes,
        sa_per_bank=32, mats_per_chip=mpc,
        num_banks=BP_NUM_BANKS, is_proteus=False,
    ),
    "PipeDRAM-Hier": SubstrateConfig(
        name="PipeDRAM-Hier",
        t_hop=hier_t, e_hop=hier_e,
        e_aap=e_mat, e_ap=e_mat_ap,
        max_concurrent=0, simd_width=cfg.pipedram_simd_lanes,
        sa_per_bank=32, mats_per_chip=mpc,
        num_banks=BP_NUM_BANKS, is_proteus=False,
    ),
}

# Run simulations
print("Running bit-serial simulations...")
bs_results = {}
for op_name in BS_OPERATIONS:
    bs_results[op_name] = {}
    for label, sim, transp in BS_CONFIGS:
        bs_results[op_name][label] = {}
        for n in PRECISIONS:
            uprog = get_operation(op_name)(n=n)
            r = sim.simulate(uprog, num_elements=NUM_ELEMENTS,
                             num_compute_subarrays=1, num_compute_banks=1,
                             transposition_mode=transp)
            bs_results[op_name][label][n] = r

print("Running bit-parallel simulations...")
bp_results = {}
for op_name in BP_OP_LIST:
    eval_fn = BP_OPERATIONS[op_name]
    bp_results[op_name] = {}
    for label, sub in BP_SUBSTRATES.items():
        bp_results[op_name][label] = {}
        c = cfg_p if sub.is_proteus else cfg
        for n in PRECISIONS:
            r = eval_fn(c, sub, n, NUM_ELEMENTS)
            bp_results[op_name][label][n] = r

# Proteus + worst-case transposition
# Proteus uses the vertical layout, so it uses the same transposition model as
# SIMDRAM. All 32 SAs are used (groups_per_bank x _next_pow2(n) = 32), and each
# SA holds one bit-plane: 32 rows per operand per bank to transpose.
from types import SimpleNamespace
_bp_num_operands = 3  # 2 inputs + 1 output for all 4 ops
_bp_row_xfer_ns = cfg_p.row_size_bytes / cfg_p.bus_bandwidth_GBps
_bp_time_per_row = cfg_p.t_RCD + _bp_row_xfer_ns + cfg_p.t_RP
_bp_per_bank_lat = _bp_num_operands * 32 * _bp_time_per_row
_bp_bus_limited = (_bp_num_operands * 32 * BP_NUM_BANKS
                   * cfg_p.row_size_bytes / cfg_p.bus_bandwidth_GBps)
_bp_transp_per_batch = max(_bp_per_bank_lat, _bp_bus_limited)

for op_name in BP_OP_LIST:
    bp_results[op_name]["Proteus + transp."] = {}
    for n in PRECISIONS:
        r = bp_results[op_name]["Proteus"][n]
        total_transp = r.num_batches * _bp_transp_per_batch
        new_total_lat = r.total_latency_ns + total_transp
        bp_results[op_name]["Proteus + transp."][n] = SimpleNamespace(
            throughput_gops=NUM_ELEMENTS / new_total_lat,
            total_latency_ns=new_total_lat,
        )

# Plotting
plt.rcParams.update({
    "figure.dpi":          150,
    "figure.facecolor":    "white",
    "font.family":         "sans-serif",
    "font.sans-serif":     ["Liberation Sans", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size":           10,
    "axes.titlesize":      11,
    "axes.labelsize":      10,
    "xtick.labelsize":     10,
    "ytick.labelsize":     10,
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

# All line styles (union of both panels)
STYLES = {
    "SIMDRAM/MIMDRAM":            dict(color="#e6a030", marker="o",  ls="-",  linewidth=2.2, markersize=6, markeredgecolor="black", markeredgewidth=0.5),
    "SIMDRAM/MIMDRAM + transp.":  dict(color="#e6a030", marker="o",  ls="--", linewidth=2.2, markersize=6, markeredgecolor="black", markeredgewidth=0.5, zorder=1),
    "Proteus":         dict(color="#6ab87a", marker="s",  ls="-",  linewidth=2.2, markersize=6, markeredgecolor="black", markeredgewidth=0.5, zorder=5),
    "Proteus + transp.":  dict(color="#6ab87a", marker="s",  ls="--", linewidth=2.2, markersize=6, markeredgecolor="black", markeredgewidth=0.5, zorder=1),
    "PipeDRAM-Seq":    dict(color="#e07070", marker="^",  ls="-",  linewidth=2.2, markersize=6, markeredgecolor="black", markeredgewidth=0.5, zorder=4),
    "PipeDRAM-Pipe":   dict(color="#9b7bb8", marker="D",  ls="-",  linewidth=2.2, markersize=5, markeredgecolor="black", markeredgewidth=0.5, zorder=3),
    "PipeDRAM-Hier":   dict(color="#7bafd6", marker="v",  ls="-",  linewidth=2.2, markersize=6, markeredgecolor="black", markeredgewidth=0.5, zorder=6),
}

# Draw orders
BS_DRAW_ORDER = ["SIMDRAM/MIMDRAM + transp.", "SIMDRAM/MIMDRAM", "PipeDRAM-Seq", "PipeDRAM-Pipe", "PipeDRAM-Hier"]
BP_DRAW_ORDER = ["Proteus + transp.", "PipeDRAM-Pipe", "PipeDRAM-Seq", "Proteus", "PipeDRAM-Hier"]

# Legend order: 5 system configs (color+marker) + 2 line types
LEGEND_ORDER = ["SIMDRAM/MIMDRAM", "Proteus", "PipeDRAM-Seq", "PipeDRAM-Pipe", "PipeDRAM-Hier"]

BS_BASELINE = ("SIMDRAM/MIMDRAM", PRECISIONS[0])
BP_BASELINE = ("Proteus", PRECISIONS[0])

ARTIFACT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
outdir = os.environ.get(
    "PIPEDRAM_OUTPUT_DIR",
    os.path.join(ARTIFACT_ROOT, "generated", "figures"),
)
os.makedirs(outdir, exist_ok=True)

# Combined figure: (a) = 4x4 bit-serial, (b) = 1x4 bit-parallel,
# 5 rows x 4 columns with a spacer row between (a) and (b)

fig = plt.figure(figsize=(8.0, 8.0))
gs = gridspec.GridSpec(6, 4, figure=fig,
                       hspace=0.35, wspace=0.3,
                       height_ratios=[1, 1, 1, 1, 0.2, 1])  # row 4 = spacer


def style_ax(ax, op_short, row, nrows_section, is_bottom):
    """Apply common styling to an axis."""
    ax.set_title(op_short, fontweight="bold", fontsize=10, pad=4)
    ax.set_xticks(range(len(PRECISIONS)))
    ax.set_yscale("log")
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(0.8)
    if is_bottom:
        ax.set_xticklabels([str(p) for p in PRECISIONS], color="black")
    else:
        ax.set_xticklabels([])
    ax.set_xlabel("")
    ax.tick_params(axis="y", colors="black")
    return ax


# (a) Bit-serial: rows 0-3
bs_axes = []
for idx, op_name in enumerate(BS_OPERATIONS):
    row, col = idx // 4, idx % 4
    ax = fig.add_subplot(gs[row, col])
    bs_axes.append(ax)

    bl, bp = BS_BASELINE
    base = bs_results[op_name][bl][bp].throughput_gops

    for label in BS_DRAW_ORDER:
        xs = list(range(len(PRECISIONS)))
        ys = [bs_results[op_name][label][n].throughput_gops / base
              for n in PRECISIONS]
        ax.plot(xs, ys, **STYLES[label])

    ax.axhline(y=1.0, color="black", linewidth=0.8, linestyle="-.", alpha=0.6)
    style_ax(ax, BS_SHORT[op_name], row, 4, is_bottom=(row == 3))

# Share y-axis per row for bit-serial; hide non-leftmost y-tick labels
for row_start in range(0, 16, 4):
    ref_ax = bs_axes[row_start]
    for ax in bs_axes[row_start+1:row_start+4]:
        ax.sharey(ref_ax)
        plt.setp(ax.get_yticklabels(), visible=False)

# Extend the last row (min, relu, mul, div) down to its smallest value
last_row_min = float('inf')
for idx in range(12, 16):
    op_name = BS_OPERATIONS[idx]
    bl, bp = BS_BASELINE
    base = bs_results[op_name][bl][bp].throughput_gops
    for label in BS_DRAW_ORDER:
        for n in PRECISIONS:
            val = bs_results[op_name][label][n].throughput_gops / base
            if val < last_row_min:
                last_row_min = val
bs_axes[12].set_ylim(bottom=last_row_min * 0.5)

# (b) Bit-parallel: row 5 (row 4 = spacer)
bp_axes = []
for idx, op_name in enumerate(BP_OP_LIST):
    ax = fig.add_subplot(gs[5, idx])
    bp_axes.append(ax)

    bl, bp = BP_BASELINE
    base = bp_results[op_name][bl][bp].throughput_gops

    for label in BP_DRAW_ORDER:
        xs = list(range(len(PRECISIONS)))
        ys = [bp_results[op_name][label][n].throughput_gops / base
              for n in PRECISIONS]
        ax.plot(xs, ys, **STYLES[label])

    ax.axhline(y=1.0, color="black", linewidth=0.8, linestyle="-.", alpha=0.6)
    style_ax(ax, BP_SHORT[op_name], 0, 1, is_bottom=True)

# Share y-axis for bit-parallel row; hide non-leftmost y-tick labels
for ax in bp_axes[1:]:
    ax.sharey(bp_axes[0])
    plt.setp(ax.get_yticklabels(), visible=False)

# Extend the bit-parallel y-axis down to its smallest value (mul/div)
bp_min = float('inf')
for op_name in BP_OP_LIST:
    bl, bp = BP_BASELINE
    base = bp_results[op_name][bl][bp].throughput_gops
    for label in BP_DRAW_ORDER:
        for n in PRECISIONS:
            val = bp_results[op_name][label][n].throughput_gops / base
            if val < bp_min:
                bp_min = val
bp_axes[0].set_ylim(bottom=bp_min * 0.5)

# Shared legend at top. Legend 1: system colors and markers (top row)
sys_handles = [plt.Line2D([0], [0], **STYLES[l], label=l) for l in LEGEND_ORDER]
leg1 = fig.legend(handles=sys_handles, loc="upper center", ncol=len(LEGEND_ORDER),
                  fontsize=10, frameon=False, bbox_to_anchor=(0.52, 1.035),
                  columnspacing=1.5, handletextpad=0.4, handlelength=1.8)

# Legend 2: line types (second row, below legend 1)
line_handles = [
    plt.Line2D([0], [0], color="black", ls="-",  linewidth=2.2, label="w/o transp."),
    plt.Line2D([0], [0], color="black", ls="--", linewidth=2.2, label="w/ transp."),
]
fig.legend(handles=line_handles, loc="upper center", ncol=2,
           fontsize=10, frameon=False, bbox_to_anchor=(0.52, 1.01),
           columnspacing=2.0, handletextpad=0.4, handlelength=2.2)
fig.add_artist(leg1)  # re-add first legend (second legend call removes it)

fig.subplots_adjust(left=0.08, right=0.98, top=0.95, bottom=0.06,
                    hspace=0.45, wspace=0.3)

# Panel labels, positioned from the axes coordinates
bs_bottom = bs_axes[-1].get_position().y0   # bottom of row 3
bp_top = bp_axes[0].get_position().y1       # top of row 5
bp_bottom = bp_axes[0].get_position().y0    # bottom of row 5
spacer_mid = (bs_bottom + bp_top) / 2

# Below each section: tick labels, then "Bit-Precision", then the panel label
# (a): Bit-Precision above, (a) below
fig.text(0.52, spacer_mid + 0.005, "Bit-Precision",
         fontsize=12, color="black", va="bottom", ha="center")
fig.text(0.52, spacer_mid - 0.005, "(a)",
         fontsize=13, fontweight="bold", va="top", ha="center")

# (b): Bit-Precision above, (b) below
fig.text(0.52, bp_bottom - 0.040, "Bit-Precision",
         fontsize=12, color="black", va="top", ha="center")
fig.text(0.52, bp_bottom - 0.065, "(b)",
         fontsize=13, fontweight="bold", va="top", ha="center")

# Single shared y-axis label
fig.text(0.001, 0.52, "Normalized Throughput (GOPs/s) \u2014 log scale",
         va="center", ha="center", rotation="vertical", fontsize=12, color="black")

path = os.path.join(outdir, "combined_scalability.pdf")
fig.savefig(path, bbox_inches="tight")
plt.close(fig)
print(f"Saved to {path}")
