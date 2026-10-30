"""
Post-processing and plot generation for PUD simulation results.
===============================================================

Generates scalability plots:
  - Latency vs bit-precision
  - Throughput vs bit-precision
  - Energy vs bit-precision
  - Energy efficiency vs bit-precision

Supports side-by-side comparison of multiple configurations
(e.g., SIMDRAM vs PipeDRAM, 1-bank vs 16-bank).

Usage (standalone)::

    python -m pud_simulator.postprocess

Usage (as library)::

    from pud_simulator.postprocess import ScalabilityStudy
    study = ScalabilityStudy(config, precisions=[8,16,32,64])
    study.run()
    study.plot_all("output_dir/")
"""

import json
import os
import math
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from .config import DRAMConfig
from .primitives import UProgram
from .operations import get_operation, list_operations
from .simulator import SIMDRAMSimulator, SimResult


# Plot style

# Consistent colors and markers for operation classes
OP_STYLES = {
    # Arithmetic
    "addition":       {"color": "#1f77b4", "marker": "o",  "ls": "-"},
    "subtraction":    {"color": "#1f77b4", "marker": "s",  "ls": "--"},
    "multiplication": {"color": "#d62728", "marker": "D",  "ls": "-"},
    "division":       {"color": "#d62728", "marker": "^",  "ls": "--"},
    # Comparison
    "greater":        {"color": "#2ca02c", "marker": "o",  "ls": "-"},
    "greater_equal":  {"color": "#2ca02c", "marker": "s",  "ls": "--"},
    "equal":          {"color": "#2ca02c", "marker": "D",  "ls": "-."},
    "min":            {"color": "#ff7f0e", "marker": "o",  "ls": "-"},
    "max":            {"color": "#ff7f0e", "marker": "s",  "ls": "--"},
    # Reduction
    "and_reduction":  {"color": "#9467bd", "marker": "o",  "ls": "-"},
    "or_reduction":   {"color": "#9467bd", "marker": "s",  "ls": "--"},
    "xor_reduction":  {"color": "#9467bd", "marker": "D",  "ls": "-."},
    # Other
    "abs":            {"color": "#8c564b", "marker": "o",  "ls": "-"},
    "relu":           {"color": "#8c564b", "marker": "s",  "ls": "--"},
    "if_then_else":   {"color": "#e377c2", "marker": "o",  "ls": "-"},
    "bitweave":       {"color": "#7f7f7f", "marker": "o",  "ls": "-"},
    "bitcount":       {"color": "#bcbd22", "marker": "o",  "ls": "-"},
    "greater_equal_0":{"color": "#17becf", "marker": "o",  "ls": "-"},
    "greater_than_0": {"color": "#17becf", "marker": "s",  "ls": "--"},
}

DEFAULT_STYLE = {"color": "#333333", "marker": "x", "ls": "-"}

# Operation groups for multi-panel plots
OP_GROUPS = {
    "Arithmetic":  ["addition", "subtraction", "multiplication", "division"],
    "Comparison":  ["greater", "greater_equal", "equal", "min", "max"],
    "Reduction":   ["and_reduction", "or_reduction", "xor_reduction"],
    "Other":       ["abs", "relu", "if_then_else", "bitweave", "bitcount"],
}


def _style(op_name: str) -> dict:
    return OP_STYLES.get(op_name, DEFAULT_STYLE)


def _setup_plot_style():
    """Apply consistent plot styling."""
    plt.rcParams.update({
        "figure.dpi":        150,
        "figure.facecolor":  "white",
        "axes.grid":         True,
        "axes.axisbelow":    True,
        "grid.alpha":        0.3,
        "grid.linestyle":    "--",
        "font.size":         10,
        "axes.titlesize":    12,
        "axes.labelsize":    11,
        "legend.fontsize":   8,
        "legend.framealpha": 0.9,
        "lines.linewidth":   1.5,
        "lines.markersize":  5,
    })


# Scalability Study

@dataclass
class RunConfig:
    """A single simulation configuration to sweep over."""
    label:           str
    config:          DRAMConfig
    num_elements:    int = 67_108_864   # 64M elements
    compute_banks:   int = 1
    compute_subarrays: int = 1
    transposition:   str = "none"
    simulator_class: type = None        # default -> SIMDRAMSimulator
    simulator_kwargs: dict = field(default_factory=dict)


@dataclass
class ScalabilityStudy:
    """Run and plot scalability analysis across bit-precisions.

    Parameters
    ----------
    run_configs : list[RunConfig]
        One or more system configurations to compare.
    operations : list[str] or None
        Operations to evaluate. None = all.
    precisions : list[int]
        Bit-precisions to sweep.
    """
    run_configs: List[RunConfig]
    operations:  Optional[List[str]] = None
    precisions:  List[int] = field(default_factory=lambda: [8, 16, 32, 64])

    # Collected results: config_label -> op_name -> precision -> SimResult
    results: Dict[str, Dict[str, Dict[int, SimResult]]] = field(
        default_factory=dict, init=False)

    def __post_init__(self):
        if self.operations is None:
            self.operations = [name for name, _ in list_operations()]

    def run(self, verbose: bool = True):
        """Execute all simulations."""
        total = len(self.run_configs) * len(self.operations) * len(self.precisions)
        count = 0

        for rc in self.run_configs:
            self.results[rc.label] = {}
            SimClass = rc.simulator_class or SIMDRAMSimulator
            sim = SimClass(rc.config, **rc.simulator_kwargs)

            for op_name in self.operations:
                self.results[rc.label][op_name] = {}
                gen_fn = get_operation(op_name)

                for n in self.precisions:
                    uprog = gen_fn(n)
                    result = sim.simulate(
                        uprog,
                        num_elements        = rc.num_elements,
                        num_compute_banks   = rc.compute_banks,
                        num_compute_subarrays = rc.compute_subarrays,
                        transposition_mode  = rc.transposition,
                    )
                    self.results[rc.label][op_name][n] = result
                    count += 1

            if verbose:
                print(f"  [{rc.label}] {count}/{total} simulations complete")

        if verbose:
            print(f"  Done: {count} simulations total.\n")

    # Data extraction helpers

    def _get_series(self, config_label: str, op_name: str,
                    metric: str) -> Tuple[List[int], List[float]]:
        """Extract (precisions, values) for a given metric."""
        data = self.results[config_label][op_name]
        xs, ys = [], []
        for n in sorted(data.keys()):
            xs.append(n)
            ys.append(getattr(data[n], metric))
        return xs, ys

    # Individual metric plots

    def plot_metric(
        self,
        metric:       str,
        ylabel:       str,
        title:        str,
        filepath:     str,
        log_y:        bool = False,
        ops:          Optional[List[str]] = None,
        config_label: Optional[str] = None,
        figsize:      Tuple[float, float] = (8, 5),
    ):
        """Plot one metric for selected operations under one config."""
        _setup_plot_style()
        fig, ax = plt.subplots(figsize=figsize)

        label = config_label or self.run_configs[0].label
        ops_to_plot = ops or self.operations

        for op_name in ops_to_plot:
            if op_name not in self.results[label]:
                continue
            xs, ys = self._get_series(label, op_name, metric)
            st = _style(op_name)
            ax.plot(xs, ys, label=op_name,
                    color=st["color"], marker=st["marker"],
                    linestyle=st["ls"])

        ax.set_xlabel("Bit Precision (n)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_xticks(self.precisions)
        if log_y:
            ax.set_yscale("log")
        ax.legend(loc="best", ncol=2)
        fig.tight_layout()
        fig.savefig(filepath, bbox_inches="tight")
        plt.close(fig)

    def plot_metric_comparison(
        self,
        metric:   str,
        ylabel:   str,
        title:    str,
        filepath: str,
        ops:      Optional[List[str]] = None,
        log_y:    bool = False,
        figsize:  Tuple[float, float] = (8, 5),
    ):
        """Plot one metric comparing multiple configs for selected ops."""
        _setup_plot_style()
        fig, ax = plt.subplots(figsize=figsize)

        ops_to_plot = ops or self.operations
        configs = [rc.label for rc in self.run_configs]

        line_styles = ["-", "--", "-.", ":"]

        for ci, clabel in enumerate(configs):
            ls = line_styles[ci % len(line_styles)]
            for op_name in ops_to_plot:
                if op_name not in self.results[clabel]:
                    continue
                xs, ys = self._get_series(clabel, op_name, metric)
                st = _style(op_name)
                display_label = f"{op_name} ({clabel})"
                ax.plot(xs, ys, label=display_label,
                        color=st["color"], marker=st["marker"],
                        linestyle=ls)

        ax.set_xlabel("Bit Precision (n)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_xticks(self.precisions)
        if log_y:
            ax.set_yscale("log")
        ax.legend(loc="best", ncol=2, fontsize=7)
        fig.tight_layout()
        fig.savefig(filepath, bbox_inches="tight")
        plt.close(fig)

    # Grouped panel plots

    def plot_grouped_panels(
        self,
        metric:       str,
        ylabel:       str,
        suptitle:     str,
        filepath:     str,
        config_label: Optional[str] = None,
        log_y:        bool = False,
        figsize:      Tuple[float, float] = (14, 8),
    ):
        """Create a 2x2 panel plot grouped by operation class."""
        _setup_plot_style()
        label = config_label or self.run_configs[0].label
        groups = OP_GROUPS

        fig, axes = plt.subplots(2, 2, figsize=figsize)
        axes_flat = axes.flatten()

        for idx, (group_name, group_ops) in enumerate(groups.items()):
            ax = axes_flat[idx]
            for op_name in group_ops:
                if op_name not in self.results[label]:
                    continue
                xs, ys = self._get_series(label, op_name, metric)
                st = _style(op_name)
                ax.plot(xs, ys, label=op_name,
                        color=st["color"], marker=st["marker"],
                        linestyle=st["ls"])

            ax.set_xlabel("Bit Precision (n)")
            ax.set_ylabel(ylabel)
            ax.set_title(group_name)
            ax.set_xticks(self.precisions)
            if log_y:
                ax.set_yscale("log")
            ax.legend(loc="best", fontsize=7)

        fig.suptitle(suptitle, fontsize=13, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        fig.savefig(filepath, bbox_inches="tight")
        plt.close(fig)

    # Transposition breakdown bar charts

    def plot_transposition_breakdown(
        self,
        filepath:     str,
        config_labels: Optional[List[str]] = None,
        precision:    int = 32,
        metric_pair:  Tuple[str, str] = ("compute_latency_ns",
                                         "transposition_latency_ns"),
        ylabel:       str = "Latency (ns)",
        suptitle:     str = "Compute vs Transposition Overhead",
        log_y:        bool = True,
        figsize:      Tuple[float, float] = (14, 8),
    ):
        """2x2 panel stacked bar chart: compute vs transposition per operation.

        Parameters
        ----------
        config_labels : list[str]
            Config labels that include transposition (uses their SimResult's
            compute and transposition fields directly).
        precision : int
            Which bit-precision to plot.
        metric_pair : tuple[str, str]
            (compute_metric, transposition_metric) attribute names on SimResult.
        """
        import numpy as np
        _setup_plot_style()

        labels = config_labels or [rc.label for rc in self.run_configs]
        compute_attr, transp_attr = metric_pair

        fig, axes = plt.subplots(2, 2, figsize=figsize)
        axes_flat = axes.flatten()

        bar_width = 0.8 / len(labels)
        compute_color = "#4878A8"
        transp_color  = "#E07050"

        for idx, (group_name, group_ops) in enumerate(OP_GROUPS.items()):
            ax = axes_flat[idx]
            # Filter to ops that exist in results
            ops = [op for op in group_ops
                   if any(op in self.results.get(l, {}) for l in labels)]
            if not ops:
                ax.set_visible(False)
                continue

            x = np.arange(len(ops))

            for ci, clabel in enumerate(labels):
                comp_vals, transp_vals = [], []
                for op_name in ops:
                    r = self.results.get(clabel, {}).get(op_name, {}).get(precision)
                    if r:
                        comp_vals.append(getattr(r, compute_attr))
                        transp_vals.append(getattr(r, transp_attr))
                    else:
                        comp_vals.append(0)
                        transp_vals.append(0)

                offset = (ci - (len(labels) - 1) / 2) * bar_width
                short_label = clabel  # used in legend
                ax.bar(x + offset, comp_vals, bar_width,
                       label=f"Compute ({short_label})" if idx == 0 else None,
                       color=compute_color, alpha=0.7 + 0.3 * (ci == 0),
                       edgecolor="white", linewidth=0.5,
                       hatch="" if ci == 0 else "//")
                ax.bar(x + offset, transp_vals, bar_width,
                       bottom=comp_vals,
                       label=f"Transposition ({short_label})" if idx == 0 else None,
                       color=transp_color, alpha=0.7 + 0.3 * (ci == 0),
                       edgecolor="white", linewidth=0.5,
                       hatch="" if ci == 0 else "//")

            ax.set_xticks(x)
            ax.set_xticklabels(ops, rotation=30, ha="right", fontsize=8)
            ax.set_ylabel(ylabel)
            ax.set_title(group_name)
            if log_y:
                ax.set_yscale("log")

        # Single shared legend from first panel
        handles, leg_labels = axes_flat[0].get_legend_handles_labels()
        fig.legend(handles, leg_labels, loc="lower center",
                   ncol=len(labels), fontsize=9,
                   bbox_to_anchor=(0.5, -0.02))

        fig.suptitle(f"{suptitle}  (n={precision})",
                     fontsize=13, fontweight="bold")
        fig.tight_layout(rect=[0, 0.05, 1, 0.96])
        fig.savefig(filepath, bbox_inches="tight")
        plt.close(fig)

    # Convenience: generate all standard plots

    def plot_all(self, output_dir: str, config_label: Optional[str] = None):
        """Generate the full set of scalability plots."""
        os.makedirs(output_dir, exist_ok=True)
        label = config_label or self.run_configs[0].label
        import re
        tag = re.sub(r"[^\w\-]", "_", label).strip("_").lower()

        plots = [
            ("total_latency_ns",     "Latency (ns)",       "Latency vs Bit Precision",           True),
            ("throughput_gops",      "Throughput (GOPS/s)", "Throughput vs Bit Precision",         True),
            ("total_energy_nj",      "Energy (nJ)",         "Energy vs Bit Precision",             True),
            ("energy_eff_gops_per_w","Efficiency (GOPS/W)", "Energy Efficiency vs Bit Precision",  True),
            ("total_primitives",     "# Primitives",        "Primitive Count vs Bit Precision",    True),
        ]

        for metric, ylabel, title, log_y in plots:
            # Single plot with all ops
            fname = f"{tag}_{metric}.pdf"
            self.plot_metric(metric, ylabel, f"{title} [{label}]",
                             os.path.join(output_dir, fname),
                             log_y=log_y, config_label=label)

            # Grouped 2x2 panel
            fname_panel = f"{tag}_{metric}_panels.pdf"
            self.plot_grouped_panels(metric, ylabel, f"{title} [{label}]",
                                     os.path.join(output_dir, fname_panel),
                                     config_label=label, log_y=log_y)

        # Comparison across configs (if multiple)
        if len(self.run_configs) > 1:
            for metric, ylabel, title, log_y in plots:
                # Only plot linear-class ops for clarity
                linear_ops = ["addition", "subtraction", "greater", "equal",
                              "min", "abs", "relu", "if_then_else"]
                fname = f"comparison_{metric}_linear.pdf"
                self.plot_metric_comparison(
                    metric, ylabel, f"{title} — Config Comparison",
                    os.path.join(output_dir, fname),
                    ops=linear_ops, log_y=log_y)

                quad_ops = ["multiplication", "division"]
                fname = f"comparison_{metric}_quadratic.pdf"
                self.plot_metric_comparison(
                    metric, ylabel, f"{title} — Config Comparison (Quadratic)",
                    os.path.join(output_dir, fname),
                    ops=quad_ops, log_y=log_y)

        print(f"  Plots saved to {output_dir}/")

    # Export to JSON/CSV

    def to_json(self, filepath: str):
        """Export all results to JSON."""
        out = {}
        for clabel, ops in self.results.items():
            out[clabel] = {}
            for op_name, precs in ops.items():
                out[clabel][op_name] = {}
                for n, r in precs.items():
                    d = asdict(r)
                    d.pop("trace", None)
                    out[clabel][op_name][str(n)] = d
        with open(filepath, "w") as f:
            json.dump(out, f, indent=2)
        print(f"  Results exported to {filepath}")

    def to_csv(self, filepath: str):
        """Export all results to CSV."""
        fields = [
            "config", "operation", "bit_precision",
            "total_aaps", "total_aps", "total_primitives",
            "latency_per_pass_ns", "total_latency_ns",
            "throughput_gops", "total_energy_nj", "energy_eff_gops_per_w",
        ]
        with open(filepath, "w") as f:
            f.write(",".join(fields) + "\n")
            for clabel, ops in self.results.items():
                for op_name, precs in ops.items():
                    for n in sorted(precs.keys()):
                        r = precs[n]
                        vals = [
                            clabel, op_name, str(n),
                            str(r.total_aaps), str(r.total_aps), str(r.total_primitives),
                            f"{r.latency_per_pass_ns:.2f}", f"{r.total_latency_ns:.2f}",
                            f"{r.throughput_gops:.4f}", f"{r.total_energy_nj:.4f}",
                            f"{r.energy_eff_gops_per_w:.4f}",
                        ]
                        f.write(",".join(vals) + "\n")
        print(f"  Results exported to {filepath}")


# Standalone entry point

def main():
    """Run the default scalability study with PipeDRAM config."""
    from .presets import pipedram_config

    cfg = pipedram_config()

    # Without transposition
    study = ScalabilityStudy(
        run_configs=[
            RunConfig("1-bank", cfg, compute_banks=1),
            RunConfig("16-bank", cfg, compute_banks=16),
        ],
        precisions=[8, 16, 32, 64],
    )

    study.run()
    study.plot_all("plots/", config_label="1-bank")
    study.plot_all("plots/", config_label="16-bank")
    study.to_csv("plots/scalability_results.csv")
    study.to_json("plots/scalability_results.json")

    # With transposition overhead
    study_transp = ScalabilityStudy(
        run_configs=[
            RunConfig("1-bank (no transp.)", cfg,
                      compute_banks=1, transposition="none"),
            RunConfig("1-bank (w/ transp.)", cfg,
                      compute_banks=1, transposition="worst"),
            RunConfig("16-bank (no transp.)", cfg,
                      compute_banks=16, transposition="none"),
            RunConfig("16-bank (w/ transp.)", cfg,
                      compute_banks=16, transposition="worst"),
        ],
        precisions=[8, 16, 32, 64],
    )

    study_transp.run()
    study_transp.plot_all("plots/transposition/",
                          config_label="1-bank (w/ transp.)")
    study_transp.plot_all("plots/transposition/",
                          config_label="16-bank (w/ transp.)")
    study_transp.to_csv("plots/transposition/scalability_transp.csv")
    study_transp.to_json("plots/transposition/scalability_transp.json")

    # Breakdown bar charts
    transp_labels = ["1-bank (w/ transp.)", "16-bank (w/ transp.)"]
    for n in [8, 16, 32, 64]:
        study_transp.plot_transposition_breakdown(
            f"plots/transposition/breakdown_latency_n{n}.pdf",
            config_labels=transp_labels, precision=n,
            metric_pair=("compute_latency_ns", "transposition_latency_ns"),
            ylabel="Latency (ns)",
            suptitle="Compute vs Transposition Latency",
        )
        study_transp.plot_transposition_breakdown(
            f"plots/transposition/breakdown_energy_n{n}.pdf",
            config_labels=transp_labels, precision=n,
            metric_pair=("compute_energy_nj", "transposition_energy_nj"),
            ylabel="Energy (nJ)",
            suptitle="Compute vs Transposition Energy",
        )


if __name__ == "__main__":
    main()
