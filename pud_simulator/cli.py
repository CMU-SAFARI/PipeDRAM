#!/usr/bin/env python3
"""
Command-line interface for the PUD simulator.
"""

import argparse
import json
import sys
from dataclasses import asdict

from .config import DRAMConfig
from .primitives import UProgram
from .operations import get_operation, list_operations
from .simulator import SIMDRAMSimulator, TRANSP_NONE
from .pipedram_simulator import PipeDRAMSimulator
from .presets import pipedram_config
from .printing import print_config, print_results


def build_parser():
    p = argparse.ArgumentParser(
        description="PUD Performance Simulator (SIMDRAM / PipeDRAM)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # operation selection
    p.add_argument("--operation", "-op", type=str, default=None,
                   help="Operation name (e.g., addition, division)")
    p.add_argument("--list-operations", action="store_true",
                   help="List all built-in operations and exit")

    # workload
    p.add_argument("--bit-precision", "-n", type=int, default=32,
                   help="Operand bit width (N)")
    p.add_argument("--num-elements", "-m", type=int, default=65536,
                   help="Number of data elements (M)")

    # parallelism
    p.add_argument("--compute-subarrays", type=int, default=1,
                   help="Subarrays in parallel per bank (SALP)")
    p.add_argument("--compute-banks", type=int, default=1,
                   help="Banks in parallel (BLP)")

    # simulator
    p.add_argument("--simulator", choices=["simdram", "pipedram"],
                   default="simdram",
                   help="Simulator: simdram or pipedram")
    p.add_argument("--scheduling", choices=["sequential", "pipelined"],
                   default="pipelined",
                   help="PipeDRAM scheduling mode (ignored for simdram)")
    p.add_argument("--carry-copy", choices=["direct", "hierarchical"],
                   default="direct",
                   help="PipeDRAM carry propagation: inter-mat copy in the "
                        "computing subarray, or hierarchical LISA + inter-mat "
                        "(ignored for simdram)")

    # transposition
    p.add_argument("--transposition", choices=["none", "worst", "common"],
                   default="none")
    p.add_argument("--cache-resident-fraction", type=float, default=0.5)

    # DRAM config
    p.add_argument("--config", type=str, default=None,
                   help="JSON DRAM config file (default: the paper's "
                        "configuration, presets.pipedram_config())")
    p.add_argument("--row-size", type=int, default=None)
    p.add_argument("--num-mats", type=int, default=None)
    p.add_argument("--num-subarrays", type=int, default=None)
    p.add_argument("--num-banks", type=int, default=None)
    p.add_argument("--num-chips", type=int, default=None)
    p.add_argument("--t-aap", type=float, default=None)
    p.add_argument("--t-ap", type=float, default=None)
    p.add_argument("--e-aap", type=float, default=None)
    p.add_argument("--e-ap", type=float, default=None)

    # output
    p.add_argument("--json-out", type=str, default=None)
    p.add_argument("--show-config", action="store_true")
    p.add_argument("--show-trace", action="store_true",
                   help="Print per-primitive execution trace")
    p.add_argument("--quiet", "-q", action="store_true")

    return p


def main():
    parser = build_parser()
    args = parser.parse_args()

    # list operations
    if args.list_operations:
        print(f"\n  {'Name':<22} {'Description'}")
        print("  " + "─" * 60)
        for name, doc in list_operations():
            desc = (doc or "").split("\n")[0].strip()
            print(f"  {name:<22} {desc}")
        print()
        return

    if not args.operation:
        parser.error("--operation is required (or use --list-operations)")

    # DRAM config
    cfg = DRAMConfig.from_json(args.config) if args.config else pipedram_config()
    if args.row_size      is not None: cfg.row_size_bytes          = args.row_size
    if args.num_mats      is not None: cfg.num_mats_per_subarray   = args.num_mats
    if args.num_subarrays is not None: cfg.num_subarrays_per_bank  = args.num_subarrays
    if args.num_banks     is not None: cfg.num_banks_per_rank      = args.num_banks
    if args.num_chips     is not None: cfg.num_chips_per_rank      = args.num_chips
    if args.t_aap         is not None: cfg.t_AAP                   = args.t_aap
    if args.t_ap          is not None: cfg.t_AP                    = args.t_ap
    if args.e_aap         is not None: cfg.e_AAP                   = args.e_aap
    if args.e_ap          is not None: cfg.e_AP                    = args.e_ap

    if args.show_config and not args.quiet:
        print_config(cfg)

    # generate uProgram
    gen_fn = get_operation(args.operation)
    uprog = gen_fn(args.bit_precision)

    if not args.quiet:
        print(f"\n  µProgram: {uprog.summary()}")

    # simulate
    if args.simulator == "pipedram":
        sim = PipeDRAMSimulator(cfg, scheduling_mode=args.scheduling,
                                carry_copy_mode=args.carry_copy)
        mode = f"PipeDRAM Mode ({args.scheduling}, {args.carry_copy} carry)"
    else:
        sim = SIMDRAMSimulator(cfg)
        mode = "SIMDRAM Mode"
    result = sim.simulate(
        uprog                   = uprog,
        num_elements            = args.num_elements,
        num_compute_subarrays   = args.compute_subarrays,
        num_compute_banks       = args.compute_banks,
        transposition_mode      = args.transposition,
        cache_resident_fraction = args.cache_resident_fraction,
        enable_trace            = args.show_trace,
    )

    if not args.quiet:
        print_results(result, uprog, cfg, mode)

    if args.show_trace and result.trace:
        print(f"  {'#':<5} {'Op':<5} {'Dst':<16} {'Src':<16} {'Bit':<5} "
              f"{'Phase':<14} {'Cum.Lat(ns)':<12}")
        print("  " + "─" * 80)
        for t in result.trace:
            print(f"  {t['idx']:<5} {t['op']:<5} {t['dst']:<16} {t['src']:<16} "
                  f"{t['bit_idx']:<5} {t['phase']:<14} {t['cumulative_lat_ns']:<12.2f}")
        print()

    if args.json_out:
        out = asdict(result)
        out.pop("trace", None)  # trace can be huge
        with open(args.json_out, "w") as f:
            json.dump(out, f, indent=2)
        if not args.quiet:
            print(f"  Results written to {args.json_out}\n")
