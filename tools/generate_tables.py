#!/usr/bin/env python3
"""
Generate the PUD cost-model lookup tables in workloads/util/bbop_manager.c
from the Python simulators.

Usage: python tools/generate_tables.py > tables.c

Produces per-pass latency (ns) and energy (nJ) tables for:
  - SIMDRAM    (from SIMDRAMSimulator)
  - Proteus    (from ProteusSimulator, sequential + pipelined)
  - PipeDRAM   (from PipeDRAMSimulator, sequential + pipelined + hierarchical)

Operations: addition, subtraction, multiplication, division, relu

Tables are indexed by bit-precision [0..63], with index 0 = -1 (sentinel).
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from pud_simulator.presets import pipedram_config, proteus_config
from pud_simulator.simulator import SIMDRAMSimulator
from pud_simulator.proteus_simulator import ProteusSimulator
from pud_simulator.pipedram_simulator import PipeDRAMSimulator
from pud_simulator.operations import get_operation


# Operations used in C code and their Python names
OPERATIONS = {
    "adder":      "addition",
    "subtractor": "subtraction",
    "multiplier": "multiplication",
    "divider":    "division",
    "relu":       "relu",
}

BIT_RANGE = range(1, 64)  # bit precisions 1..63


def compute_simdram_tables(cfg):
    """Compute per-pass latency and energy for SIMDRAM."""
    sim = SIMDRAMSimulator(cfg)
    tables = {}
    for c_name, py_name in OPERATIONS.items():
        lat_vals = [-1]
        eng_vals = [-1.0]
        for n in BIT_RANGE:
            uprog = get_operation(py_name)(n=n)
            r = sim.simulate(uprog, num_elements=1, num_compute_subarrays=1, num_compute_banks=1)
            lat_vals.append(round(r.latency_per_pass_ns))
            eng_vals.append(r.energy_per_pass_nj)
        tables[c_name] = {"latency": lat_vals, "energy": eng_vals}
    return tables


def compute_proteus_tables(cfg):
    """Compute per-pass latency/energy for Proteus sequential and pipelined."""
    sim_seq = ProteusSimulator(cfg, scheduling_mode="sequential")
    sim_pipe = ProteusSimulator(cfg, scheduling_mode="pipelined")
    tables = {}
    for c_name, py_name in OPERATIONS.items():
        seq_lat, seq_eng = [-1], [-1.0]
        pipe_lat, pipe_eng = [-1], [-1.0]
        pipe_fill, pipe_batch = [-1], [-1]
        for n in BIT_RANGE:
            uprog = get_operation(py_name)(n=n)
            rs = sim_seq.simulate(uprog, num_elements=1, num_compute_banks=1)
            rp = sim_pipe.simulate(uprog, num_elements=1, num_compute_banks=1)
            seq_lat.append(round(rs.latency_per_pass_ns))
            seq_eng.append(rs.energy_per_pass_nj)
            pipe_lat.append(round(rp.latency_per_pass_ns))
            pipe_eng.append(rp.energy_per_pass_nj)
            pipe_fill.append(round(rp.pipeline_fill_ns))
            pipe_batch.append(round(rp.per_batch_time_ns))
        tables[c_name] = {
            "seq_latency": seq_lat,
            "seq_energy": seq_eng,
            "pipe_latency": pipe_lat,
            "pipe_energy": pipe_eng,
            "pipe_fill": pipe_fill,
            "pipe_batch": pipe_batch,
        }
    return tables


def compute_pipedram_tables(cfg):
    """Compute per-pass latency/energy for PipeDRAM sequential, pipelined, hierarchical."""
    sim_seq = PipeDRAMSimulator(cfg, scheduling_mode="sequential", carry_copy_mode="direct")
    sim_pipe = PipeDRAMSimulator(cfg, scheduling_mode="pipelined", carry_copy_mode="direct")
    sim_hier = PipeDRAMSimulator(cfg, scheduling_mode="pipelined", carry_copy_mode="hierarchical")
    tables = {}
    for c_name, py_name in OPERATIONS.items():
        seq_lat, seq_eng = [-1], [-1.0]
        pipe_lat, pipe_eng = [-1], [-1.0]
        hier_lat, hier_eng = [-1], [-1.0]
        pipe_fill, pipe_batch = [-1], [-1]
        hier_fill, hier_batch = [-1], [-1]
        for n in BIT_RANGE:
            uprog = get_operation(py_name)(n=n)
            rs = sim_seq.simulate(uprog, num_elements=1, num_compute_subarrays=1, num_compute_banks=1)
            rp = sim_pipe.simulate(uprog, num_elements=1, num_compute_subarrays=1, num_compute_banks=1)
            rh = sim_hier.simulate(uprog, num_elements=1, num_compute_subarrays=1, num_compute_banks=1)
            seq_lat.append(round(rs.latency_per_pass_ns))
            seq_eng.append(rs.energy_per_pass_nj)
            pipe_lat.append(round(rp.latency_per_pass_ns))
            pipe_eng.append(rp.energy_per_pass_nj)
            hier_lat.append(round(rh.latency_per_pass_ns))
            hier_eng.append(rh.energy_per_pass_nj)
            pipe_fill.append(round(rp.pipeline_fill_ns))
            pipe_batch.append(round(rp.per_batch_time_ns))
            hier_fill.append(round(rh.pipeline_fill_ns))
            hier_batch.append(round(rh.per_batch_time_ns))
        tables[c_name] = {
            "seq_latency": seq_lat, "seq_energy": seq_eng,
            "pipe_latency": pipe_lat, "pipe_energy": pipe_eng,
            "hier_latency": hier_lat, "hier_energy": hier_eng,
            "pipe_fill": pipe_fill, "pipe_batch": pipe_batch,
            "hier_fill": hier_fill, "hier_batch": hier_batch,
        }
    return tables


def fmt_int_array(name, vals, size=64):
    """Format a C int array."""
    lines = [f"int {name}[{size}] = {{"]
    for i in range(0, len(vals), 8):
        chunk = vals[i:i+8]
        line = ", ".join(f"{v:>10d}" for v in chunk)
        if i + 8 < len(vals):
            line += ","
        lines.append("    " + line)
    lines.append("};")
    return "\n".join(lines)


def fmt_double_array(name, vals, size=64):
    """Format a C double array."""
    lines = [f"double {name}[{size}] = {{"]
    for i in range(0, len(vals), 8):
        chunk = vals[i:i+8]
        formatted = []
        for v in chunk:
            if isinstance(v, int) and v == -1:
                formatted.append(f"{'  -1.0':>10s}")
            else:
                formatted.append(f"{v:10.4f}")
        line = ", ".join(formatted)
        if i + 8 < len(vals):
            line += ","
        lines.append("    " + line)
    lines.append("};")
    return "\n".join(lines)


def main():
    print("Generating tables...", file=sys.stderr)

    pd_cfg = pipedram_config()
    pr_cfg = proteus_config()

    simdram = compute_simdram_tables(pd_cfg)
    proteus = compute_proteus_tables(pr_cfg)
    pipedram = compute_pipedram_tables(pd_cfg)

    print("// " + "=" * 70)
    print("//  Lookup tables generated from the Python PUD simulators")
    print("//  Generated by: tools/generate_tables.py")
    print("//  Config: DDR4-2400, 128 mats/subarray, 4 chips, 32 banks")
    print("//         t_AAP=49ns, t_AP=42ns, e_AAP=e_AP=0.871nJ")
    print("//         LISA: t=142ns, e=57.6nJ (DDR4-2400 scaled)")
    print("//         HFF: t_reloc=0.57ns, hff_width=4 bits")
    print("// " + "=" * 70)
    print()

    # SIMDRAM tables
    print("// " + "-" * 70)
    print("//  SIMDRAM: per-pass latency (ns) and energy (nJ)")
    print("//  One subarray, full row = 65,536 SIMD lanes")
    print("// " + "-" * 70)
    print()
    for op in OPERATIONS:
        print(fmt_int_array(f"latency_simdram_{op}", simdram[op]["latency"]))
        print()
    for op in OPERATIONS:
        print(fmt_double_array(f"energy_simdram_{op}", simdram[op]["energy"]))
        print()

    # Proteus tables
    print("// " + "-" * 70)
    print("//  Proteus: per-pass latency (ns) and energy (nJ)")
    print("//  Bits distributed across subarrays, LISA carry (142ns)")
    print("//  Full row = 65,536 SIMD lanes")
    print("// " + "-" * 70)
    print()
    print("// Proteus Sequential (latency, ns)")
    for op in OPERATIONS:
        print(fmt_int_array(f"latency_proteus_sequential_{op}", proteus[op]["seq_latency"]))
        print()

    print("// Proteus Pipelined (latency, ns)")
    for op in OPERATIONS:
        print(fmt_int_array(f"latency_proteus_pipelined_{op}", proteus[op]["pipe_latency"]))
        print()

    print("// Proteus Pipelined (pipeline scheduling arrays)")
    for op in OPERATIONS:
        print(fmt_int_array(f"pipeline_fill_proteus_pipelined_{op}", proteus[op]["pipe_fill"]))
        print()
        print(fmt_int_array(f"per_batch_proteus_pipelined_{op}", proteus[op]["pipe_batch"]))
        print()

    print("// Proteus energy (nJ per pass)")
    for op in OPERATIONS:
        print(fmt_double_array(f"energy_proteus_sequential_{op}", proteus[op]["seq_energy"]))
        print()
    for op in OPERATIONS:
        print(fmt_double_array(f"energy_proteus_pipelined_{op}", proteus[op]["pipe_energy"]))
        print()

    # PipeDRAM tables
    print("// " + "-" * 70)
    print("//  PipeDRAM: per-pass latency (ns) and energy (nJ)")
    print("//  MABM layout, mat-level SIMD = 2,048 lanes")
    print("//  Energy at mat granularity (1/128 of full row)")
    print("// " + "-" * 70)
    print()
    for mode in ["seq", "pipe", "hier"]:
        label = {"seq": "Sequential", "pipe": "Pipelined", "hier": "Hierarchical"}[mode]
        print(f"// PipeDRAM {label} (latency, ns)")
        for op in OPERATIONS:
            print(fmt_int_array(f"latency_pipedram_{label.lower()}_{op}", pipedram[op][f"{mode}_latency"]))
            print()

    for mode in ["pipe", "hier"]:
        label = {"pipe": "Pipelined", "hier": "Hierarchical"}[mode]
        print(f"// PipeDRAM {label} (pipeline scheduling arrays)")
        for op in OPERATIONS:
            print(fmt_int_array(f"pipeline_fill_pipedram_{label.lower()}_{op}", pipedram[op][f"{mode}_fill"]))
            print()
            print(fmt_int_array(f"per_batch_pipedram_{label.lower()}_{op}", pipedram[op][f"{mode}_batch"]))
            print()

    print("// PipeDRAM energy arrays (nJ per pass)")
    for mode in ["seq", "hier"]:
        label = {"seq": "Sequential", "hier": "Hierarchical"}[mode]
        for op in OPERATIONS:
            print(fmt_double_array(f"energy_pipedram_{label.lower()}_{op}", pipedram[op][f"{mode}_energy"]))
            print()

    print("Done.", file=sys.stderr)


if __name__ == "__main__":
    main()
