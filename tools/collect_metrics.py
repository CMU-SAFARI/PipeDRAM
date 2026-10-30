#!/usr/bin/env python3
"""Collect the numerical metrics that validate.py compares with expected_metrics.json."""
import argparse
import csv
import glob
import json
import math
import os
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from pud_simulator.ddg import build_ddg
from pud_simulator.operations import get_operation
from pud_simulator.pipedram_simulator import PipeDRAMSimulator
from pud_simulator.presets import pipedram_config
from pud_simulator.primitives import OpType
from pud_simulator.simulator import SIMDRAMSimulator


def gmean(values):
    values = [value for value in values if value > 0]
    return math.exp(sum(math.log(value) for value in values) / len(values))


def summaries(path):
    result = {}
    with open(path, encoding="utf-8") as stream:
        for row in csv.reader(stream, skipinitialspace=True):
            if len(row) >= 7 and row[0].strip() == "Summary":
                result[row[4].strip()] = {
                    "latency_ms": float(row[5]),
                    "energy_mj": float(row[6]),
                }
    return result


def sensitivity_metrics():
    config = pipedram_config()
    hierarchy = PipeDRAMSimulator(config, "pipelined", "hierarchical")
    simdram = SIMDRAMSimulator(config)
    workload_specs = {
        "pca": (32, 16000, "multiplication"),
        "2mm": (32, 32000, "multiplication"),
        "3mm": (32, 32000, "multiplication"),
        "cov": (32, 32000, "division"),
        "dg": (32, 10000, "multiplication"),
        "fdtd": (32, 10000, "division"),
        "gmm": (32, 32000, "multiplication"),
        "gs": (32, 32000, "multiplication"),
        "bp": (32, 134217729, "multiplication"),
        "hw": (32, 10201, "multiplication"),
        "km": (32, 16384, "multiplication"),
        "x264": (8, 320, "addition"),
    }
    carry = {}
    for name, (precision, elements, operation) in workload_specs.items():
        program = get_operation(operation)(precision)
        result = hierarchy.simulate(program, config.pipedram_simd_lanes, 1, 1)
        positions = build_ddg(program).num_bit_positions
        batches = max(1, math.ceil(elements / config.pipedram_simd_lanes))
        numerator = (
            max(0, positions - 1) * config.t_hier_round_trip
            + batches * config.t_LISA_sensed
        )
        denominator = (
            (result.pipeline_fill_ns or 0)
            + batches * (result.per_batch_time_ns or result.latency_per_pass_ns)
        )
        carry[name] = 100 * numerator / denominator

    crossover = {}
    for precision in (8, 16, 32):
        program = get_operation("multiplication")(precision)
        sim_result = simdram.simulate(program, 1, 1, 1, transposition_mode="none")
        pipe_result = hierarchy.simulate(program, 1, 1, 1)
        sim_pass = sim_result.latency_per_pass_ns
        fill = pipe_result.pipeline_fill_ns or 0
        per_batch = pipe_result.per_batch_time_ns or pipe_result.latency_per_pass_ns
        candidates = np.logspace(2, 9, 20000)
        ratios = (
            sim_pass * np.maximum(1, candidates / (config.row_size_bytes * 8))
            / (fill + per_batch * np.maximum(1, candidates / config.pipedram_simd_lanes))
        )
        index = int(np.argmin(np.abs(np.log2(ratios))))
        crossover[str(precision)] = float(candidates[index])
    mixes = {
        "pca": ({"multiplication": 2, "addition": 4, "division": 2,
                 "subtraction": 4}, 32),
        "2mm": ({"multiplication": 8, "addition": 4}, 32),
        "3mm": ({"multiplication": 6, "addition": 6}, 32),
        "cov": ({"multiplication": 2, "addition": 4, "division": 2,
                 "subtraction": 2}, 32),
        "dg": ({"multiplication": 2, "addition": 2}, 32),
        "fdtd": ({"multiplication": 12, "division": 12, "addition": 4,
                  "subtraction": 4}, 32),
        "gmm": ({"multiplication": 6, "addition": 2}, 32),
        "gs": ({"multiplication": 6, "addition": 4, "division": 2,
                "subtraction": 2}, 32),
        "bp": ({"multiplication": 8, "addition": 6}, 32),
        "hw": ({"multiplication": 2, "addition": 4}, 32),
        "km": ({"multiplication": 2, "addition": 2, "subtraction": 2}, 32),
        "x264": ({"addition": 8}, 8),
    }
    utilization = {}
    for name, (mix, precision) in mixes.items():
        busy = np.zeros(32)
        for operation, count in mix.items():
            program = get_operation(operation)(precision)
            graph = build_ddg(program)
            profile = [
                sum(
                    config.t_AAP if program.primitives[index].op == OpType.AAP
                    else config.t_AP
                    for index in graph.bit_groups[bit]
                )
                for bit in sorted(key for key in graph.bit_groups if key >= 0)
            ]
            for bit, work in enumerate(profile):
                busy[bit % 32] += count * work
        utilization[name] = {
            "mats_used": int(np.count_nonzero(busy)),
            "mean_percent": float(100 * np.mean(busy / np.max(busy))),
        }
    return carry, crossover, utilization


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    workloads = {}
    for path in sorted(glob.glob(os.path.join(args.results, "*_bbop_statistics.csv"))):
        name = os.path.basename(path).removesuffix("_bbop_statistics.csv")
        workloads[name] = summaries(path)

    configs = {
        "simdram_worst": "SIMDRAM_64_WORST",
        "mimdram_worst": "MIMDRAM_64_WORST",
        "proteus_worst": "PROTEUS_PIPE_WORST",
        "pipedram_seq": "PIPEDRAM_SEQ",
        "pipedram_pipe": "PIPEDRAM_PIPE",
        "pipedram_hier": "PIPEDRAM_HIER",
    }
    normalized = {}
    for label, key in configs.items():
        for metric in ("latency_ms", "energy_mj"):
            values = []
            for mechanisms in workloads.values():
                baseline = mechanisms["SIMDRAM_64"][metric]
                values.append(mechanisms[key][metric] / baseline)
            normalized[f"{label}_{metric}_gmean"] = gmean(values)

    overhead = {}
    for path in sorted(glob.glob(os.path.join(args.results, "*_transposition.csv"))):
        name = os.path.basename(path).removesuffix("_transposition.csv")
        with open(path, encoding="utf-8") as stream:
            for row in csv.reader(stream, skipinitialspace=True):
                if len(row) >= 4 and row[0].strip() == "worst" and int(row[1]) == 16:
                    compute = workloads[name]["SIMDRAM_64"]
                    overhead[name] = {
                        "latency_percent": 100 * float(row[2]) /
                        (compute["latency_ms"] + float(row[2])),
                        "energy_percent": 100 * float(row[3]) /
                        (compute["energy_mj"] + float(row[3])),
                    }

    carry, crossover, utilization = sensitivity_metrics()
    metrics = {
        "schema_version": 1,
        "workload_count": len(workloads),
        "x264_frozen_result_present": "x264" in workloads,
        "normalized_geomeans": normalized,
        "figure4_workload_overhead": overhead,
        "figure13_crossover_elements": crossover,
        "figure12_mat_utilization": utilization,
        "figure14_exposed_carry_percent": carry,
    }
    output_directory = os.path.dirname(args.output)
    if output_directory:
        os.makedirs(output_directory, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as stream:
        json.dump(metrics, stream, indent=2, sort_keys=True)
        stream.write("\n")


if __name__ == "__main__":
    main()
