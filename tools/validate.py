#!/usr/bin/env python3
"""Validate generated metrics, figures, and (full mode) regenerated CSVs against the references."""
import argparse
import csv
import json
import math
import os
import pathlib
import subprocess
import tempfile

import matplotlib.image as mpimg
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
FIGURES = {
    "area_comparison": ["DRAM Area Overhead", "PipeDRAM"],
    "transposition_reuse_combined": ["Operational Intensity", "Performance"],
    "combined_scalability": ["Normalized Throughput", "Bit-Precision"],
    "workload_normalized_combined": ["Normalized Exec", "Normalized Energy"],
    "energy_efficiency_cpu_normalized": ["Performance/Watt"],
    "precision_datasize_tradeoff": ["Bit-Precision", "Data Elements"],
    "mat_utilization": ["Mat within Chip", "Utilization"],
    "exposed_carry": ["Carry Prop", "In-Flight Batches"],
}


def close(expected, actual, rel=1e-9, absolute=1e-9):
    return math.isclose(expected, actual, rel_tol=rel, abs_tol=absolute)


def compare_tree(expected, actual, prefix=""):
    errors = []
    if isinstance(expected, dict):
        if set(expected) != set(actual):
            return [f"{prefix}: keys differ"]
        for key in expected:
            errors.extend(compare_tree(expected[key], actual[key], f"{prefix}.{key}"))
    elif isinstance(expected, bool) or isinstance(expected, int):
        if expected != actual:
            errors.append(f"{prefix}: expected {expected}, got {actual}")
    elif isinstance(expected, float):
        if not close(expected, actual):
            errors.append(f"{prefix}: expected {expected}, got {actual}")
    elif expected != actual:
        errors.append(f"{prefix}: expected {expected!r}, got {actual!r}")
    return errors


def raster(pdf, output):
    subprocess.run(
        ["pdftoppm", "-f", "1", "-singlefile", "-r", "150", "-png", str(pdf), str(output)],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    return pathlib.Path(f"{output}.png")


def numeric_rows(path):
    rows = []
    with open(path, encoding="utf-8") as stream:
        for row in csv.reader(stream):
            converted = []
            for field in row:
                try:
                    converted.append(float(field.strip()))
                except ValueError:
                    converted.append(field.strip())
            rows.append(converted)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("quick", "full"), required=True)
    args = parser.parse_args()
    errors = []

    expected = json.loads((ROOT / "expected_metrics.json").read_text())
    actual = json.loads((ROOT / "generated/metrics.json").read_text())
    errors.extend(compare_tree(expected, actual, "metrics"))
    area = json.loads((ROOT / "generated/area_metrics.json").read_text())
    if area.get("mabm_logic_cells") != 0:
        errors.append("area: MABM did not synthesize to zero logic cells")
    if area.get("round_trip_test") != "passed":
        errors.append("area: MABM round-trip test did not pass")
    if not close(area.get("dram_component_sum_percent", -1), 1.86):
        errors.append("area: DRAM component sum is not 1.86%")
    if area.get("schedule_bytes") != 5400:
        errors.append("area: largest schedule is not 5,400 bytes")

    for name, labels in FIGURES.items():
        pdf = ROOT / "generated/figures" / f"{name}.pdf"
        if not pdf.exists() or pdf.stat().st_size < 5000:
            errors.append(f"{name}: missing or implausibly small PDF")
            continue
        info = subprocess.run(["pdfinfo", str(pdf)], check=True, text=True,
                              capture_output=True).stdout
        if "Pages:           1" not in info:
            errors.append(f"{name}: PDF is not one page")
        text = subprocess.run(["pdftotext", str(pdf), "-"], check=True, text=True,
                              capture_output=True).stdout
        for label in labels:
            if label not in text:
                errors.append(f"{name}: expected label {label!r} missing")
        with tempfile.TemporaryDirectory() as directory:
            generated_png = raster(pdf, pathlib.Path(directory) / "generated")
            reference_png = ROOT / "reference/renderings" / f"{name}.png"
            generated = mpimg.imread(generated_png)
            reference = mpimg.imread(reference_png)
            if generated.shape != reference.shape:
                errors.append(
                    f"{name}: rendering dimensions differ "
                    f"{generated.shape} != {reference.shape}"
                )
            else:
                mae = float(np.mean(np.abs(generated.astype(float) - reference.astype(float))))
                if mae > 0.025:
                    errors.append(f"{name}: rendering MAE {mae:.5f} exceeds 0.025")

    if args.mode == "full":
        frozen = {"heartwall", "x264"}
        for reference in (ROOT / "reference/results").glob("*.csv"):
            name = reference.name.split("_", 1)[0]
            generated = ROOT / "generated/results" / reference.name
            if not generated.exists():
                errors.append(f"full results missing {reference.name}")
                continue
            if name in frozen:
                if generated.read_bytes() != reference.read_bytes():
                    errors.append(f"{reference.name}: frozen result changed")
                continue
            left, right = numeric_rows(reference), numeric_rows(generated)
            if len(left) != len(right):
                errors.append(f"{reference.name}: row count differs")
                continue
            for row_number, (a, b) in enumerate(zip(left, right), 1):
                if len(a) != len(b):
                    errors.append(f"{reference.name}:{row_number}: column count differs")
                    break
                for column, (x, y) in enumerate(zip(a, b), 1):
                    if isinstance(x, float) and isinstance(y, float):
                        if not close(x, y, rel=1e-6, absolute=1e-6):
                            errors.append(
                                f"{reference.name}:{row_number}:{column}: {x} != {y}"
                            )
                    elif x != y:
                        errors.append(
                            f"{reference.name}:{row_number}:{column}: {x!r} != {y!r}"
                        )

    if errors:
        print("\n".join(errors))
        raise SystemExit(1)
    print("Numerical, PDF, label, and rendering validation passed")


if __name__ == "__main__":
    main()
