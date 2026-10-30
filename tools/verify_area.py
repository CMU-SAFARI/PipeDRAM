#!/usr/bin/env python3
"""Check the area arithmetic and the MABM RTL (Yosys cell count, Icarus round trip)."""
import json
import math
import pathlib
import re
import shutil
import subprocess

root = pathlib.Path(__file__).resolve().parents[1]
metrics = json.loads((root / "area/area_metrics.json").read_text())
components = metrics["dram_components_percent"]
component_total = sum(components.values())
assert math.isclose(component_total, metrics["dram_total_percent"], abs_tol=1e-12)

schedule_bytes = (
    metrics["largest_schedule_entries"] * metrics["bytes_per_schedule_entry"]
)
assert math.isclose(
    schedule_bytes / 1024,
    metrics["largest_schedule_kib"],
    abs_tol=1e-12,
)
assert schedule_bytes <= metrics["schedule_scratchpad_kib"] * 1024

rtl = root / "area/rtl"
results = {
    "dram_component_sum_percent": component_total,
    "schedule_bytes": schedule_bytes,
    "mabm_logic_cells": None,
    "round_trip_test": None,
}

if shutil.which("yosys"):
    command = (
        "read_verilog mabm_permute.v mabm_inverse.v mabm_top.v; "
        "hierarchy -check -top mabm_top; proc; flatten; opt; stat"
    )
    stat = subprocess.run(
        ["yosys", "-p", command],
        cwd=rtl,
        check=True,
        text=True,
        capture_output=True,
    ).stdout
    match = re.search(r"Number of cells:\s+(\d+)", stat)
    if not match:
        raise RuntimeError("could not parse Yosys cell count")
    results["mabm_logic_cells"] = int(match.group(1))
    assert results["mabm_logic_cells"] == metrics["mabm_expected_logic_cells"]
else:
    raise RuntimeError("yosys is required for MABM area verification")

if shutil.which("iverilog") and shutil.which("vvp"):
    executable = root / "generated/mabm_round_trip.vvp"
    subprocess.run(
        [
            "iverilog", "-o", executable, "tb_mabm.v", "mabm_top.v",
            "mabm_permute.v", "mabm_inverse.v",
        ],
        cwd=rtl,
        check=True,
    )
    output = subprocess.run(
        ["vvp", executable], cwd=rtl, check=True, text=True, capture_output=True
    ).stdout
    if "ALL TESTS PASSED" not in output:
        raise RuntimeError("MABM round-trip verification failed")
    results["round_trip_test"] = "passed"
else:
    raise RuntimeError("iverilog and vvp are required for MABM verification")

output = root / "generated/area_metrics.json"
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
print(json.dumps(results, indent=2, sort_keys=True))
