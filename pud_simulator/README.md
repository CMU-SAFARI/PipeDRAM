# PUD Simulator

Analytical performance and energy simulator for Processing-Using-DRAM (PUD)
architectures. It models:

- **SIMDRAM**: bit-serial SIMD execution over a vertical data layout.
- **Proteus**: bit-level parallelism across subarrays with LISA carry
  propagation, over a vertical data layout.
- **PipeDRAM**: pipelined bit-serial and bit-parallel execution over a
  horizontal data layout, with sequential, pipelined, and hierarchical-carry
  execution modes.

The simulator walks each uProgram primitive (AAP/AP) individually and keeps
the operand-level information needed to build the data dependency graph
(DDG) and the modulo schedule used by PipeDRAM.

## Features

- **19 operations** traced from the [MIMDRAM](https://github.com/CMU-SAFARI/MIMDRAM)
  microworkload C sources.
- **Per-primitive execution**: no closed-form shortcuts; every AAP/AP carries
  its read and write sets.
- **DDG construction**: each primitive records `bit_idx`, `dst`, `src`,
  `write_rows`, and `read_rows`, which classify it as bit-dependent (carry
  chain) or bit-independent.
- **Configurable DRAM organization**: row size, mats, chips, subarrays,
  banks, timing, and energy.
- **Transposition model**: none, worst-case, and common-case
  horizontal-to-vertical transposition for the vertical-layout baselines.
- **Bit-parallel arithmetic**: a Simplified Kogge-Stone (SKS) majority-logic
  adder simulator for addition, subtraction, multiplication, and division.
- **Scalability studies**: sweeps over bit-precision and parallelism with
  CSV/JSON export and plots.

## Package structure

```
pud_simulator/
    __init__.py              Package entry point
    __main__.py              python -m pud_simulator
    cli.py                   Command-line interface
    config.py                DRAMConfig dataclass (organization, timing, energy)
    presets.py               PipeDRAM, Proteus, and SIMDRAM parameter sets
    primitives.py            Primitive, OpType, UProgram, read/write sets
    ddg.py                   DDG construction, bit-dependent/independent classification
    simulator.py             SIMDRAMSimulator and transposition model
    pipedram_simulator.py    PipeDRAMSimulator (sequential, pipelined, hierarchical carry)
    proteus_simulator.py     ProteusSimulator (sequential, pipelined)
    sks_simulator.py         SKS bit-parallel adder: uProgram builder, scheduler, functional check
    sks_integration.py       SKS addition/subtraction cost queries
    bit_parallel.py          Bit-parallel evaluation: Proteus vs. PipeDRAM substrates
    postprocess.py           Scalability studies, plots, CSV/JSON export
    printing.py              Result and configuration printing
    operations/
        __init__.py          Registry and @register decorator
        arithmetic.py        addition, subtraction, multiplication, division
        comparison.py        greater, greater_equal, equal, min, max
        reduction.py         and/or/xor_reduction, greater_equal_0, greater_than_0
        other.py             abs, relu, if_then_else, bitweave, bitcount
```

## Command-line usage

The package depends only on the Python packages in the repository's
`requirements.txt` (Matplotlib is needed only for `postprocess.py`).

```bash
# List the built-in operations
python -m pud_simulator --list-operations

# 32-bit addition over 64M elements on SIMDRAM
python -m pud_simulator -op addition -n 32 -m 67108864

# Same operation on PipeDRAM with pipelined scheduling
python -m pud_simulator -op addition -n 32 -m 67108864 --simulator pipedram --scheduling pipelined

# PipeDRAM with hierarchical (LISA + inter-mat) carry propagation, 32 subarrays
python -m pud_simulator -op multiplication -n 32 -m 67108864 --simulator pipedram \
    --carry-copy hierarchical --compute-subarrays 32

# 16-bit multiplication using 16 banks
python -m pud_simulator -op multiplication -n 16 -m 67108864 --compute-banks 16

# Per-primitive execution trace (use a small precision for readability)
python -m pud_simulator -op addition -n 4 -m 65536 --show-trace

# Export results to JSON
python -m pud_simulator -op division -n 32 -m 67108864 --json-out results.json
```

The CLI uses the paper's configuration (`presets.pipedram_config()`, see
below). A different configuration can be loaded from a JSON file with
`--config`:

```bash
python -c "from pud_simulator.presets import pipedram_config; print(pipedram_config().to_json())" > my_config.json
python -m pud_simulator -op addition -n 32 --config my_config.json
```

Individual parameters can be overridden with `--row-size`, `--num-mats`,
`--num-subarrays`, `--num-banks`, `--num-chips`, `--t-aap`, `--t-ap`,
`--e-aap`, and `--e-ap`.

## Library usage

```python
from pud_simulator.operations import get_operation
from pud_simulator.presets import pipedram_config
from pud_simulator.simulator import SIMDRAMSimulator
from pud_simulator.pipedram_simulator import PipeDRAMSimulator

cfg = pipedram_config()

# Generate a uProgram
uprog = get_operation("addition")(32)
print(uprog.summary())  # Addition (n=32): 193 AAP + 64 AP = 257 primitives

# Inspect primitives
for p in uprog.primitives[:5]:
    print(p, p.write_rows, p.read_rows)

# SIMDRAM, 32 subarrays, worst-case transposition
simdram = SIMDRAMSimulator(cfg)
r = simdram.simulate(uprog, num_elements=67_108_864,
                     num_compute_subarrays=32, transposition_mode="worst")

# PipeDRAM with modulo scheduling and hierarchical (LISA) carry propagation
pipedram = PipeDRAMSimulator(cfg, scheduling_mode="pipelined",
                             carry_copy_mode="hierarchical")
r = pipedram.simulate(uprog, num_elements=67_108_864, num_compute_subarrays=16)
print(f"{r.throughput_gops:.2f} GOPS/s, {r.total_energy_nj:.2f} nJ")
```

A scalability study over bit-precisions:

```python
from pud_simulator.postprocess import ScalabilityStudy, RunConfig
from pud_simulator.presets import pipedram_config

cfg = pipedram_config()
study = ScalabilityStudy(
    run_configs=[
        RunConfig("1-bank",  cfg, compute_banks=1),
        RunConfig("16-bank", cfg, compute_banks=16),
    ],
    precisions=[8, 16, 32, 64],
)
study.run()
study.plot_all("plots/", config_label="1-bank")
study.to_csv("plots/results.csv")
```

## PipeDRAM evaluation parameters

`presets.pipedram_config()`:

| Parameter | Value | Notes |
|---|---|---|
| Row size | 8 KB (65,536 bits) | SIMD lanes per subarray (SIMDRAM) |
| Mat size | 512 bits | Columns per DRAM mat |
| Mats/subarray | 128 | 8 KB / 512 bits |
| Mats/chip | 32 | |
| Chips/rank | 4 | PipeDRAM SIMD width = 512 x 4 = 2,048 |
| Subarrays/bank | 32 | |
| Banks | 32 | |
| Rows/subarray | 1024 | |
| t_AAP | 49 ns | Row copy |
| t_AP | 42 ns | Majority (TRA) |
| t_RAS | 32 ns | Row activation |
| t_RP | 15 ns | Precharge |
| t_LISA | 142 ns | LISA 1-hop, DDR4-2400 |
| e_AAP, e_AP | 0.871 nJ | Full-row activation |

## Supported operations

| Operation | Primitives | Class | MIMDRAM source |
|---|---|---|---|
| addition | 8n+1 | Linear | `00_addition-plus.c` |
| subtraction | 8n+1 | Linear | `01_subtraction-plus.c` |
| min / max | 10n+2 | Linear | `02_min-plus.c` / `03_max-plus.c` |
| greater / greater_equal | 3n+2 | Linear | `40_greater-plus.c` / `41_greater_equal-plus.c` |
| equal | 4n+3 | Linear | `42_equal-plus.c` |
| if_then_else | 7n | Linear | `60_if_then_else-plus.c` |
| abs | 10n-1 | Linear | `50_abs-plus.c` |
| relu | 3n+((n-1)%2) | Linear | `51_relu-plus.c` |
| and/or_reduction | 5*floor(n/2)+2 | Linear (N-input) | `31_and_reduction-plus.c` |
| xor_reduction | 6*floor(n/2)+2 | Linear (N-input) | `30_xor_reduction-plus.c` |
| multiplication | 11n^2-5n-1 | Quadratic | `11_multu-plus.c` |
| division | ~8n^2+12n | Quadratic | `16_divu-plus.c` |
| bitcount | ~8n | Recursive | `91_bitcount-plus.c` |
| bitweave | 4n+4 | Linear | `90_bitweave-plus.c` |
| greater_equal_0 | 2 | Constant | `35_greater_equal_0-plus.c` |
| greater_than_0 | 5*floor((n-1)/2)+4 | Linear | `36_greater_than_0-plus.c` |

Primitive counts follow the loop structure of the MIMDRAM microworkload C
sources rather than closed-form approximations.

## Execution model

**SIMDRAM.** All primitives execute sequentially within a subarray.
Parallelism comes from SIMD lanes (one 65,536-bit row), subarrays (SALP), and
banks (BLP). Each batch holds `SIMD_lanes x subarrays x banks` elements, and
the uProgram repeats `ceil(M / batch_size)` times for M elements.

**PipeDRAM.** MABM maps bit j of an element to mat (j mod 32) of one chip, so
the SIMD width per subarray is 512 columns x 4 chips = 2,048 elements and
energy scales with mat-level activation (1/128 of a row). In pipelined mode,
the DDG splits each bit-position's primitives into bit-dependent (carry
chain) and bit-independent work; the initiation interval is
`II = max(bit-dependent latency) + carry transfer`. The pipeline fill is paid
once and later batches stream through the mats. With hierarchical carry, the
carry moves through a neighboring copying subarray using LISA and the
inter-mat interconnect while staying in the sense amplifiers.

## References

- SIMDRAM: Hajinazar et al., "SIMDRAM: A Framework for Bit-Serial SIMD
  Processing Using DRAM," ASPLOS 2021.
- MIMDRAM: Oliveira et al., "MIMDRAM: An End-to-End Processing-Using-DRAM
  System for High-Throughput, Energy-Efficient and Programmer-Transparent
  Multiple-Instruction Multiple-Data Computing," HPCA 2024.
- Proteus: Oliveira et al., "Proteus: Achieving High-Performance
  Processing-Using-DRAM via Dynamic Precision Bit-Serial Arithmetic,"
  ICS 2025.
- PipeDRAM: Oliveira et al., "PipeDRAM: A Data-Transposition-Free
  Processing-Using-DRAM Architecture with Hardware/Software Pipelining,"
  MICRO 2026.
