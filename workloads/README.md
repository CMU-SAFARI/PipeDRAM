# Instrumented applications

This directory contains the real-world applications used in the PipeDRAM
evaluation. Each application marks its data-parallel loops as bulk bitwise
PUD operations (bbops) and reports their latency and energy on several PUD
systems through an analytical cost model. The instrumentation infrastructure
derives from Proteus (ICS 2025, https://github.com/CMU-SAFARI/proteus).

| Directory      | Suite      | Kernel                                                |
| -------------- | ---------- | ----------------------------------------------------- |
| `2mm/`         | PolyBench  | Two chained matrix multiplications                    |
| `3mm/`         | PolyBench  | Three matrix multiplications, `(A*B)*(C*D)`           |
| `covariance/`  | PolyBench  | Covariance matrix                                     |
| `doitgen/`     | PolyBench  | Multi-resolution analysis kernel (MADNESS)            |
| `fdtd-apml/`   | PolyBench  | FDTD with anisotropic perfectly matched layer         |
| `gemm/`        | PolyBench  | General matrix multiply, `C = alpha*A*B + beta*C`     |
| `gramschmidt/` | PolyBench  | Gram-Schmidt decomposition                            |
| `backprop/`    | Rodinia    | Neural network back-propagation                       |
| `heartwall/`   | Rodinia    | Heart wall tracking on ultrasound video               |
| `kmeans/`      | Rodinia    | K-means clustering                                    |
| `pca/`         | Phoenix    | Mean vector and covariance matrix (first step of PCA) |

The twelfth application of the paper, x264, comes from SPEC CPU2017. Its
source and inputs cannot be redistributed; only its frozen results are
included under `reference/results/`.

## Cost model

`util/bbop_manager.{c,h}` implements the cost model. Every `bbop_op()` call
looks up per-pass latency and energy tables generated from the Python
simulators in `pud_simulator/` (`tools/generate_tables.py`) and scales them by
the number of SIMD batches, subarrays, and banks. The model reports:

- SIMDRAM with one subarray (`SIMDRAM_1`), with `SUBARRAYS` subarrays per bank
  (`SIMDRAM_64`; `SUBARRAYS` is 32 in `bbop_manager.h`), and with dynamic
  bit-precision (`SIMDRAM_64_DYNAMIC`).
- MIMDRAM (`MIMDRAM_1`, `MIMDRAM_64`, `MIMDRAM_64_DYNAMIC`), derived from the
  SIMDRAM costs by overlapping independent bbops on different DRAM mats.
- Proteus with sequential and pipelined execution (`PROTEUS_SEQ`,
  `PROTEUS_PIPE`).
- PipeDRAM with MABM only (`PIPEDRAM_SEQ`), with the modulo scheduler
  (`PIPEDRAM_PIPE`), and with the scheduler and hierarchical in-DRAM data copy
  (`PIPEDRAM_HIER`).
- The same systems at 16-bit precision (`*_16` rows).

Except for the `*_DYNAMIC` rows, all systems use a fixed bit-precision
(`BBOP_BIT_PRECISION` in `bbop_manager.h`: 32 bits; x264 uses 8 bits).

Arrays registered with `bbop_register_input()` and `bbop_register_output()`
cross the kernel boundary. Systems with a vertical data layout must transpose
them between the horizontal and vertical layouts; PipeDRAM does not.

## Outputs

A PUD build writes the following files to its working directory:

- `bbop_statistics.csv`: one row per bbop and system (latency in ms, energy in
  mJ), followed by `Summary` rows with the totals per system. The `*_WORST`,
  `*_COMMON`, and `*_IDEAL` summary rows add worst-case, common-case
  (cache-resident inputs), and zero transposition cost to the vertical-layout
  systems. Lines starting with `#` describe the MIMDRAM dependency analysis.
- `transposition.csv`: transposition latency and energy per mode (`worst`,
  `common`, `ideal`) and number of banks.
- `bbop_resolution.csv`: histogram of the bit-precision required by the bbop
  inputs.

## Build and run

Each directory has a `Makefile` (`makefile` for heartwall) and a `README` with
the exact arguments. The targets are:

| Executable           | Description                                                                  |
| -------------------- | ---------------------------------------------------------------------------- |
| `<app>_baseline.exe` | CPU baseline                                                                 |
| `<app>_pim.exe`      | PUD-instrumented version; also executes every bbop on the CPU                |
| `<app>_pim_fast.exe` | PUD-instrumented version without CPU execution of bbops (`-DBBOP_SKIP_COMPUTE`) |
| `<app>_pim_int4.exe` | PUD-instrumented version with 4-bit data (2mm, 3mm, and gemm only)           |

The PolyBench kernels are built with the paper-scale dataset
(`-DPAPER_DATASET`); the other dataset sizes are defined in each kernel's
header file.

`scripts/run_workloads.sh` builds and runs every `*_pim_fast.exe` with the
paper-scale arguments and collects the CSV files:

| Application | Command                                         |
| ----------- | ----------------------------------------------- |
| PolyBench   | `./<app>_pim_fast.exe`                          |
| backprop    | `./backprop_pim_fast.exe 134217728`             |
| pca         | `./pca_pim_fast.exe -r 16000 -c 16000`          |
| kmeans      | `./kmeans_pim_fast.exe -n 1 -i 10000_16384.txt` |
| heartwall   | `./heartwall_pim_fast.exe test_4k.avi 20 8`     |

The kmeans input is generated from `kmeans/datagen.cpp` by
`scripts/generate_kmeans_input.sh`. The heartwall input video is not
redistributed; `docs/DATASETS.md` lists its checksum.
