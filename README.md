# PipeDRAM: A Data-Transposition-Free Processing-Using-DRAM Architecture with Hardware/Software Pipelining

PipeDRAM is a processing-using-DRAM (PUD) architecture that executes bulk
bitwise and arithmetic operations directly on horizontally laid-out data.
Prior PUD architectures (e.g., SIMDRAM) store operands in a vertical layout,
with the bits of each element stacked along a DRAM column, and therefore have
to transpose data between the horizontal layout used by the rest of the system
and the vertical layout used for computation. PipeDRAM removes this
transposition with two key ideas:

- **Mat-aware bit mapping (MABM).** A fixed bit permutation in the memory
  controller places all bits of a data element in one DRAM chip and maps its
  bit-position j to mat (j mod 32) of that chip. The permutation is applied to
  every cache line on writes and inverted on reads, so the CPU view of memory
  is unchanged.
- **Pipelined execution.** A modulo scheduler splits each PUD operation into
  bit-dependent primitives (on the carry chain) and bit-independent
  primitives, and overlaps them across mats, with each mat acting as one
  pipeline stage. Carries move between mats over a hierarchical in-DRAM copy
  path (LISA between subarrays, the inter-mat interconnect within a subarray).

This repository contains the simulation infrastructure, applications, and
scripts used to evaluate PipeDRAM in our MICRO 2026 paper:

- `pud_simulator/`: an analytical simulator that executes each PUD primitive
  (AAP/AP) of a uProgram individually. It models SIMDRAM, PipeDRAM
  (sequential, pipelined, and hierarchical carry propagation), Proteus, and a
  bit-parallel Kogge-Stone substrate, including data-transposition costs.
- `workloads/`: 11 real-world applications instrumented with an analytical
  PUD cost model.
- `area/`: Verilog for the MABM permutation and the area analysis.
- `scripts/`: one-command reproduction and validation of Figures 4 and 9-14.

```sh
./scripts/setup.sh
./scripts/reproduce.sh quick    # all paper figures, under two minutes
./scripts/validate.sh quick
```

## Contents

- [Getting started](#getting-started)
- [Reproducing the paper figures](#reproducing-the-paper-figures)
- [Repository structure](#repository-structure)
- [Documentation](#documentation)
- [Citation](#citation)
- [Acknowledgments](#acknowledgments)
- [License](#license)

## Getting started

### Requirements

- Ubuntu 24.04 (x86-64) with Python 3.12
- GCC/G++ and GNU Make (applications)
- Yosys and Icarus Verilog (MABM RTL checks)
- Poppler and `pdfcrop` (figure post-processing and validation)
- Liberation Sans (figure fonts)

```sh
sudo apt update
sudo apt install -y build-essential python3-venv texlive-extra-utils \
  poppler-utils fonts-liberation yosys iverilog
```

### Setup

```sh
./scripts/setup.sh
```

This creates `.venv/` with the pinned Python packages from `requirements.txt`
and checks that the system tools above are installed.

### Run the simulator

```sh
source .venv/bin/activate

# List the 19 supported PUD operations
python -m pud_simulator --list-operations

# 32-bit addition on 64M elements in SIMDRAM, including worst-case transposition
python -m pud_simulator -op addition -n 32 -m 67108864 --transposition worst

# 32-bit multiplication in PipeDRAM with modulo scheduling and hierarchical
# carry propagation, using 32 subarrays
python -m pud_simulator -op multiplication -n 32 -m 67108864 \
  --simulator pipedram --scheduling pipelined --carry-copy hierarchical \
  --compute-subarrays 32

# Per-primitive trace of a 4-bit division
python -m pud_simulator -op division -n 4 --show-trace
```

The CLI uses the DRAM configuration of the paper (`presets.pipedram_config()`).
See [`pud_simulator/README.md`](pud_simulator/README.md) for the Python API,
all options, and the DRAM parameters.

## Reproducing the paper figures

```sh
./scripts/reproduce.sh quick
./scripts/validate.sh quick
```

`quick` mode regenerates Figures 4 and 9-14 and the area analysis from the
paper's application results in `reference/results/` (under two minutes, less
than 2 GB of memory). `full` mode rebuilds and reruns all redistributable
applications at paper scale before plotting (up to 24 hours, 64 GB of memory,
10 GB of disk):

```sh
./scripts/reproduce.sh full [--heartwall-input /path/to/test_4k.avi]
./scripts/validate.sh full
```

Figures are written to `generated/figures/`, regenerated CSVs to
`generated/results/`, and logs to `generated/logs/`. A single figure can be
reproduced with `./scripts/reproduce_figure.sh <figure> replot|full`, e.g.,
`./scripts/reproduce_figure.sh 10 replot`.

| Figure | Content | Generator | Output |
|---:|---|---|---|
| 4 | Data-transposition overhead | `plotting/plot_transposition_reuse.py` | `transposition_reuse_combined.pdf` |
| 9 | Bit-serial and bit-parallel throughput | `plotting/plot_combined_scalability.py` | `combined_scalability.pdf` |
| 10 | Application execution time and energy | `plotting/plot_speedup.py` | `workload_normalized_combined.pdf` |
| 11 | CPU-normalized energy efficiency | `plotting/plot_energy_efficiency.py` | `energy_efficiency_cpu_normalized.pdf` |
| 12 | Mat and chip utilization | `plotting/plot_mat_utilization.py` | `mat_utilization.pdf` |
| 13 | Bit-precision and data-size trade-off | `plotting/plot_precision_datasize_tradeoff.py` | `precision_datasize_tradeoff.pdf` |
| 14 | Exposed carry-propagation overhead | `plotting/plot_exposed_carry.py` | `exposed_carry.pdf` |
| - | DRAM area overhead | `plotting/plot_area_comparison.py` | `area_comparison.pdf` |

`PipeDRAM.ipynb` walks through the same steps figure by figure.

Two inputs are not redistributed: the SPEC CPU2017 x264 source and media, and
the paper-scale heartwall video. Their paper results are included, so both
applications appear in all figures; `full` mode reruns heartwall when the
checksum-verified video is supplied. The kmeans input is generated
deterministically. See [`docs/DATASETS.md`](docs/DATASETS.md).

## Repository structure

```
.
├── pud_simulator/      Per-primitive PUD simulator (SIMDRAM, PipeDRAM, Proteus, bit-parallel)
│   └── operations/     uProgram generators for the 19 PUD operations
├── workloads/          Instrumented applications (PolyBench, Rodinia, Phoenix)
│   └── util/           Analytical PUD cost model (bbop_manager.c/.h)
├── plotting/           Figure generators
├── area/               Area data and MABM Verilog (permutation, inverse, testbench)
├── scripts/            Setup, reproduction, and validation
├── tools/              Metrics collection, validation, area checks, cost-model table generator
├── reference/          Paper results, reference figures, and renderings
├── docs/               Reproduction, datasets, and area analysis
├── licenses/           Third-party notices
└── PipeDRAM.ipynb      Figure-by-figure walkthrough
```

The cost-model lookup tables in `workloads/util/bbop_manager.c` are generated
from the simulator with `python tools/generate_tables.py`.

## Documentation

- [Simulator](pud_simulator/README.md): operations, options, Python API, DRAM
  parameters, and execution model.
- [Workloads](workloads/README.md): instrumented applications and the PUD
  cost model.
- [Reproducibility](docs/REPRODUCIBILITY.md): what is reproduced, fixed inputs,
  and validation criteria.
- [Datasets](docs/DATASETS.md): kmeans generation and heartwall/x264
  provenance.
- [Area analysis](docs/AREA_ANALYSIS.md): area values, their sources, and the
  MABM RTL checks.

## Citation

If you use PipeDRAM, please cite:

```bibtex
@inproceedings{oliveira2026pipedram,
    author = "Oliveira, Geraldo F and Olgun, Ataberk and Y{\"u}ksel, Ismail Emir and Bostanc{\i}, F. Nisa and Becker, Pedro H. E. and Sadrosadati, Mohammad and Ghose, Saugata and G{\'o}mez-Luna, Juan and Mutlu, Onur",
    title = "{PipeDRAM: A Data-Transposition-Free Processing-Using-DRAM Architecture with Hardware/Software Pipelining}",
    booktitle = "MICRO",
    year = "2026"
}
```

```bibtex
@misc{oliveira2026pipedram_arxiv,
    author = "Oliveira, Geraldo F and Olgun, Ataberk and Y{\"u}ksel, Ismail Emir and Bostanc{\i}, F. Nisa and Becker, Pedro H. E. and Sadrosadati, Mohammad and Ghose, Saugata and G{\'o}mez-Luna, Juan and Mutlu, Onur",
    title = "{PipeDRAM: A Data-Transposition-Free Processing-Using-DRAM Architecture with Hardware/Software Pipelining}",
    howpublished = "arXiv:2609.30998 [cs.AR]",
    year = "2026"
}
```

## Acknowledgments

The application instrumentation and the SIMDRAM/MIMDRAM cost models build on
[Proteus](https://github.com/CMU-SAFARI/Proteus), and the uPrograms follow
[MIMDRAM](https://github.com/CMU-SAFARI/MIMDRAM). We thank the authors of
PolyBench, Rodinia, and Phoenix, whose applications we instrument.

We acknowledge the generous gifts from our industrial partners, including
Google, Huawei, Intel, and Microsoft. This work is supported in part by the ETH
Future Computing Laboratory (EFCL), Semiconductor Research Corporation, AI Chip
Center for Emerging Smart Systems (ACCESS), sponsored by InnoHK funding, Hong
Kong SAR, and European Union's Horizon programme for research and innovation
[101047160 - BioPIM].

## License

PipeDRAM is released under the [MIT License](LICENSE). The applications in
`workloads/` keep their original licenses and notices (see
[`licenses/THIRD_PARTY_NOTICES.md`](licenses/THIRD_PARTY_NOTICES.md)).

Questions and bug reports are welcome as GitHub issues.
