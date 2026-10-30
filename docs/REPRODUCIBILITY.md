# Reproducing the paper results

## What is reproduced

The repository reproduces Figures 4 and 9-14 of the paper and the area analysis:
data-transposition overhead (Fig. 4), bit-serial and bit-parallel throughput
(Fig. 9), application execution time and energy (Fig. 10), CPU-normalized energy
efficiency (Fig. 11), mat and chip utilization (Fig. 12), the bit-precision and
data-size trade-off (Fig. 13), and exposed carry propagation (Fig. 14). The
simulator executes the same uPrograms with the same DRAM timing and energy
parameters as the paper.

`quick` mode regenerates every figure and metric from the paper's application
results in `reference/results/`. `full` mode additionally compiles and runs every
redistributable application at paper scale and plots the regenerated results.

## Inputs that are not regenerated

The CPU (RAPL) and GPU measurements in `reference/results/baseline/summary.csv`
and `reference/results/cpu_gpu_results.csv` are fixed inputs. They depend on the
measurement platform and are not remeasured.

x264 source and media belong to SPEC CPU2017 and cannot be redistributed. The
x264 latency, energy, and transposition results are included, so x264 appears in
Figures 4, 10, and 11 and in the geometric means.

The paper-scale heartwall video is not redistributed. `full` mode runs heartwall
only with the file whose SHA-256 is recorded in `input_checksums.json`;
otherwise it keeps the paper result. The kmeans input is generated from the
included Rodinia generator with a fixed seed (see `DATASETS.md`).

## Validation

`scripts/validate.sh` compares a reproduction against the reference data:

- **Numerical results.** `expected_metrics.json` records per-application
  transposition overheads, normalized geometric means, crossover points, mat
  utilization, and exposed carry fractions. Simulator-derived values must match
  within a relative and absolute tolerance of 1e-9. In `full` mode, regenerated
  application CSVs must match `reference/results/` within 1e-6.
- **Figures.** Each PDF must exist, have one page, and contain the expected axis
  labels. Each PDF is rasterized at 150 DPI and compared with
  `reference/renderings/`: the images must have the same dimensions and a mean
  absolute pixel error of at most 0.025. PDFs are not compared byte by byte,
  because metadata and backend serialization differ across systems.
- **Area.** The DRAM-side area sum must be 1.86%, the largest schedule must be
  5,400 bytes, MABM must synthesize to zero logic cells, and the RTL round-trip
  test must pass.
- **Reference files.** `reference_checksums.sha256` guards the files under
  `reference/` against accidental modification.

Changing simulator parameters is useful for sensitivity studies, but the changed
results are not expected to pass validation against the paper references.
