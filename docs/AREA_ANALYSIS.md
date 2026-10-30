# Area analysis

The area analysis separates the calculations and RTL checks that this
repository reproduces from values taken from prior published evaluations.

## Reproduced checks

Run:

```bash
./scripts/setup.sh
./scripts/reproduce_area.sh
```

This performs four checks:

1. The DRAM-side components sum to 1.86%:
   1.11% fine-grained mat substrate + 0.60% LISA + 0.15% SALP.
2. The largest 1,350-entry schedule requires 5,400 bytes at four bytes per
   entry and therefore fits in the provisioned 16 KiB scratchpad.
3. Yosys synthesizes the MABM permutation and inverse to zero logic cells.
   MABM is fixed bit wiring rather than combinational logic.
4. Icarus Verilog confirms that the inverse permutation reconstructs every
   supplied test vector.

The command writes `generated/area_metrics.json` and
`generated/figures/area_comparison.pdf`. `scripts/reproduce.sh` runs the same
steps in both `quick` and `full` mode.

## Reported area values

| Component or result | Value | Basis |
|---|---:|---|
| MIMDRAM fine-grained mat substrate | 1.11% of DRAM die | Prior CACTI-based MIMDRAM evaluation |
| LISA inter-subarray isolation | 0.60% of DRAM die | Prior LISA evaluation |
| SALP row-decoder latches | 0.15% of DRAM die | Prior SALP/Proteus evaluation |
| PipeDRAM DRAM-side total | 1.86% | Recomputed sum |
| MABM logic | 0 cells | Reproduced Yosys synthesis |
| Schedule scratchpad | 16 KiB | Design provision |
| Extended controller | 0.27 mm^2 | CACTI estimate |
| CPU-die overhead | 0.05% | 0.27 mm^2 normalized to the evaluated Xeon die |
| Eliminated transposition unit | 0.06 mm^2 | Prior SIMDRAM/Proteus evaluation |

The prior-work and CACTI-derived values are taken as published, not
remeasured. Reproducing them would require the CACTI configurations and
technology assumptions of each cited work. `area/area_metrics.json` records
which values are which and is the single source for the plot and the
arithmetic checks.

## Comparison plot

The comparison plot places PipeDRAM's 1.86% structural overhead next to
published DRAM mechanisms: fine-grained mats, Sectored DRAM, DDR5 per-row
counters, and in-DRAM MAC compute. The comparison values come from the cited
works, not from the included RTL.
