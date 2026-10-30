"""Pretty-printing for simulation results and DRAM configuration."""

from .config import DRAMConfig
from .simulator import SimResult
from .primitives import UProgram


def _fmt(val: float, unit: str = "") -> str:
    if val == 0:
        return f"0 {unit}".strip()
    if abs(val) >= 1e6:
        return f"{val:,.0f} {unit}".strip()
    if abs(val) >= 1:
        return f"{val:,.2f} {unit}".strip()
    if abs(val) >= 1e-3:
        return f"{val:.4f} {unit}".strip()
    return f"{val:.6g} {unit}".strip()


def print_config(cfg: DRAMConfig):
    W = 62
    thin = "─" * W
    print(f"  {thin}")
    print(f"  DRAM Configuration")
    print(f"  {thin}")
    print(f"  Channels           : {cfg.num_channels}")
    print(f"  Ranks / channel    : {cfg.num_ranks_per_channel}")
    print(f"  Banks / rank       : {cfg.num_banks_per_rank}")
    print(f"  Subarrays / bank   : {cfg.num_subarrays_per_bank}")
    print(f"  Rows / subarray    : {cfg.num_rows_per_subarray}")
    print(f"  Row size           : {cfg.row_size_bytes} B "
          f"({cfg.simd_lanes:,} bits)")
    print(f"  Mats / subarray    : {cfg.num_mats_per_subarray}")
    print(f"  Columns / mat      : {cfg.mat_columns:,}")
    print(f"  Chips / rank       : {cfg.num_chips_per_rank}  "
          f"(×{cfg.dq_width_per_chip})")
    print(f"  t_AAP              : {cfg.t_AAP} ns")
    print(f"  t_AP               : {cfg.t_AP} ns")
    print(f"  e_AAP              : {cfg.e_AAP} nJ")
    print(f"  e_AP               : {cfg.e_AP} nJ")
    print()


def print_results(r: SimResult, uprog: UProgram, cfg: DRAMConfig,
                  mode: str = "SIMDRAM Mode"):
    W = 62
    sep = "═" * W
    thin = "─" * W

    print()
    print(f"  {sep}")
    print(f"  PUD Simulator — {mode}")
    print(f"  {sep}")
    print()
    print(f"  Operation          : {r.operation}")
    print(f"  Bit Precision      : {r.bit_precision}")
    print(f"  Num Elements       : {r.num_elements:,}")
    print()
    print(f"  {thin}")
    print(f"  µProgram Breakdown")
    print(f"  {thin}")
    print(f"  Total primitives   : {r.total_aaps} AAP + "
          f"{r.total_aps} AP = {r.total_primitives}")
    print()
    print(f"  {thin}")
    print(f"  Parallelism")
    print(f"  {thin}")
    print(f"  SIMD lanes/sub.    : {r.simd_lanes:,}")
    print(f"  Compute subarrays  : {r.num_compute_subarrays}")
    print(f"  Compute banks      : {r.num_compute_banks}")
    print(f"  Elements per batch : {r.elements_per_batch:,}")
    print(f"  Sequential batches : {r.num_batches:,}")
    print()
    print(f"  {thin}")
    print(f"  Performance")
    print(f"  {thin}")
    print(f"  Latency / pass     : {_fmt(r.latency_per_pass_ns, 'ns')}")
    print(f"  Compute latency    : {_fmt(r.compute_latency_ns, 'ns')}"
          f"  ({_fmt(r.compute_latency_ns / 1e3, 'µs')})")
    if r.transposition_latency_ns > 0:
        print(f"  Transposition lat. : {_fmt(r.transposition_latency_ns, 'ns')}")
    print(f"  Total latency      : {_fmt(r.total_latency_ns, 'ns')}"
          f"  ({_fmt(r.total_latency_ns / 1e3, 'µs')})")
    print(f"  Throughput         : {_fmt(r.throughput_gops, 'GOPS/s')}")
    print()
    print(f"  {thin}")
    print(f"  Energy")
    print(f"  {thin}")
    print(f"  Energy / pass      : {_fmt(r.energy_per_pass_nj, 'nJ')}")
    print(f"  Compute energy     : {_fmt(r.compute_energy_nj, 'nJ')}")
    if r.transposition_energy_nj > 0:
        print(f"  Transposition E    : {_fmt(r.transposition_energy_nj, 'nJ')}")
    print(f"  Total energy       : {_fmt(r.total_energy_nj, 'nJ')}")
    print(f"  Energy efficiency  : {_fmt(r.energy_eff_gops_per_w, 'GOPS/W')}")
    print()
    print(f"  {sep}")
    print()
