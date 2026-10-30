"""
SIMDRAM Simulator
=================

Walks each primitive in a uProgram individually, accumulating
latency and energy per primitive.  PipeDRAM and Proteus reuse the same
primitive sequence for DDG construction and pipeline scheduling.
"""

import math
from dataclasses import dataclass, field, asdict
from typing import List

from .config import DRAMConfig
from .primitives import UProgram, Primitive, OpType


# Simulation results

@dataclass
class SimResult:
    """All simulation outputs."""

    # Configuration
    operation:       str = ""
    bit_precision:   int = 0
    num_elements:    int = 0

    # Primitive breakdown
    total_aaps:      int = 0
    total_aps:       int = 0
    total_primitives:int = 0

    # Parallelism
    simd_lanes:              int = 0
    num_compute_subarrays:   int = 0
    num_compute_banks:       int = 0
    elements_per_batch:      int = 0
    num_batches:             int = 0

    # Performance
    latency_per_pass_ns:     float = 0.0
    pipeline_fill_ns:        float = 0.0   # one-time pipeline fill cost (pipelined modes)
    per_batch_time_ns:       float = 0.0   # steady-state per-batch cost (pipelined modes)
    compute_latency_ns:      float = 0.0
    transposition_latency_ns:float = 0.0
    total_latency_ns:        float = 0.0
    throughput_gops:          float = 0.0

    # Energy
    energy_per_pass_nj:      float = 0.0
    compute_energy_nj:       float = 0.0
    transposition_energy_nj: float = 0.0
    total_energy_nj:         float = 0.0
    energy_eff_gops_per_w:   float = 0.0

    # Per-primitive trace (optional)
    trace: List[dict] = field(default_factory=list)


# Simulator

# Transposition modes
TRANSP_NONE   = "none"
TRANSP_WORST  = "worst"
TRANSP_COMMON = "common"


class SIMDRAMSimulator:
    """Walk each primitive in a uProgram, accumulating timing and energy.

    For SIMDRAM all primitives execute sequentially within a subarray.
    Parallelism comes from SIMD lanes (columns), subarrays (SALP),
    and banks (BLP).
    """

    def __init__(self, config: DRAMConfig):
        self.cfg = config

    def simulate(
        self,
        uprog:                   UProgram,
        num_elements:            int,
        num_compute_subarrays:   int  = 1,
        num_compute_banks:       int  = 1,
        transposition_mode:      str  = TRANSP_NONE,
        cache_resident_fraction: float = 0.5,
        enable_trace:            bool = False,
    ) -> SimResult:

        cfg = self.cfg
        n   = uprog.bit_precision
        r   = SimResult()

        # config echo
        r.operation     = uprog.name
        r.bit_precision = n
        r.num_elements  = num_elements

        # primitive counts
        r.total_aaps       = uprog.num_aaps
        r.total_aps        = uprog.num_aps
        r.total_primitives = uprog.total_primitives

        # parallelism
        r.simd_lanes            = cfg.simd_lanes
        r.num_compute_subarrays = min(num_compute_subarrays,
                                      cfg.num_subarrays_per_bank)
        r.num_compute_banks     = min(num_compute_banks,
                                      cfg.num_banks_per_rank)
        r.elements_per_batch    = (r.simd_lanes
                                   * r.num_compute_subarrays
                                   * r.num_compute_banks)
        r.num_batches = max(1, math.ceil(num_elements / r.elements_per_batch))

        # walk primitives
        pass_latency_ns = 0.0
        pass_energy_nj  = 0.0

        for i, prim in enumerate(uprog.primitives):
            if prim.op == OpType.AAP:
                lat = cfg.t_AAP
                eng = cfg.e_AAP
            else:
                lat = cfg.t_AP
                eng = cfg.e_AP

            pass_latency_ns += lat
            pass_energy_nj  += eng

            if enable_trace:
                r.trace.append({
                    "idx": i,
                    "op": prim.op.value,
                    "dst": prim.dst,
                    "src": prim.src,
                    "bit_idx": prim.bit_idx,
                    "phase": prim.phase,
                    "cumulative_lat_ns": pass_latency_ns,
                    "cumulative_eng_nj": pass_energy_nj,
                })

        r.latency_per_pass_ns = pass_latency_ns
        r.energy_per_pass_nj  = pass_energy_nj

        # scale to all batches
        r.compute_latency_ns = r.num_batches * r.latency_per_pass_ns

        total_subarray_passes = r.num_batches * (r.num_compute_subarrays
                                                  * r.num_compute_banks)
        r.compute_energy_nj = total_subarray_passes * r.energy_per_pass_nj

        # transposition
        r.transposition_latency_ns = self._transposition_latency(
            uprog, n, r.num_batches,
            r.num_compute_subarrays, r.num_compute_banks,
            transposition_mode, cache_resident_fraction)
        r.transposition_energy_nj = self._transposition_energy(
            uprog, n, r.num_compute_subarrays, r.num_compute_banks,
            r.num_batches, transposition_mode, cache_resident_fraction)

        # totals
        r.total_latency_ns = r.compute_latency_ns + r.transposition_latency_ns
        r.total_energy_nj  = r.compute_energy_nj  + r.transposition_energy_nj

        if r.total_latency_ns > 0:
            r.throughput_gops = num_elements / r.total_latency_ns
        if r.total_energy_nj > 0:
            r.energy_eff_gops_per_w = num_elements / r.total_energy_nj

        return r

    # Transposition helpers

    def _transposition_latency(self, uprog, n, num_batches,
                               num_compute_subarrays, num_compute_banks,
                               mode, cache_frac):
        if mode == TRANSP_NONE:
            return 0.0

        cfg = self.cfg
        num_operands = uprog.num_inputs + uprog.num_outputs

        # Time to write/read one row: activate + bus transfer + precharge
        row_transfer_ns = cfg.row_size_bytes / cfg.bus_bandwidth_GBps  # bytes / (GB/s) = ns
        time_per_row = cfg.t_RCD + row_transfer_ns + cfg.t_RP

        # Per-bank bottleneck: rows are serialized within each bank
        per_bank_latency = (num_operands * num_compute_subarrays
                            * n * time_per_row)

        # Bus bottleneck: all banks share the same bus
        total_bytes = (num_operands * num_compute_subarrays
                       * num_compute_banks * n * cfg.row_size_bytes)
        bus_limited_latency = total_bytes / cfg.bus_bandwidth_GBps  # ns

        transpose_per_batch = max(per_bank_latency, bus_limited_latency)

        if mode == TRANSP_COMMON:
            transpose_per_batch *= cache_frac

        return num_batches * transpose_per_batch

    def _transposition_energy(self, uprog, n, num_compute_subarrays,
                              num_compute_banks, num_batches, mode, cache_frac):
        if mode == TRANSP_NONE:
            return 0.0

        cfg = self.cfg
        num_operands = uprog.num_inputs + uprog.num_outputs

        # Total bits transposed across all batches and all compute units
        total_bits = (num_operands * num_compute_subarrays
                      * num_compute_banks * n
                      * cfg.simd_lanes  # bits per row
                      * num_batches)

        # Energy: e_transp_pj_per_bit, convert pJ to nJ (/ 1000)
        energy_nj = total_bits * cfg.e_transp_pj_per_bit / 1000.0

        if mode == TRANSP_COMMON:
            energy_nj *= cache_frac

        return energy_nj
