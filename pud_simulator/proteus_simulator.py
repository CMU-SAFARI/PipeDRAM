"""
Proteus Simulator
=================

Simulates Proteus execution with two modes:

- **sequential**: bits distributed across subarrays, processed one at a time
  with LISA inter-subarray carry propagation between consecutive bit positions.
- **pipelined**: overlaps bit-independent work across subarrays using modulo
  scheduling (II = max dep_latency + LISA carry cost).

Differences from SIMDRAM:
  - Bit j is mapped to subarray j (bit-level parallelism across subarrays)
  - Subarrays are used for bit distribution, not data parallelism
  - elements_per_batch = simd_lanes x num_banks (no subarray multiplier)
  - Carry propagation via LISA (1-hop: 142 ns, 57.6 nJ for DDR4-2400)
  - Full-row activation (65,536 SIMD lanes), no mat-level scaling
  - Requires transposition (vertical data layout, same as SIMDRAM)

Differences from PipeDRAM:
  - Pipeline stages = subarrays (not mats)
  - SIMD width = full row (65,536) instead of mat_columns x num_chips (2,048)
  - Carry via LISA (inter-subarray) instead of inter-mat interconnect
  - Energy per primitive = full row (no 1/128 scaling)
  - Requires transposition (PipeDRAM's MABM avoids it)
"""

import math
from typing import Dict

from .config import DRAMConfig
from .ddg import build_ddg, DependencyGraph, PrimClass
from .primitives import UProgram, OpType
from .simulator import SimResult, SIMDRAMSimulator


class ProteusSimulator:
    """Proteus simulator with sequential or pipelined scheduling.

    Proteus distributes bit-positions across subarrays and uses LISA
    for inter-subarray carry propagation.  Each subarray processes one
    bit-position using the full row width (65,536 SIMD lanes).
    """

    def __init__(self, config: DRAMConfig, scheduling_mode: str = "pipelined"):
        if scheduling_mode not in ("sequential", "pipelined"):
            raise ValueError(f"Unknown scheduling_mode: {scheduling_mode!r}")
        self.cfg = config
        self.scheduling_mode = scheduling_mode
        # Reuse SIMDRAM's transposition model
        self._simdram = SIMDRAMSimulator(config)

    def simulate(
        self,
        uprog:                   UProgram,
        num_elements:            int,
        num_compute_subarrays:   int  = 1,
        num_compute_banks:       int  = 1,
        transposition_mode:      str  = "none",
        cache_resident_fraction: float = 0.5,
        enable_trace:            bool = False,
    ) -> SimResult:

        cfg = self.cfg
        r = SimResult()

        # config echo
        r.operation     = uprog.name
        r.bit_precision = uprog.bit_precision
        r.num_elements  = num_elements

        # primitive counts
        r.total_aaps       = uprog.num_aaps
        r.total_aps        = uprog.num_aps
        r.total_primitives = uprog.total_primitives

        # parallelism
        # Proteus uses subarrays for bit distribution, not data parallelism.
        # Data parallelism comes only from SIMD lanes and banks.
        r.simd_lanes            = cfg.simd_lanes  # full row: 65,536
        r.num_compute_subarrays = 1  # subarrays used for bits, not data
        r.num_compute_banks     = min(num_compute_banks,
                                      cfg.num_banks_per_rank)
        r.elements_per_batch    = r.simd_lanes * r.num_compute_banks
        r.num_batches = max(1, math.ceil(num_elements / r.elements_per_batch))

        # build DDG and compute per-bit latency breakdown
        ddg = build_ddg(uprog)
        breakdown = self._per_bit_latency(uprog, ddg)

        init_latency       = breakdown["init_latency"]
        dep_latencies      = breakdown["dep_latencies"]
        full_bit_latencies = breakdown["full_bit_latencies"]
        num_bit_positions  = breakdown["num_bit_positions"]
        all_unclassified   = breakdown["all_unclassified"]

        # compute latency_per_pass
        II = 0.0

        if all_unclassified:
            # No bit structure (e.g., division): sequential sum
            r.latency_per_pass_ns = sum(
                cfg.t_AAP if p.op == OpType.AAP else cfg.t_AP
                for p in uprog.primitives
            )
        elif self.scheduling_mode == "sequential":
            num_copies = max(0, num_bit_positions - 1)
            r.latency_per_pass_ns = (
                init_latency
                + sum(full_bit_latencies.values())
                + num_copies * cfg.t_LISA
            )
        else:  # pipelined
            if num_bit_positions <= 1:
                r.latency_per_pass_ns = (
                    init_latency
                    + sum(full_bit_latencies.values())
                )
            else:
                max_dep = max(dep_latencies.values()) if dep_latencies else 0.0
                max_full_bit = max(full_bit_latencies.values()) if full_bit_latencies else 0.0

                II = max_dep + cfg.t_LISA
                r.latency_per_pass_ns = (
                    init_latency
                    + max(0, num_bit_positions - 1) * II
                    + max_full_bit
                )
                r.pipeline_fill_ns = init_latency + max(0, num_bit_positions - 1) * II
                r.per_batch_time_ns = max_full_bit

        # energy per pass (full-row activation)
        pass_energy = 0.0
        for p in uprog.primitives:
            e = cfg.e_AAP if p.op == OpType.AAP else cfg.e_AP
            pass_energy += e  # full row, no scaling

        # LISA carry energy
        if not all_unclassified and num_bit_positions > 1:
            num_copies = max(0, num_bit_positions - 1)
            pass_energy += num_copies * cfg.e_LISA

        r.energy_per_pass_nj = pass_energy

        # scale to all batches
        if (self.scheduling_mode == "pipelined"
                and not all_unclassified
                and num_bit_positions > 1):
            max_full_bit = max(full_bit_latencies.values())
            pipeline_fill = init_latency + max(0, num_bit_positions - 1) * II
            r.compute_latency_ns = pipeline_fill + r.num_batches * max_full_bit
        else:
            r.compute_latency_ns = r.num_batches * r.latency_per_pass_ns

        # Energy: each batch activates subarrays across all banks
        # Proteus uses num_bit_positions subarrays per bank for one pass
        total_subarray_passes = r.num_batches * r.num_compute_banks
        r.compute_energy_nj = total_subarray_passes * r.energy_per_pass_nj

        # transposition (same model as SIMDRAM)
        # Proteus uses vertical data layout, so transposition is needed.
        # Use SIMDRAM's transposition model but with Proteus batching.
        r.transposition_latency_ns = self._simdram._transposition_latency(
            uprog, uprog.bit_precision, r.num_batches,
            1,  # transposition is per-bank, not per-subarray
            r.num_compute_banks,
            transposition_mode, cache_resident_fraction)
        r.transposition_energy_nj = self._simdram._transposition_energy(
            uprog, uprog.bit_precision,
            1, r.num_compute_banks,
            r.num_batches, transposition_mode, cache_resident_fraction)

        # totals
        r.total_latency_ns = r.compute_latency_ns + r.transposition_latency_ns
        r.total_energy_nj  = r.compute_energy_nj  + r.transposition_energy_nj

        if r.total_latency_ns > 0:
            r.throughput_gops = num_elements / r.total_latency_ns
        if r.total_energy_nj > 0:
            r.energy_eff_gops_per_w = num_elements / r.total_energy_nj

        return r

    def _per_bit_latency(
        self, uprog: UProgram, ddg: DependencyGraph
    ) -> dict:
        """Compute per-bit-position latency breakdown from the DDG."""
        cfg = self.cfg

        init_latency = 0.0
        dep_latencies: Dict[int, float] = {}
        indep_latencies: Dict[int, float] = {}
        full_bit_latencies: Dict[int, float] = {}

        has_bit_positions = ddg.num_bit_positions > 0
        all_unclassified = not has_bit_positions

        if has_bit_positions:
            for i in ddg.bit_groups.get(-1, []):
                p = uprog.primitives[i]
                init_latency += cfg.t_AAP if p.op == OpType.AAP else cfg.t_AP

            for bit_idx in sorted(ddg.bit_groups):
                if bit_idx < 0:
                    continue
                dep_lat = 0.0
                indep_lat = 0.0
                for i in ddg.bit_groups[bit_idx]:
                    p = uprog.primitives[i]
                    lat = cfg.t_AAP if p.op == OpType.AAP else cfg.t_AP
                    cls = ddg.classifications.get(i, PrimClass.UNCLASSIFIED)
                    if cls == PrimClass.BIT_DEPENDENT:
                        dep_lat += lat
                    else:
                        indep_lat += lat

                dep_latencies[bit_idx] = dep_lat
                indep_latencies[bit_idx] = indep_lat
                full_bit_latencies[bit_idx] = dep_lat + indep_lat

        return {
            "init_latency": init_latency,
            "dep_latencies": dep_latencies,
            "indep_latencies": indep_latencies,
            "full_bit_latencies": full_bit_latencies,
            "num_bit_positions": ddg.num_bit_positions,
            "all_unclassified": all_unclassified,
        }
