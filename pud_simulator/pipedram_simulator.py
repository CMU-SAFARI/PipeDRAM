"""
PipeDRAM Simulator
==================

Simulates PipeDRAM execution with two modes:

- **sequential**: horizontal layout (MABM, no transposition) but
  bit-positions processed one at a time across mats.
- **pipelined**: overlaps bit-independent work across mats using
  modulo scheduling (II = max dep_latency + carry_propagation).

Carry copy modes (pipelined only):

- **direct**: inter-mat carry propagation within the computing subarray
  (FIGARO HFF path). The computing subarray is blocked for the full
  t_inter_mat_copy duration.
- **hierarchical**: LISA + inter-mat propagation through a neighboring
  copying subarray:
    1. LISA out (computing SA -> copying SA): t_RBM (8 ns). The carry is
       already in the sense amplifiers after MAJ, so only charge sharing
       through the isolation transistors is needed (no ACT, no PRE).
    2. HFF in the copying SA (mat j -> mat j+1): relay only (73 ns).
    3. LISA back (copying SA -> computing SA): t_RBM (8 ns).
    4. Final precharge of the copying SA: t_RP (15 ns).
  The carry is never written to a DRAM row; it stays in the sense
  amplifiers, so the intermediate ACT/PRE steps are skipped. Round trip
  ~104 ns (vs. ~404 ns with a full ACT/PRE at each step). The computing
  subarray is blocked only for step 1 (8 ns); LISA serializes only the
  involved mats, so the other mats keep executing bit-independent work.

Differences from SIMDRAM:
  - SIMD width = mat_columns x num_chips (not the full row)
  - Energy per primitive scales by 1/num_mats_per_subarray
  - No transposition (MABM keeps the horizontal layout)
"""

import math
from collections import defaultdict
from typing import Dict, List

from .config import DRAMConfig
from .ddg import build_ddg, DependencyGraph, PrimClass
from .primitives import UProgram, OpType
from .simulator import SimResult


class PipeDRAMSimulator:
    """PipeDRAM simulator with sequential or pipelined scheduling."""

    def __init__(self, config: DRAMConfig, scheduling_mode: str = "pipelined",
                 carry_copy_mode: str = "direct"):
        if scheduling_mode not in ("sequential", "pipelined"):
            raise ValueError(f"Unknown scheduling_mode: {scheduling_mode!r}")
        if carry_copy_mode not in ("direct", "hierarchical"):
            raise ValueError(f"Unknown carry_copy_mode: {carry_copy_mode!r}")
        self.cfg = config
        self.scheduling_mode = scheduling_mode
        self.carry_copy_mode = carry_copy_mode

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

        # parallelism (mat-level SIMD)
        r.simd_lanes            = cfg.pipedram_simd_lanes
        r.num_compute_subarrays = min(num_compute_subarrays,
                                      cfg.num_subarrays_per_bank)
        r.num_compute_banks     = min(num_compute_banks,
                                      cfg.num_banks_per_rank)
        r.elements_per_batch    = (r.simd_lanes
                                   * r.num_compute_subarrays
                                   * r.num_compute_banks)
        r.num_batches = max(1, math.ceil(num_elements / r.elements_per_batch))

        # build DDG and compute per-bit latency breakdown
        ddg = build_ddg(uprog)
        latency_breakdown = self._per_bit_latency(uprog, ddg)

        init_latency       = latency_breakdown["init_latency"]
        dep_latencies      = latency_breakdown["dep_latencies"]
        indep_latencies    = latency_breakdown["indep_latencies"]
        full_bit_latencies = latency_breakdown["full_bit_latencies"]
        num_bit_positions  = latency_breakdown["num_bit_positions"]
        all_unclassified   = latency_breakdown["all_unclassified"]

        # compute II and latency_per_pass
        II = 0.0  # initiation interval (meaningful only in pipelined mode)
        per_batch_carry_overhead = 0.0

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
                + num_copies * cfg.t_inter_mat_copy
            )
        else:  # pipelined
            if num_bit_positions <= 1:
                # Single bit position: nothing to pipeline
                r.latency_per_pass_ns = (
                    init_latency
                    + sum(full_bit_latencies.values())
                )
            else:
                max_dep = max(dep_latencies.values()) if dep_latencies else 0.0
                max_full_bit = max(full_bit_latencies.values()) if full_bit_latencies else 0.0

                if self.carry_copy_mode == "hierarchical":
                    # Carry stays in the sense amplifiers:
                    #   1. LISA out  (compute SA -> copy SA): t_RBM (8 ns)
                    #   2. HFF       (mat j -> j+1 in copy SA): relay (73 ns)
                    #   3. LISA back (copy SA -> compute SA): t_RBM (8 ns)
                    #   4. Final precharge: t_RP (15 ns)
                    # Round trip ~104 ns (vs. ~404 ns with full ACT/PRE)
                    II = max_dep + cfg.t_hier_round_trip
                    # Per batch, the computing SA is blocked only for LISA out
                    per_batch_carry_overhead = cfg.t_LISA_sensed
                    # Bit-independent work of the busiest mat overlaps with
                    # the carry round trip; only its excess over II counts.
                    max_indep = (max(indep_latencies.values())
                                 if indep_latencies else 0.0)
                    drain = max_dep + max(0.0, max_indep - II)
                else:  # direct
                    # Pipeline II: carry goes through inter-mat in computing SA
                    II = max_dep + cfg.t_inter_mat_copy
                    # Per-batch: computing SA blocked during full inter-mat copy
                    per_batch_carry_overhead = cfg.t_inter_mat_copy
                    # Direct mode: entire subarray is blocked during copy,
                    # so drain = full bit (dep + indep).
                    drain = max_full_bit

                r.latency_per_pass_ns = (
                    init_latency
                    + max(0, num_bit_positions - 1) * II
                    + drain
                )
                r.pipeline_fill_ns = init_latency + max(0, num_bit_positions - 1) * II
                r.per_batch_time_ns = drain + per_batch_carry_overhead

        # energy per pass (mat-level)
        e_scale = 1.0 / cfg.num_mats_per_subarray
        pass_energy = 0.0
        for p in uprog.primitives:
            e = cfg.e_AAP if p.op == OpType.AAP else cfg.e_AP
            pass_energy += e * e_scale

        # Inter-bit copy energy
        if not all_unclassified and num_bit_positions > 1:
            num_copies = max(0, num_bit_positions - 1)
            if self.carry_copy_mode == "hierarchical":
                # 2 LISA transfers (mat-level, fine-grained activation)
                # + 1 inter-mat copy (mat-level) per carry
                avg_mat_energy = ((cfg.e_AAP + cfg.e_AP) / 2.0) * e_scale
                pass_energy += num_copies * (2 * cfg.e_LISA_mat + avg_mat_energy)
            else:
                # Just the inter-mat copy at mat level
                avg_mat_energy = ((cfg.e_AAP + cfg.e_AP) / 2.0) * e_scale
                pass_energy += num_copies * avg_mat_energy

        r.energy_per_pass_nj  = pass_energy

        # scale to all batches
        # In pipelined mode, batches overlap in the pipeline: once mat 0
        # finishes bit 0 of batch K, it starts batch K+1. The fill cost
        # (init + (n-1) x II) is paid once; each batch then costs
        # drain + per_batch_carry_overhead, where the overhead is t_RBM for
        # hierarchical copy and t_inter_mat_copy for direct copy.
        if (self.scheduling_mode == "pipelined"
                and not all_unclassified
                and num_bit_positions > 1):
            pipeline_fill = init_latency + max(0, num_bit_positions - 1) * II
            per_batch_time = drain + per_batch_carry_overhead
            r.compute_latency_ns = pipeline_fill + r.num_batches * per_batch_time
        else:
            r.compute_latency_ns = r.num_batches * r.latency_per_pass_ns

        total_subarray_passes = r.num_batches * (r.num_compute_subarrays
                                                  * r.num_compute_banks)
        r.compute_energy_nj = total_subarray_passes * r.energy_per_pass_nj

        # no transposition for PipeDRAM
        r.transposition_latency_ns = 0.0
        r.transposition_energy_nj  = 0.0

        # totals
        r.total_latency_ns = r.compute_latency_ns
        r.total_energy_nj  = r.compute_energy_nj

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

        # Check if all primitives are UNCLASSIFIED
        has_bit_positions = ddg.num_bit_positions > 0
        all_unclassified = not has_bit_positions

        if has_bit_positions:
            # Init latency: all bit_idx == -1 primitives
            for i in ddg.bit_groups.get(-1, []):
                p = uprog.primitives[i]
                init_latency += cfg.t_AAP if p.op == OpType.AAP else cfg.t_AP

            # Per-bit latency
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
