"""
SKS Adder Integration with PUD Simulator Framework
====================================================

Wraps the standalone SKS simulator (bit-parallel Simplified Kogge-Stone
majority-logic adder) so that it can be queried by the cost-model table
generator (tools/generate_tables.py) alongside the bit-serial operations.

Differences from bit-serial operations:
  - SKS uses subarray-level parallelism (Proteus-style): bit i -> SA_i
  - Carry propagation via prefix tree + LISA (O(log n) depth, not O(n))
  - The SKS simulator handles its own scheduling (SALP, tFAW, dependencies)
  - Only per-pass latency/energy at supported precisions is queried
  - Subtraction = addition with inverted B and carry-in=1

Supported precisions: powers of 2 >= 2 (2, 4, 8, 16, 32).
Non-power-of-2 precisions use the next power of 2 (conservative).
"""

import math
from typing import Tuple

from .config import DRAMConfig
from .sks_simulator import (
    DRAMConfig as SKS_DRAMConfig,
    MicroProgram as SKS_MicroProgram,
    analyse as sks_analyse,
)


def _pud_to_sks_config(cfg: DRAMConfig, n_bits: int) -> SKS_DRAMConfig:
    """Map PUD DRAMConfig to SKS DRAMConfig, aligning parameters.

    The SKS simulator distinguishes AAP_NORMAL (2-wordline copy) from
    AAP_TRA (3-wordline majority).  In the PUD framework:
      - t_AAP corresponds to AAP_NORMAL (row copy = ACT src, ACT dst, PRE)
      - t_AP  corresponds to the AP primitive
      - AAP_TRA uses the same timing as AAP_NORMAL (both are 2-ACT sequences;
        the TRA address just selects which rows to activate simultaneously)

    Energy: PUD uses uniform e_AAP for all AAP variants.  The SKS simulator
    allows a separate e_aap_tra; we set it equal to e_AAP for consistency.
    """
    return SKS_DRAMConfig(
        # Timing
        t_aap_normal      = cfg.t_AAP,
        t_aap_tra         = cfg.t_AAP,      # TRA same timing as normal AAP
        t_ap              = cfg.t_AP,
        t_lisa            = cfg.t_LISA,
        t_faw             = 0.0,             # tFAW not modeled in PUD framework

        # Energy
        e_aap_normal      = cfg.e_AAP,
        e_aap_tra         = cfg.e_AAP,       # uniform energy
        e_ap              = cfg.e_AP,
        e_lisa            = cfg.e_LISA,

        # Structure (Proteus-style: full-row SIMD, subarrays for bit distribution)
        num_subarrays     = max(cfg.num_subarrays_per_bank, n_bits),
        num_rows          = cfg.num_rows_per_subarray,
        num_cols          = cfg.simd_lanes,   # 65,536 bits (full row)

        # Concurrency: unlimited SALP (consistent with Proteus model)
        max_concurrent_sa = 0,

        # tFAW disabled (not in PUD framework)
        tfaw_enabled      = False,
        tfaw_act_count    = 4,
    )


def _next_power_of_2(n: int) -> int:
    """Round up to the next power of 2 (minimum 2)."""
    if n <= 2:
        return 2
    return 1 << math.ceil(math.log2(n))


def sks_addition_latency_energy(
    cfg: DRAMConfig, n_bits: int
) -> Tuple[float, float]:
    """Return (latency_ns, energy_nJ) for one SKS addition pass.

    The pass processes cfg.simd_lanes independent additions in SIMD.
    Latency is the total time from first instruction to last instruction.
    Energy is the total for one invocation (one SIMD row).

    For non-power-of-2 precisions, rounds up to the next power of 2
    (conservative estimate; the prefix tree requires a power-of-2 width).
    """
    n = _next_power_of_2(n_bits)
    sks_cfg = _pud_to_sks_config(cfg, n)
    prog = SKS_MicroProgram(n, sks_cfg)
    result = sks_analyse(prog, sks_cfg, verify=False)
    return result.latency_ns, result.total_energy_nJ


def sks_subtraction_latency_energy(
    cfg: DRAMConfig, n_bits: int
) -> Tuple[float, float]:
    """Return (latency_ns, energy_nJ) for one SKS subtraction pass.

    Subtraction = A + ~B + 1.  Additional cost over addition:
      - NOT(B): 2 AAPs per bit position (local to each subarray, parallel)
      - Carry-in = 1: changes c_{-1} from C0 to C1 (no latency cost,
        just uses a different constant row in SA_0)

    The NOT operations execute in parallel across all subarrays before
    the addition begins, so they add exactly 2 * t_AAP to the latency
    (2 sequential AAPs in one subarray, all subarrays in parallel).
    Energy adds 2 * n_bits * e_AAP (one NOT per bit, 2 AAPs each).
    """
    n = _next_power_of_2(n_bits)
    add_lat, add_eng = sks_addition_latency_energy(cfg, n_bits)

    # NOT overhead: 2 AAPs per bit, fully parallel across subarrays
    not_latency = 2 * cfg.t_AAP          # 2 sequential AAPs (one NOT)
    not_energy  = 2 * n * cfg.e_AAP       # 2 AAPs x n subarrays

    return add_lat + not_latency, add_eng + not_energy
