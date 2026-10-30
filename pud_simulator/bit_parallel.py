"""
Bit-parallel operation evaluation: Proteus vs PipeDRAM.

Uses the SKS (Simplified Kogge-Stone) adder as the core building block,
parameterized for different physical substrates:

  - Proteus:  subarray-level bit parallelism, LISA carry, linear scheduling
  - PipeDRAM: mat-level bit parallelism, HFF or hierarchical carry, modulo scheduling

All substrates run the same algorithm; only the physical layer differs.

Batching model:
  - Proteus:  groups_per_bank = floor(SAs_per_bank / n_bits)
              Each group has 65,536 SIMD lanes (full row).
  - PipeDRAM: groups_per_sa = floor(mats_per_chip / n_bits)
              Each group has 2,048 SIMD lanes (mat_cols x chips).
              groups_per_bank = groups_per_sa x compute_SAs_per_bank.
"""

import math
from dataclasses import dataclass
from typing import List, Tuple

from .config import DRAMConfig
from .sks_simulator import (
    DRAMConfig as SKS_DRAMConfig,
    MicroProgram as SKS_MicroProgram,
    analyse as sks_analyse,
)


# Helpers

def _next_pow2(n: int) -> int:
    if n <= 2:
        return 2
    return 1 << math.ceil(math.log2(n))


# Substrate configuration

@dataclass
class SubstrateConfig:
    """Physical substrate for bit-parallel evaluation."""
    name: str

    # Inter-unit transfer (LISA for Proteus, HFF or hierarchical for PipeDRAM)
    t_hop: float        # ns per hop
    e_hop: float        # nJ per hop

    # Compute primitive energy (full-row for Proteus, mat-level for PipeDRAM)
    e_aap: float        # nJ per AAP
    e_ap: float         # nJ per AP/TRA

    # Scheduling: 0 = modulo (unlimited SALP), 1 = linear (serial)
    max_concurrent: int

    # SIMD width per group
    simd_width: int

    # Parallelism
    sa_per_bank: int    # total SAs/bank (Proteus) or compute SAs/bank (PipeDRAM)
    mats_per_chip: int  # mats per chip (32 for PipeDRAM; ignored for Proteus)
    num_banks: int
    is_proteus: bool

    def groups_per_bank(self, n_bits: int) -> int:
        """Independent SKS groups that fit in one bank."""
        n = _next_pow2(n_bits)
        if self.is_proteus:
            return self.sa_per_bank // n
        else:
            groups_per_sa = self.mats_per_chip // n
            return groups_per_sa * self.sa_per_bank

    def elements_per_batch(self, n_bits: int) -> int:
        return self.simd_width * self.groups_per_bank(n_bits) * self.num_banks


def build_substrates(cfg_p: DRAMConfig, cfg_d: DRAMConfig) -> List[SubstrateConfig]:
    """Build all 5 evaluation substrates.

    cfg_p: Proteus DRAMConfig (64 SAs/bank)
    cfg_d: PipeDRAM DRAMConfig (32 SAs/bank)
    """
    e_mat = cfg_d.e_AAP / cfg_d.num_mats_per_subarray
    e_mat_ap = cfg_d.e_AP / cfg_d.num_mats_per_subarray
    mpc = cfg_d.num_mats_per_subarray // cfg_d.num_chips_per_rank  # 32

    hier_t = cfg_d.t_hier_round_trip
    hier_e = 2 * cfg_d.e_LISA_mat + e_mat  # mat-level LISA (fine-grained activation)

    common_d = dict(
        e_aap=e_mat, e_ap=e_mat_ap, max_concurrent=0,
        simd_width=cfg_d.pipedram_simd_lanes, mats_per_chip=mpc,
        num_banks=cfg_d.num_banks_per_rank, is_proteus=False,
    )

    return [
        SubstrateConfig(
            name="Proteus",
            t_hop=cfg_p.t_LISA, e_hop=cfg_p.e_LISA,
            e_aap=cfg_p.e_AAP, e_ap=cfg_p.e_AP,
            max_concurrent=1,
            simd_width=cfg_p.simd_lanes,
            sa_per_bank=cfg_p.num_subarrays_per_bank,
            mats_per_chip=0, num_banks=cfg_p.num_banks_per_rank,
            is_proteus=True,
        ),
        SubstrateConfig(name="PipeDRAM-Direct-1SA",
                        t_hop=cfg_d.t_inter_mat_copy, e_hop=e_mat,
                        sa_per_bank=1, **common_d),
        SubstrateConfig(name="PipeDRAM-Hier-2SA",
                        t_hop=hier_t, e_hop=hier_e,
                        sa_per_bank=1, **common_d),
        SubstrateConfig(name="PipeDRAM-Direct-All",
                        t_hop=cfg_d.t_inter_mat_copy, e_hop=e_mat,
                        sa_per_bank=cfg_d.num_subarrays_per_bank, **common_d),
        SubstrateConfig(name="PipeDRAM-Hier-All",
                        t_hop=hier_t, e_hop=hier_e,
                        sa_per_bank=cfg_d.num_subarrays_per_bank // 2,
                        **common_d),
    ]


# SKS simulator bridge

def _make_sks_config(cfg: DRAMConfig, sub: SubstrateConfig,
                     n_bits: int) -> SKS_DRAMConfig:
    """Build SKS DRAMConfig from PUD config + substrate."""
    return SKS_DRAMConfig(
        t_aap_normal=cfg.t_AAP,
        t_aap_tra=cfg.t_AAP,
        t_ap=cfg.t_AP,
        t_lisa=sub.t_hop,
        t_faw=0.0,
        e_aap_normal=sub.e_aap,
        e_aap_tra=sub.e_aap,
        e_ap=sub.e_ap,
        e_lisa=sub.e_hop,
        num_subarrays=max(64, n_bits),
        num_rows=cfg.num_rows_per_subarray,
        num_cols=sub.simd_width,
        max_concurrent_sa=sub.max_concurrent,
        tfaw_enabled=False,
        tfaw_act_count=4,
    )


def _sks_add(cfg: DRAMConfig, sub: SubstrateConfig,
             n_bits: int) -> Tuple[float, float]:
    """Return (latency_ns, energy_nJ) for one SKS addition pass on one group."""
    n = _next_pow2(n_bits)
    sks_cfg = _make_sks_config(cfg, sub, n)
    prog = SKS_MicroProgram(n, sks_cfg)
    result = sks_analyse(prog, sks_cfg, verify=False)
    return result.latency_ns, result.total_energy_nJ


# Result container

@dataclass
class EvalResult:
    config: str
    operation: str
    precision: int

    # Per-pass (one group, one batch)
    latency_per_pass_ns: float
    energy_per_pass_nj: float

    # Batching
    simd_width: int
    groups_per_bank: int
    num_banks: int
    elements_per_batch: int
    num_batches: int

    # Totals (for num_elements)
    num_elements: int
    total_latency_ns: float
    total_energy_nj: float
    throughput_gops: float


def _make_result(sub: SubstrateConfig, op: str, n_bits: int,
                 lat: float, eng: float, num_elements: int) -> EvalResult:
    """Wrap per-pass metrics into a full EvalResult with batching."""
    gpb = sub.groups_per_bank(n_bits)
    epb = sub.elements_per_batch(n_bits)
    nb = max(1, math.ceil(num_elements / epb))

    total_lat = nb * lat
    # Energy: every batch activates gpb groups x num_banks banks
    total_eng = nb * eng * gpb * sub.num_banks
    throughput = num_elements / total_lat  # GOps/s (elements/ns)

    return EvalResult(
        config=sub.name, operation=op, precision=n_bits,
        latency_per_pass_ns=lat, energy_per_pass_nj=eng,
        simd_width=sub.simd_width, groups_per_bank=gpb,
        num_banks=sub.num_banks, elements_per_batch=epb,
        num_batches=nb, num_elements=num_elements,
        total_latency_ns=total_lat, total_energy_nj=total_eng,
        throughput_gops=throughput,
    )


# Operation evaluators

def eval_addition(cfg: DRAMConfig, sub: SubstrateConfig,
                  n_bits: int, num_elements: int) -> EvalResult:
    lat, eng = _sks_add(cfg, sub, n_bits)
    return _make_result(sub, "addition", n_bits, lat, eng, num_elements)


def eval_subtraction(cfg: DRAMConfig, sub: SubstrateConfig,
                     n_bits: int, num_elements: int) -> EvalResult:
    """Subtraction = NOT(B) + SKS_add(A, ~B, carry_in=1)."""
    n = _next_pow2(n_bits)
    add_lat, add_eng = _sks_add(cfg, sub, n_bits)

    # NOT overhead: 2 AAPs per bit, parallel across all units (= 2 sequential AAPs)
    not_lat = 2 * cfg.t_AAP
    not_eng = 2 * n * sub.e_aap

    return _make_result(sub, "subtraction", n_bits,
                        add_lat + not_lat, add_eng + not_eng, num_elements)


# Multiplication (partial-product reduction, modular n-bit)

def eval_multiplication(cfg: DRAMConfig, sub: SubstrateConfig,
                        n_bits: int, num_elements: int) -> EvalResult:
    """Modular n-bit multiplication using partial-product reduction tree.

    Algorithm:
      1. Broadcast: for each multiplier bit b_k, broadcast from unit k
         to units k..n-1 (needed for AND masking).
      2. AND masking: PP_k[j] = AND(A[j], b_k) for j in [k, n-1].
         AND = MAJ(A, b_k, C0) = 4 AAPs per bit.
      3. Reduction: (n-1) sequential SKS additions to sum all PPs.
         All additions use the same n units (different rows), so they
         cannot overlap; the tree is effectively sequential.

    Scheduling differences:
      - Linear (Proteus): all hops and per-unit operations serialize.
      - Modulo (PipeDRAM): hops from different broadcasts overlap;
        AND operations within each iteration run in parallel across units.
    """
    n = _next_pow2(n_bits)
    add_lat, add_eng = _sks_add(cfg, sub, n_bits)
    is_linear = (sub.max_concurrent == 1)

    # Phase 1: Broadcast multiplier bits
    # b_k needs (n-1-k) hops to reach unit n-1.  Total = n(n-1)/2 hops.
    total_bcast_hops = n * (n - 1) // 2
    if is_linear:
        bcast_lat = total_bcast_hops * sub.t_hop           # fully sequential
    else:
        bcast_lat = (n - 1) * sub.t_hop                    # longest chain
    bcast_eng = total_bcast_hops * sub.e_hop

    # Phase 2: AND masking, n iterations (one per PP)
    # PP_k has (n-k) AND ops (units k..n-1). Total = n(n+1)/2 ANDs.
    total_and_ops = n * (n + 1) // 2
    if is_linear:
        and_lat = total_and_ops * 4 * cfg.t_AAP             # all sequential
    else:
        and_lat = n * 4 * cfg.t_AAP                          # n iterations, parallel within
    and_eng = total_and_ops * 4 * sub.e_aap

    # Phase 3: reduction, (n-1) sequential SKS additions
    reduce_lat = (n - 1) * add_lat
    reduce_eng = (n - 1) * add_eng

    lat = bcast_lat + and_lat + reduce_lat
    eng = bcast_eng + and_eng + reduce_eng
    return _make_result(sub, "multiplication", n_bits, lat, eng, num_elements)


# Division (non-restoring, shift-and-add/subtract)

def eval_division(cfg: DRAMConfig, sub: SubstrateConfig,
                  n_bits: int, num_elements: int) -> EvalResult:
    """Non-restoring division: n iterations of conditional add/subtract.

    Algorithm per iteration:
      1. Shift remainder left by 1: n-1 parallel adjacent hops.
      2. Broadcast sign bit (MSB) to all units: (n-1)-hop chain.
      3. Conditional complement: MUX(sign, D, ~D) = 14 AAPs/bit.
         Selects divisor or its complement based on sign.
      4. SKS addition (with carry-in = ~sign).
      5. Set quotient bit: 1 AAP in MSB unit.

    Init: pre-compute ~D (NOT each bit = 2 AAPs/bit).

    Scheduling differences:
      - Linear (Proteus): shifts, MUX operations serialize across units.
      - Modulo (PipeDRAM): shifts and MUX run in parallel across units;
        only the sign broadcast chain is inherently sequential.
    """
    n = _next_pow2(n_bits)
    add_lat, add_eng = _sks_add(cfg, sub, n_bits)
    is_linear = (sub.max_concurrent == 1)

    # Init: NOT(D) per bit (pre-compute ~D once)
    if is_linear:
        init_lat = 2 * n * cfg.t_AAP                        # sequential across units
    else:
        init_lat = 2 * cfg.t_AAP                             # parallel
    init_eng = 2 * n * sub.e_aap

    # Per iteration costs
    # 1. Shift left by 1: n-1 adjacent hops (all parallel or sequential)
    if is_linear:
        shift_lat = (n - 1) * sub.t_hop
    else:
        shift_lat = sub.t_hop                                # all parallel
    shift_eng = (n - 1) * sub.e_hop

    # 2. Sign broadcast: chain from MSB to all units (always sequential)
    sign_lat = (n - 1) * sub.t_hop
    sign_eng = (n - 1) * sub.e_hop

    # 3. Conditional complement: MUX(sign, D, ~D) = 14 AAPs per bit
    #    MUX = (s AND a) OR (~s AND b) = NOT(2) + AND(4) + AND(4) + OR(4)
    if is_linear:
        mux_lat = 14 * n * cfg.t_AAP                        # sequential across units
    else:
        mux_lat = 14 * cfg.t_AAP                             # parallel
    mux_eng = 14 * n * sub.e_aap

    # 4. SKS addition
    # (add_lat, add_eng already computed)

    # 5. Set quotient bit: 1 AAP
    qbit_lat = cfg.t_AAP
    qbit_eng = sub.e_aap

    iter_lat = shift_lat + sign_lat + mux_lat + add_lat + qbit_lat
    iter_eng = shift_eng + sign_eng + mux_eng + add_eng + qbit_eng

    lat = init_lat + n * iter_lat
    eng = init_eng + n * iter_eng
    return _make_result(sub, "division", n_bits, lat, eng, num_elements)


# Evaluation runner

OPERATIONS = {
    "addition": eval_addition,
    "subtraction": eval_subtraction,
    "multiplication": eval_multiplication,
    "division": eval_division,
}


def evaluate_all(cfg_proteus: DRAMConfig, cfg_pipedram: DRAMConfig,
                 precisions: List[int], num_elements: int,
                 operations: List[str] = None) -> List[EvalResult]:
    """Run all substrates x operations x precisions."""
    substrates = build_substrates(cfg_proteus, cfg_pipedram)
    if operations is None:
        operations = list(OPERATIONS.keys())

    results = []
    for op_name in operations:
        eval_fn = OPERATIONS[op_name]
        for n in precisions:
            for sub in substrates:
                cfg = cfg_proteus if sub.is_proteus else cfg_pipedram
                results.append(eval_fn(cfg, sub, n, num_elements))
    return results
