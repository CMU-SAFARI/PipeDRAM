#!/usr/bin/env python3
"""
SKS Majority-Logic Parallel-Prefix Adder Simulator
====================================================
Simulates the Ambit + LISA + SALP uProgram for the n-bit Simplified
Kogge-Stone (SKS) majority-logic adder at instruction granularity.

Defaults:
  - tFAW : 45 ns (DDR3 typical)         -> set via --t_faw
  - SALP : unlimited by default          -> set via --max_concurrent_sa (0=unlimited)
  - TRA energy : separate from normal AAP energy (22% higher per Ambit paper)
  - Correctness: full bit-level functional simulation, enabled by default

tFAW scope: per-bank (counts all ACTs across all subarrays in the same bank).
Each AAP issues 2 ACTs; each AP issues 1 ACT; each LISA hop issues 1 ACT.

Throughput definition: n-bit additions per second.
Since each uProgram execution operates on `num_cols` bits in SIMD fashion,
one uProgram invocation computes `num_cols` independent additions in parallel.
Batching: if the number of additions exceeds `num_cols`, multiple batches
are needed; throughput accounts for this automatically.

Usage examples:
  python sks_simulator.py --n_bits 8
  python sks_simulator.py --n_bits 16 --num_cols 16384 --tfaw_enabled
  python sks_simulator.py --n_bits 8 --max_concurrent_sa 2
  python sks_simulator.py --n_bits 32 --t_aap_normal 49 --e_aap_tra 4.0
"""

import math
import argparse
import random
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Set, Callable, Any
from enum import Enum, auto
from collections import defaultdict


# 1. Configuration

@dataclass
class DRAMConfig:
    """
    All timing, energy, and structural parameters.
    Defaults are DDR3-1600 Ambit values from the paper
    (Seshadri et al., MICRO-50 2017, Table 3 / Section 5.3).
    """
    # Timing (ns)
    t_aap_normal: float = 49.0   # 2-wordline AAP  (copy / NOT step)
    t_aap_tra:    float = 49.0   # 3-wordline TRA  (B12 majority)
    t_ap:         float = 35.0   # Single AP       (1 ACT + precharge)
    t_lisa:       float = 15.0   # Per-hop inter-subarray LISA transfer
    t_faw:        float = 45.0   # tFAW rolling window (ns)

    # Energy per operation (nJ; per full row-width activation)
    e_aap_normal: float = 3.2    # 2-wordline AAP
    e_aap_tra:    float = 4.0    # 3-wordline TRA (+22% for extra wordline per Ambit)
    e_ap:         float = 1.6    # AP
    e_lisa:       float = 1.0    # Per-hop LISA

    # DRAM structure
    num_subarrays:     int  = 64     # Total available subarrays (must be >= n_bits)
    num_rows:          int  = 1024   # Rows per subarray
    num_cols:          int  = 8192   # Columns per row = SIMD width (bits)

    # Concurrency
    # 0 = unlimited (Full-SALP); N > 0 = max N subarrays active simultaneously
    max_concurrent_sa: int  = 0

    # tFAW
    tfaw_enabled:      bool = True
    tfaw_act_count:    int  = 4      # Standard: 4 ACTs per tFAW window

    def validate(self, n_bits: int):
        assert n_bits >= 2, "n_bits must be >= 2"
        assert (n_bits & (n_bits - 1)) == 0, "n_bits must be a power of 2"
        assert self.num_subarrays >= n_bits, \
            f"Need num_subarrays ({self.num_subarrays}) >= n_bits ({n_bits})"
        assert self.num_rows >= 32, "num_rows too small"
        assert self.num_cols >= 1, "num_cols must be >= 1"


# 2. Instruction model

class IType(Enum):
    AAP_NORMAL = auto()  # 2-wordline AAP: row copy or DCC NOT-capture step
    AAP_TRA    = auto()  # 3-wordline TRA: B12 triple-row activation (MAJ)
    AP         = auto()  # Single activate-precharge (rarely used here)
    LISA       = auto()  # One inter-subarray hop (adjacent subarrays only)


@dataclass
class Instr:
    """
    One primitive DRAM instruction in the uProgram.

    Dependency tracking:
      `deps` holds the instruction IDs (iid) that must complete before this
      instruction can start. The scheduler enforces this ordering.

    Functional simulation:
      `func` is a closure over (sa_id, row names) that updates the
      `SubarrayState` dict when called during functional simulation.
    """
    itype:    IType
    sa:       int                          # subarray (source SA for LISA)
    dst_sa:   int    = -1                  # destination SA (LISA only)
    desc:     str    = ""
    iid:      int    = -1                  # set by MicroProgram builder
    deps:     List[int] = field(default_factory=list)

    # Functional sim: callable(states: List[Dict[str,int]]) -> None
    func: Any = field(default=None, repr=False)

    # Scheduled fields (set by Scheduler)
    start_ns: float = 0.0
    end_ns:   float = 0.0

    def latency(self, cfg: DRAMConfig) -> float:
        return {
            IType.AAP_NORMAL: cfg.t_aap_normal,
            IType.AAP_TRA:    cfg.t_aap_tra,
            IType.AP:         cfg.t_ap,
            IType.LISA:       cfg.t_lisa,
        }[self.itype]

    def energy(self, cfg: DRAMConfig) -> float:
        return {
            IType.AAP_NORMAL: cfg.e_aap_normal,
            IType.AAP_TRA:    cfg.e_aap_tra,
            IType.AP:         cfg.e_ap,
            IType.LISA:       cfg.e_lisa,
        }[self.itype]

    def num_acts(self) -> int:
        """Number of ACTIVATE commands this instruction issues (for tFAW)."""
        return {
            IType.AAP_NORMAL: 2,
            IType.AAP_TRA:    2,
            IType.AP:         1,
            IType.LISA:       1,
        }[self.itype]


# 3. uProgram builder

class MicroProgram:
    """
    Builds the instruction-level SKS majority-logic adder uProgram for n bits.

    The builder tracks data-flow dependencies automatically: whenever a row
    is written, the writer's iid is recorded.  Any subsequent instruction
    that reads that row adds this iid to its deps list, creating a precise
    data-flow dependency graph.

    Physical model:
      - Each bit position i maps to subarray SA_i.
      - Each subarray has the standard Ambit row layout:
          T0, T1, T2, T3          B-group scratch rows
          DCC0_d, DCC0_n          DCC0 d-wordline / n-wordline
          DCC1_d, DCC1_n          DCC1 d-wordline / n-wordline
          C0_const                C-group: all-zeros
          C1_const                C-group: all-ones
          A, B                    D-group: input operands
          C, S, NC, INN, ...      D-group: computed rows
    """

    def __init__(self, n_bits: int, cfg: DRAMConfig):
        cfg.validate(n_bits)
        self.n:    int        = n_bits
        self.cfg:  DRAMConfig = cfg
        self.instrs: List[Instr] = []
        self._next_iid: int   = 0

        # (sa_id, row_name) -> iid of last instruction that wrote this row
        # -1 means "written before program start (input data or constant)"
        self._last_writer: Dict[Tuple[int, str], int] = {}

        # carry_loc[bit_pos] = (sa_id, row_name)
        self.carry_loc: Dict[int, Tuple[int, str]] = {}

        # group_gen[(hi, lo, variant)] = (sa_id, row_name)
        # variant in {'a', 'b'}
        self._gg: Dict[Tuple[int, int, str], Tuple[int, str]] = {}

        self._build()

    # Low-level instruction emitters

    def _next_id(self) -> int:
        iid = self._next_iid
        self._next_iid += 1
        return iid

    def _last_write(self, sa: int, row: str) -> int:
        """Return iid of last write to (sa, row), or -1 (pre-program data)."""
        return self._last_writer.get((sa, row), -1)

    def _set_write(self, sa: int, row: str, iid: int):
        self._last_writer[(sa, row)] = iid

    def _emit(self, instr: Instr) -> int:
        instr.iid = self._next_id()
        # Deduplicate and remove sentinel -1 from deps
        instr.deps = sorted({d for d in instr.deps if d >= 0})
        self.instrs.append(instr)
        return instr.iid

    # Primitive operations

    def _copy(self, sa: int, src: str, dst: str,
              extra_deps: Tuple[int, ...] = ()) -> int:
        """
        AAP(src, dst): copy src row -> dst row within same subarray.
        Ambit: AAP(D_src, D_dst); the sense amplifier drives src data onto dst.
        """
        deps = [self._last_write(sa, src), *extra_deps]

        def _f(states: List[Dict[str, int]],
               _sa=sa, _src=src, _dst=dst):
            states[_sa][_dst] = states[_sa].get(_src, 0)

        iid = self._emit(Instr(IType.AAP_NORMAL, sa=sa,
                               desc=f"SA{sa}: COPY {src}->{dst}",
                               deps=deps, func=_f))
        self._set_write(sa, dst, iid)
        return iid

    def _not(self, sa: int, src: str, dst: str,
             extra_deps: Tuple[int, ...] = ()) -> int:
        """
        NOT(src -> dst): 2 AAPs via DCC0.
          Step 1: AAP(src, B5): activate src, then activate DCC0 n-wordline
                                  -> DCC0 capacitor captures ~src
          Step 2: AAP(B4, dst): activate DCC0 d-wordline -> dst <- DCC0 = ~src
        """
        dcc = '__DCC0__'
        dep_src = self._last_write(sa, src)
        deps1 = [dep_src, *extra_deps]

        def _f1(states: List[Dict[str, int]],
                _sa=sa, _src=src, _dcc=dcc):
            states[_sa][_dcc] = ~states[_sa].get(_src, 0)

        iid1 = self._emit(Instr(IType.AAP_NORMAL, sa=sa,
                                desc=f"SA{sa}: NOT step1 ~{src}->DCC0",
                                deps=deps1, func=_f1))
        self._set_write(sa, dcc, iid1)

        deps2 = [iid1]

        def _f2(states: List[Dict[str, int]],
                _sa=sa, _dcc=dcc, _dst=dst):
            states[_sa][_dst] = states[_sa].get(_dcc, 0)

        iid2 = self._emit(Instr(IType.AAP_NORMAL, sa=sa,
                                desc=f"SA{sa}: NOT step2 DCC0->{dst}",
                                deps=deps2, func=_f2))
        self._set_write(sa, dst, iid2)
        return iid2

    def _maj(self, sa: int, x: str, y: str, z: str, dst: str,
             extra_deps: Tuple[int, ...] = ()) -> int:
        """
        MAJ(x, y, z -> dst): 4 AAPs.
          3x AAP_NORMAL: copy x->T0, y->T1, z->T2 (staging into B-group)
          1x AAP_TRA:    TRA B12 -> MAJ(T0,T1,T2) copied to dst

        The three copies are independent and can run in any order
        (they share the same B-group staging area T0/T1/T2 within this
        subarray, but since they write different rows they are scheduled
        to the same SA sequentially by the per-SA availability tracker).
        """
        dep_x = self._last_write(sa, x)
        dep_y = self._last_write(sa, y)
        dep_z = self._last_write(sa, z)
        base  = [dep_x, dep_y, dep_z, *extra_deps]

        def _fx(states, _sa=sa, _x=x):
            states[_sa]['T0'] = states[_sa].get(_x, 0)

        def _fy(states, _sa=sa, _y=y):
            states[_sa]['T1'] = states[_sa].get(_y, 0)

        def _fz(states, _sa=sa, _z=z):
            states[_sa]['T2'] = states[_sa].get(_z, 0)

        iid_t0 = self._emit(Instr(IType.AAP_NORMAL, sa=sa,
                                  desc=f"SA{sa}: T0←{x}", deps=base, func=_fx))
        self._set_write(sa, 'T0', iid_t0)

        iid_t1 = self._emit(Instr(IType.AAP_NORMAL, sa=sa,
                                  desc=f"SA{sa}: T1←{y}", deps=base, func=_fy))
        self._set_write(sa, 'T1', iid_t1)

        iid_t2 = self._emit(Instr(IType.AAP_NORMAL, sa=sa,
                                  desc=f"SA{sa}: T2←{z}", deps=base, func=_fz))
        self._set_write(sa, 'T2', iid_t2)

        def _ftra(states, _sa=sa, _dst=dst):
            t0 = states[_sa].get('T0', 0)
            t1 = states[_sa].get('T1', 0)
            t2 = states[_sa].get('T2', 0)
            states[_sa][_dst] = (t0 & t1) | (t1 & t2) | (t0 & t2)

        # TRA must wait for all three T rows to be written
        iid_tra = self._emit(Instr(IType.AAP_TRA, sa=sa,
                                   desc=f"SA{sa}: TRA MAJ({x},{y},{z})->{dst}",
                                   deps=[iid_t0, iid_t1, iid_t2], func=_ftra))
        self._set_write(sa, dst, iid_tra)
        return iid_tra

    def _lisa_hop(self, src_sa: int, src_row: str,
                  dst_sa: int, dst_row: str,
                  extra_deps: Tuple[int, ...] = ()) -> int:
        """
        Single LISA hop between adjacent subarrays (|dst_sa - src_sa| == 1).
        LISA reads src_sa.src_row without destroying it (RowClone-FPM semantics).
        """
        assert abs(dst_sa - src_sa) == 1, \
            f"LISA hop must be between adjacent SAs: {src_sa} → {dst_sa}"

        dep_src = self._last_write(src_sa, src_row)
        deps = [dep_src, *extra_deps]

        def _f(states, _ssrc=src_sa, _srow=src_row, _sdst=dst_sa, _drow=dst_row):
            states[_sdst][_drow] = states[_ssrc].get(_srow, 0)

        iid = self._emit(Instr(IType.LISA, sa=src_sa, dst_sa=dst_sa,
                               desc=f"LISA SA{src_sa}.{src_row}→SA{dst_sa}.{dst_row}",
                               deps=deps, func=_f))
        self._set_write(dst_sa, dst_row, iid)
        return iid

    def _lisa_chain(self, src_sa: int, src_row: str,
                    dst_sa: int, dst_row: str,
                    extra_deps: Tuple[int, ...] = ()) -> int:
        """
        Multi-hop LISA chain: src_sa -> ... -> dst_sa, hopping through every
        intermediate subarray.  Each hop is a single LISA instruction.
        Transit rows are named automatically.
        Returns iid of the final-hop instruction.
        """
        if src_sa == dst_sa:
            return self._last_write(src_sa, src_row)

        step = 1 if dst_sa > src_sa else -1
        cur_sa  = src_sa
        cur_row = src_row
        last_iid = -1
        first = True

        while cur_sa != dst_sa:
            nxt_sa = cur_sa + step
            is_last_hop = (nxt_sa == dst_sa)
            # Use the final name only on the last hop; use transit names in between
            nxt_row = dst_row if is_last_hop else f"_tr_{src_sa}_{src_row}_at{nxt_sa}"

            hop_extra = extra_deps if first else ()
            first = False
            last_iid = self._lisa_hop(cur_sa, cur_row, nxt_sa, nxt_row,
                                      extra_deps=hop_extra)
            cur_sa  = nxt_sa
            cur_row = nxt_row

        return last_iid

    def _ensure_in(self, target_sa: int, src_sa: int, src_row: str,
                   local_name: str, extra_deps: Tuple[int, ...] = ()) -> str:
        """
        Ensure src_sa.src_row is available in target_sa as local_name.
        - If already there (same SA), returns src_row directly.
        - If already transferred (last_write exists), returns local_name.
        - Otherwise, emits a LISA chain and returns local_name.
        """
        if target_sa == src_sa:
            return src_row
        if self._last_write(target_sa, local_name) >= 0:
            return local_name
        self._lisa_chain(src_sa, src_row, target_sa, local_name,
                         extra_deps=extra_deps)
        return local_name

    # uProgram construction

    def _build(self):
        """
        Construct the full n-bit SKS uProgram:
          Phase 0  : c0, leaf group generates
          Phase 1  : c1
          Phase 2+ : SKS prefix tree (carries c3, c5, ..., c_{n-1})
          Phase 4  : Sum bits s0 ... s_{n-1}
        """
        n = self.n

        # c0 = AND(a0, b0) = MAJ(a0, b0, 0)
        # C0_const is the permanent all-zeros C-group row (pre-initialised).
        self._maj(0, 'A', 'B', 'C0_const', 'C')
        self.carry_loc[0] = (0, 'C')

        if n == 2:
            # Degenerate: 2-bit adder: s0, s1 only, no prefix tree needed.
            self._build_sum_bits()
            return

        # Leaf group generates
        # For each odd SA_p (p >= 3):
        #   LISA a[p-1], b[p-1] from SA_{p-1} (1 hop, always adjacent)
        #   G_{p:p-1,a} = MAJ(a_p, b_p, a_{p-1})   in SA_p
        #   G_{p:p-1,b} = MAJ(a_p, b_p, b_{p-1})   in SA_p
        for p in range(3, n, 2):
            self._lisa_chain(p - 1, 'A', p, f'a{p-1}_in')
            self._lisa_chain(p - 1, 'B', p, f'b{p-1}_in')
            self._maj(p, 'A', 'B', f'a{p-1}_in', f'G{p}_{p-1}a')
            self._maj(p, 'A', 'B', f'b{p-1}_in', f'G{p}_{p-1}b')
            self._gg[(p, p-1, 'a')] = (p, f'G{p}_{p-1}a')
            self._gg[(p, p-1, 'b')] = (p, f'G{p}_{p-1}b')

        # c1
        self._lisa_chain(0, 'C', 1, 'c0_in')
        self._maj(1, 'A', 'B', 'c0_in', 'C')
        self.carry_loc[1] = (1, 'C')

        # SKS prefix tree
        # The odd positions {3, 5, ..., n-1} map to "reduced indices"
        # r = (p-1)/2 in {1, 2, ..., n/2-1}.  The tree is a standard
        # Kogge-Stone tree of depth log2(n)-1 operating on these r values.
        #
        # At KS level l (l = 1 ... log2(n)-1):
        #   Each reduced position r receives from r_src = r - 2^(l-1)
        #   iff r_src >= 1  (i.e., r >= 2^(l-1) + 1).
        #
        # The actual (physical) positions are p = 2r+1, q = 2*r_src+1.
        # Span of upper group at level l: (p, p - 2^l + 1), i.e., (hi, lo+1)
        # Span of lower group at level l: (q, q - 2^l + 1), i.e., (q, lo+1)
        # where lo = q - 2^l = p - 2^{l+1}.
        #
        # The carry root is reached when the lower group spans down to position 1
        # (i.e., when the lower group's carry is c1 or c0).

        num_levels = int(math.log2(n)) - 1  # depth of the half-tree

        for level in range(1, num_levels + 1):
            half_span = 1 << (level - 1)   # 2^(l-1) in reduced units -> 2^l actual
            full_span = half_span * 2       # full span in actual bit positions

            # Iterate over all odd positions p that participate at this level.
            # A position p participates if its reduced index r = (p-1)/2 >= 2^(l-1)+1,
            # i.e., p >= 2*(2^(l-1)+1)+1 = 2^l + 3.
            for p in range(3, n, 2):
                r = (p - 1) // 2
                # r_src = r - half_span; skip only if truly below the tree (r_src < 0)
                if r < half_span:
                    continue

                # q = physical position of the right-neighbour source
                q = p - 2 * half_span           # e.g. l=1: q=p-2; l=2: q=p-4
                if q < 3:
                    # q < 3 means the lower group reaches below the SKS tree;
                    # use c1 as the lower carry.
                    pass

                # Determine upper and lower group spans
                # Upper group (already computed):  G_{p : p-2^l+1, a/b}
                # Lower group (already computed):  G_{q : q-2^l+1, a/b}
                #   or the carry c_{q} if the lower group is just one pair leaf.

                upper_hi = p
                upper_lo = p - full_span + 1  # inclusive lower bound of upper span

                lower_hi = q
                lower_lo = q - full_span + 1

                # For level 1 the groups are the leaf pair-generators:
                #   upper = G_{p:p-1,a/b},   lower = G_{q:q-1,a/b}
                # For level l > 1 the groups were computed in the previous iteration.

                key_ua = (upper_hi, upper_lo, 'a')
                key_ub = (upper_hi, upper_lo, 'b')
                key_la = (lower_hi, lower_lo, 'a')
                key_lb = (lower_hi, lower_lo, 'b')

                if key_ua not in self._gg or key_ub not in self._gg:
                    continue  # upper group not ready (shouldn't happen in correct tree)

                # Is this the carry-root level for position p?
                # The carry root is produced when the lower group represents the
                # very lowest part of the prefix tree, i.e., when lower_lo <= 1.
                # At that point we combine the upper group with c_{q} (= c1 or c0).
                is_root = (lower_lo <= 1)

                sa_dst = p  # always store result in the highest SA involved

                # Bring upper group signals into sa_dst
                (sa_ua, row_ua) = self._gg[key_ua]
                (sa_ub, row_ub) = self._gg[key_ub]
                row_ua_l = self._ensure_in(sa_dst, sa_ua, row_ua,
                                           f'gg{upper_hi}_{upper_lo}a')
                row_ub_l = self._ensure_in(sa_dst, sa_ub, row_ub,
                                           f'gg{upper_hi}_{upper_lo}b')

                if is_root:
                    # Carry root: c_p = MAJ(G_upper_a, G_upper_b, c_q)
                    # c_q is a previously computed carry at physical position q.
                    c_q_pos = q  # c_q = carry at position q (must already be in carry_loc)
                    if c_q_pos not in self.carry_loc:
                        # fallback: use c1 (should not happen for correct tree)
                        c_q_pos = 1
                    (sa_c, row_c) = self.carry_loc[c_q_pos]
                    row_c_l = self._ensure_in(sa_dst, sa_c, row_c, f'c{c_q_pos}_in')
                    self._maj(sa_dst, row_ua_l, row_ub_l, row_c_l, 'C')
                    self.carry_loc[p] = (sa_dst, 'C')
                else:
                    # Intermediate: combine upper and lower groups into a wider group
                    if key_la not in self._gg or key_lb not in self._gg:
                        continue  # lower group not ready

                    (sa_la, row_la) = self._gg[key_la]
                    (sa_lb, row_lb) = self._gg[key_lb]
                    row_la_l = self._ensure_in(sa_dst, sa_la, row_la,
                                               f'gg{lower_hi}_{lower_lo}a')
                    row_lb_l = self._ensure_in(sa_dst, sa_lb, row_lb,
                                               f'gg{lower_hi}_{lower_lo}b')

                    new_hi  = upper_hi
                    new_lo  = lower_lo
                    row_a   = f'G{new_hi}_{new_lo}a'
                    row_b   = f'G{new_hi}_{new_lo}b'
                    # G_{new_hi:new_lo,a} = MAJ(G_upper_a, G_upper_b, G_lower_a)  [Eq.16]
                    # G_{new_hi:new_lo,b} = MAJ(G_upper_a, G_upper_b, G_lower_b)  [Eq.17]
                    self._maj(sa_dst, row_ua_l, row_ub_l, row_la_l, row_a)
                    self._maj(sa_dst, row_ua_l, row_ub_l, row_lb_l, row_b)
                    self._gg[(new_hi, new_lo, 'a')] = (sa_dst, row_a)
                    self._gg[(new_hi, new_lo, 'b')] = (sa_dst, row_b)

        # Sum bits
        self._build_sum_bits()

    def _build_sum_bits(self):
        """
        Compute all n sum bits using the pair formulation (Eqs. 25, 26, 28),
        with the inverter on c_i (Fig. 6a):
          Single bit i:   s_i = MAJ(~c_i, c_{i-1}, MAJ(a_i, b_i, ~c_i))
          Pair upper i+1: c_{i+1} = MAJ(a_{i+1}, b_{i+1}, c_i)
                          s_{i+1} = MAJ(~c_{i+1}, c_i, MAJ(a_{i+1}, b_{i+1}, ~c_{i+1}))
          Pair lower i:   s_i = MAJ(~c_i, MAJ(a_{i-1}, b_{i-1}, c_{i-2}), MAJ(a_i, b_i, ~c_i))

        Layout:
          s0         : single (c_{-1} = 0 -> reduces to AND(~c0, MAJ(a0,b0,~c0)))
          (s1, s2)   : pair
          (s3, s4)   : pair
          ...
          (s_{n-3}, s_{n-2}) : pair
          s_{n-1}    : single
        """
        n = self.n

        # s0 : single bit, c_{-1} = 0
        # s0 = MAJ(~c0, 0, MAJ(a0,b0,~c0)) = AND(~c0, MAJ(a0,b0,~c0))
        self._not(0, 'C', 'NC')
        self._maj(0, 'A', 'B', 'NC', 'INN')
        # MAJ(~c0, INN, 0) = ~c0 AND INN
        self._maj(0, 'NC', 'INN', 'C0_const', 'S')

        # Pairs (s_i, s_{i+1}) for i = 1, 3, ..., n-3
        for i in range(1, n - 1, 2):
            sa_i  = i          # SA for s_i  (pair lower)
            sa_i1 = i + 1      # SA for s_{i+1} (pair upper)

            (c_i_sa, c_i_row) = self.carry_loc[i]  # c_i is always an odd carry

            # Pair upper: s_{i+1}
            # Bring c_i into SA_{i+1}
            c_i_l_i1 = self._ensure_in(sa_i1, c_i_sa, c_i_row, f'c{i}_in')
            # c_{i+1} = MAJ(a_{i+1}, b_{i+1}, c_i)
            self._maj(sa_i1, 'A', 'B', c_i_l_i1, 'C_even')
            # ~c_{i+1}
            self._not(sa_i1, 'C_even', 'NC_even')
            # INN = MAJ(a_{i+1}, b_{i+1}, ~c_{i+1})
            self._maj(sa_i1, 'A', 'B', 'NC_even', 'INN')
            # s_{i+1} = MAJ(~c_{i+1}, c_i, INN)
            self._maj(sa_i1, 'NC_even', c_i_l_i1, 'INN', 'S')

            # Pair lower: s_i
            # ~c_i
            self._not(sa_i, 'C', 'NC')
            # INN = MAJ(a_i, b_i, ~c_i)
            self._maj(sa_i, 'A', 'B', 'NC', 'INN')
            # c_{i-1} inline = MAJ(a_{i-1}, b_{i-1}, c_{i-2})
            if i == 1:
                # c_0 is already computed; c_{-1} = 0 means c_0 = AND(a0,b0)
                # Bring c0 into SA_1
                (c0_sa, c0_row) = self.carry_loc[0]
                c_prev_l = self._ensure_in(sa_i, c0_sa, c0_row, 'c0_in_sum')
            else:
                # c_{i-2} is a previously computed (odd) carry
                (c_im2_sa, c_im2_row) = self.carry_loc[i - 2]
                c_im2_l = self._ensure_in(sa_i, c_im2_sa, c_im2_row, f'c{i-2}_in_sum')
                # a_{i-1}, b_{i-1} needed in SA_i
                a_im1_l = self._ensure_in(sa_i, i - 1, 'A', f'a{i-1}_in_sum')
                b_im1_l = self._ensure_in(sa_i, i - 1, 'B', f'b{i-1}_in_sum')
                # c_{i-1} inline
                self._maj(sa_i, a_im1_l, b_im1_l, c_im2_l, 'C_prev_inl')
                c_prev_l = 'C_prev_inl'

            # s_i = MAJ(~c_i, c_{i-1}, INN)
            self._maj(sa_i, 'NC', c_prev_l, 'INN', 'S')

        # s_{n-1} : single bit
        # c_{n-2} was computed as 'C_even' in SA_{n-2} during the last pair.
        # c_{n-1} is in SA_{n-1} = carry_loc[n-1].
        c_nm2_l = self._ensure_in(n - 1, n - 2, 'C_even', f'c{n-2}_in')
        self._not(n - 1, 'C', 'NC')
        self._maj(n - 1, 'A', 'B', 'NC', 'INN')
        # s_{n-1} = MAJ(~c_{n-1}, c_{n-2}, INN)
        self._maj(n - 1, 'NC', c_nm2_l, 'INN', 'S')

    def program_stats(self) -> Dict[str, int]:
        counts = defaultdict(int)
        for instr in self.instrs:
            counts[instr.itype.name] += 1
        counts['TOTAL'] = len(self.instrs)
        return dict(counts)


# 4. Scheduler

class Scheduler:
    """
    Instruction-level event-driven scheduler with:
      - Data-flow dependency enforcement
      - Per-subarray serialisation (one instruction active at a time per SA)
      - SALP concurrency limit (max_concurrent_sa)
      - tFAW: per-bank rolling window; counts all ACTIVATEs across all SAs
    """

    def __init__(self, program: MicroProgram, cfg: DRAMConfig):
        self.prog = program
        self.cfg  = cfg
        self.n    = program.n

    def schedule(self) -> List[Instr]:
        """
        Assign start_ns / end_ns to every instruction.
        Returns the (same) list of instructions after scheduling.
        """
        cfg    = self.cfg
        instrs = self.prog.instrs
        N      = len(instrs)

        # finish_time[iid] = when instruction iid finishes
        finish_time: List[float] = [0.0] * N

        # Per-subarray: earliest time the subarray is free
        sa_free: Dict[int, float] = defaultdict(float)

        # tFAW: start times of all issued ACTs (bank-wide)
        act_history: List[float] = []  # sorted list of ACT timestamps

        # SALP: end times of issued instructions; the number of entries
        # still in the future at time t is the number of active subarrays
        active_ends: List[float] = []

        _EPS = 1e-6  # sub-picosecond epsilon to break boundary ties

        def earliest_start_tfaw(t_want: float, n_acts: int) -> float:
            """
            Return the earliest t >= t_want at which n_acts new ACTs can be
            issued without violating tFAW (no more than tfaw_act_count ACTs in
            any rolling window of length t_faw).

            Window definition: [t - t_faw, t), half-open on the right.
            Strategy: find the (n_evict)-th oldest ACT that must leave the
            window, then advance t until it does.
            """
            if not cfg.tfaw_enabled or n_acts == 0:
                return t_want
            t = t_want
            for _ in range(len(act_history) + n_acts + 10):
                recent = sorted(a for a in act_history
                                if t - cfg.t_faw <= a < t)
                if len(recent) + n_acts <= cfg.tfaw_act_count:
                    return t
                # Must evict this many of the oldest entries
                n_evict = len(recent) + n_acts - cfg.tfaw_act_count
                # Advance t so that the n_evict-th oldest ACT drops out
                t = recent[n_evict - 1] + cfg.t_faw + _EPS
            return t  # should not reach here

        def earliest_start_salp(t_want: float, lat: float) -> float:
            """
            Return the earliest t >= t_want such that fewer than
            max_concurrent_sa subarrays are active at time t.
            """
            if cfg.max_concurrent_sa <= 0:
                return t_want  # unlimited Full-SALP
            t = t_want
            for _ in range(len(active_ends) + 10):
                overlap = sum(1 for e in active_ends if e > t)
                if overlap < cfg.max_concurrent_sa:
                    return t
                future = [e for e in active_ends if e > t]
                if not future:
                    return t
                t = min(future) + _EPS
            return t

        # The builder emits instructions in dependency (topological) order
        for instr in instrs:
            iid = instr.iid
            lat = instr.latency(cfg)

            # 1. All data dependencies must be satisfied
            t_dep = max((finish_time[d] for d in instr.deps), default=0.0)

            # 2. The subarray must be free
            #    For LISA: both source and destination subarrays must be free
            if instr.itype == IType.LISA:
                t_sa = max(sa_free[instr.sa], sa_free[instr.dst_sa], t_dep)
            else:
                t_sa = max(sa_free[instr.sa], t_dep)

            # 3. tFAW constraint
            t_faw_ok = earliest_start_tfaw(t_sa, instr.num_acts())

            # 4. SALP concurrency constraint
            t_salp = earliest_start_salp(t_faw_ok, lat)

            start = t_salp
            end   = start + lat

            instr.start_ns = start
            instr.end_ns   = end
            finish_time[iid] = end

            # Update per-SA availability
            sa_free[instr.sa] = end
            if instr.itype == IType.LISA:
                sa_free[instr.dst_sa] = end

            # Record ACTs in tFAW history (n_acts ACTs issued at `start`)
            for _ in range(instr.num_acts()):
                act_history.append(start)
            act_history.sort()  # keep sorted for window queries

            # Track SALP active intervals
            active_ends.append(end)

        return instrs


# 5. Functional simulator

class FunctionalSim:
    """
    Replays the uProgram in dependency order, updating a per-subarray
    state dict to verify the addition result.

    State model: each row stores a Python int (treated as a SIMD bit vector
    of width `num_cols`).  A value of 0 represents all-zero bits; -1 (all ones
    in two's complement) represents all-one bits; intermediate values represent
    mixed patterns.  The mask `(1 << num_cols) - 1` keeps values in range.
    """

    def __init__(self, n: int, num_cols: int):
        self.n        = n
        self.num_cols = num_cols
        self.mask     = (1 << num_cols) - 1
        self.states: List[Dict[str, int]] = [{} for _ in range(n)]

    def init_inputs(self, a_vals: List[int], b_vals: List[int]):
        """
        Load operand bits into the subarray state.
        a_vals[i] / b_vals[i] are `num_cols`-wide bit vectors for bit position i.
        Row 'C0_const' = 0, 'C1_const' = -1 (all ones) in every subarray.
        """
        assert len(a_vals) == self.n and len(b_vals) == self.n
        mask = self.mask
        for i in range(self.n):
            self.states[i] = {
                'A':        a_vals[i] & mask,
                'B':        b_vals[i] & mask,
                'C0_const': 0,
                'C1_const': mask,
            }

    def run(self, instrs: List[Instr]):
        """Execute all instructions in iid order (= topological order)."""
        for instr in sorted(instrs, key=lambda x: x.iid):
            if instr.func is not None:
                instr.func(self.states)
            # Mask all written rows to num_cols bits after each instruction
            # (handles NOT which produces Python's negative two's-complement ints)
            sa = instr.sa
            mask = self.mask
            for row, val in self.states[sa].items():
                self.states[sa][row] = val & mask
            if instr.itype == IType.LISA and instr.dst_sa >= 0:
                dsa = instr.dst_sa
                for row, val in self.states[dsa].items():
                    self.states[dsa][row] = val & mask

    def read_result(self) -> List[int]:
        """Return computed sum bit vectors [s0, s1, ..., s_{n-1}]."""
        return [self.states[i].get('S', 0) & self.mask for i in range(self.n)]

    def verify(self, a_int_list: List[int], b_int_list: List[int]) -> bool:
        """
        For each column j in 0..num_cols-1, extract bit j of each a_val / b_val,
        compute the expected sum, and compare with the S rows.
        Returns True if all columns match.
        """
        num_cols = self.num_cols
        n        = self.n
        s_rows   = self.read_result()
        ok       = True
        for j in range(num_cols):
            a_word = sum(((self.states[i].get('A', 0) >> j) & 1) << i for i in range(n))
            b_word = sum(((self.states[i].get('B', 0) >> j) & 1) << i for i in range(n))
            expected = (a_word + b_word) & ((1 << n) - 1)  # n-bit sum (no overflow)
            got      = sum(((s_rows[i] >> j) & 1) << i for i in range(n))
            if expected != got:
                print(f"  [FAIL] column {j}: a={a_word:#x} b={b_word:#x} "
                      f"expected={expected:#x} got={got:#x}")
                ok = False
        return ok


# 6. Performance / energy analysis

@dataclass
class SimResult:
    n_bits:            int
    num_cols:          int
    latency_ns:        float    # Time from first instruction start to last end
    total_energy_nJ:   float    # Total energy for one uProgram invocation
    icount:            Dict[str, int]
    throughput_Gops:   float    # n-bit additions per second (SIMD-aware)
    correct:           Optional[bool] = None

    def report(self):
        bar = "═" * 60
        print(bar)
        print(f"  SKS In-DRAM Adder Simulation Results")
        print(bar)
        print(f"  Operand width      : {self.n_bits} bits")
        print(f"  SIMD width (cols)  : {self.num_cols} bits")
        print(f"  Latency            : {self.latency_ns:.2f} ns")
        print(f"  Total energy       : {self.total_energy_nJ:.3f} nJ  "
              f"(per {self.num_cols}-wide SIMD row)")
        print(f"  Energy / addition  : {self.total_energy_nJ / self.num_cols * 1e3:.4f} pJ")
        print(f"  Throughput         : {self.throughput_Gops:.4f} GOps/s  "
              f"(n-bit additions/s)")
        print()
        print(f"  Instruction breakdown:")
        for name in ['AAP_NORMAL', 'AAP_TRA', 'AP', 'LISA', 'TOTAL']:
            print(f"    {name:<14}: {self.icount.get(name, 0)}")
        if self.correct is not None:
            status = "PASS ✓" if self.correct else "FAIL ✗"
            print(f"\n  Correctness check  : {status}")
        print(bar)


def analyse(program: MicroProgram, cfg: DRAMConfig,
            verify: bool = True,
            num_verify_cols: int = 256) -> SimResult:
    """
    Run scheduler + optional functional simulation.
    Returns a SimResult with latency, energy, and throughput.

    `num_verify_cols` is the number of columns used for correctness checking
    (can be < num_cols for speed; the pattern repeats).
    """
    # Schedule
    sched = Scheduler(program, cfg)
    instrs = sched.schedule()

    latency_ns  = max(i.end_ns for i in instrs)
    total_energy = sum(i.energy(cfg) for i in instrs)

    # Throughput
    # One uProgram invocation computes `num_cols` independent n-bit additions
    # (one per column).  Throughput = num_cols / latency_ns additions per ns
    # = num_cols / latency_ns * 1e9 additions per second.
    additions_per_invoke = cfg.num_cols
    throughput_Gops = (additions_per_invoke / latency_ns)  # GOps/s (1e9 adds/s)

    icount = program.program_stats()

    # Functional correctness
    correct = None
    if verify:
        n   = program.n
        nc  = min(num_verify_cols, cfg.num_cols)
        fsim = FunctionalSim(n, nc)

        # Generate random test inputs: for each bit position i, a random nc-wide vector
        rng = random.Random(42)
        a_vals = [rng.randrange(0, 1 << nc) for _ in range(n)]
        b_vals = [rng.randrange(0, 1 << nc) for _ in range(n)]
        fsim.init_inputs(a_vals, b_vals)
        fsim.run(instrs)
        correct = fsim.verify(a_vals, b_vals)

    return SimResult(
        n_bits          = program.n,
        num_cols        = cfg.num_cols,
        latency_ns      = latency_ns,
        total_energy_nJ = total_energy,
        icount          = icount,
        throughput_Gops = throughput_Gops,
        correct         = correct,
    )


# 7. Command-line interface

def make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="SKS In-DRAM Majority-Logic Adder Simulator",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    p.add_argument('--n_bits', type=int, default=8,
                   help="Operand width in bits (must be power of 2, >= 2)")

    # Timing
    p.add_argument('--t_aap_normal', type=float, default=49.0,
                   help="2-wordline AAP latency (ns)")
    p.add_argument('--t_aap_tra',    type=float, default=49.0,
                   help="3-wordline TRA (B12) latency (ns)")
    p.add_argument('--t_ap',         type=float, default=35.0,
                   help="Single AP latency (ns)")
    p.add_argument('--t_lisa',       type=float, default=15.0,
                   help="Per-hop LISA transfer latency (ns)")
    p.add_argument('--t_faw',        type=float, default=45.0,
                   help="tFAW rolling window duration (ns)")

    # Energy
    p.add_argument('--e_aap_normal', type=float, default=3.2,
                   help="2-wordline AAP energy (nJ)")
    p.add_argument('--e_aap_tra',    type=float, default=4.0,
                   help="3-wordline TRA energy (nJ)")
    p.add_argument('--e_ap',         type=float, default=1.6,
                   help="AP energy (nJ)")
    p.add_argument('--e_lisa',       type=float, default=1.0,
                   help="Per-hop LISA energy (nJ)")

    # Structure
    p.add_argument('--num_subarrays',     type=int, default=64,
                   help="Number of available subarrays (>= n_bits)")
    p.add_argument('--num_rows',          type=int, default=1024,
                   help="Rows per subarray")
    p.add_argument('--num_cols',          type=int, default=8192,
                   help="Columns per row = SIMD width (bits)")

    # Concurrency
    p.add_argument('--max_concurrent_sa', type=int, default=0,
                   help="SALP: max simultaneous active subarrays "
                        "(0 = unlimited Full-SALP)")

    # tFAW
    p.add_argument('--tfaw_enabled',      action='store_true', default=True,
                   help="Enable tFAW constraint (default: on)")
    p.add_argument('--no_tfaw',           action='store_true', default=False,
                   help="Disable tFAW constraint")
    p.add_argument('--tfaw_act_count',    type=int, default=4,
                   help="Number of ACTs in tFAW window (standard=4)")

    # Misc
    p.add_argument('--no_verify',     action='store_true', default=False,
                   help="Skip functional correctness verification")
    p.add_argument('--verify_cols',   type=int, default=256,
                   help="Number of SIMD columns to use for correctness check")
    p.add_argument('--print_program', action='store_true', default=False,
                   help="Print the full μProgram instruction list")
    p.add_argument('--sweep_nbits',   action='store_true', default=False,
                   help="Sweep n_bits over {4,8,16,32} and print a summary table")

    return p


def run_once(args) -> SimResult:
    cfg = DRAMConfig(
        t_aap_normal      = args.t_aap_normal,
        t_aap_tra         = args.t_aap_tra,
        t_ap              = args.t_ap,
        t_lisa            = args.t_lisa,
        t_faw             = args.t_faw,
        e_aap_normal      = args.e_aap_normal,
        e_aap_tra         = args.e_aap_tra,
        e_ap              = args.e_ap,
        e_lisa            = args.e_lisa,
        num_subarrays     = args.num_subarrays,
        num_rows          = args.num_rows,
        num_cols          = args.num_cols,
        max_concurrent_sa = args.max_concurrent_sa,
        tfaw_enabled      = args.tfaw_enabled and not args.no_tfaw,
        tfaw_act_count    = args.tfaw_act_count,
    )
    prog = MicroProgram(args.n_bits, cfg)

    if args.print_program:
        print(f"\n{'─'*60}")
        print(f"  μProgram for {args.n_bits}-bit SKS adder ({len(prog.instrs)} instructions)")
        print(f"{'─'*60}")
        for instr in prog.instrs:
            deps_str = f"  deps={instr.deps}" if instr.deps else ""
            print(f"  [{instr.iid:4d}] {instr.itype.name:<12} {instr.desc}{deps_str}")
        print()

    result = analyse(prog, cfg,
                     verify=not args.no_verify,
                     num_verify_cols=args.verify_cols)
    return result


def sweep(args):
    widths = [4, 8, 16, 32]
    print(f"\n{'═'*82}")
    print(f"  Sweep: n_bits ∈ {widths}  |  "
          f"tFAW={'ON' if (args.tfaw_enabled and not args.no_tfaw) else 'OFF'}  |  "
          f"SALP={'unlimited' if args.max_concurrent_sa == 0 else args.max_concurrent_sa}")
    print(f"{'─'*82}")
    print(f"  {'n':>5}  {'Instrs':>7}  {'Latency(ns)':>12}  "
          f"{'Energy(nJ)':>12}  {'Throughput(GOps/s)':>20}  {'OK?':>5}")
    print(f"{'─'*82}")
    for n in widths:
        args.n_bits = n
        args.num_subarrays = max(args.num_subarrays, n)
        r = run_once(args)
        ok_str = ("✓" if r.correct else "✗") if r.correct is not None else "—"
        print(f"  {n:>5}  {r.icount['TOTAL']:>7}  {r.latency_ns:>12.2f}  "
              f"{r.total_energy_nJ:>12.3f}  {r.throughput_Gops:>20.4f}  {ok_str:>5}")
    print(f"{'═'*82}\n")


def main():
    parser = make_parser()
    args   = parser.parse_args()

    if args.sweep_nbits:
        sweep(args)
    else:
        result = run_once(args)
        result.report()


if __name__ == '__main__':
    main()
