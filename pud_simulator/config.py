"""
DRAM system configuration: organization, timing, and energy parameters.
"""

import json
import math
from dataclasses import dataclass, asdict


@dataclass
class DRAMConfig:
    """DRAM organization, timing, and energy parameters.

    Defaults correspond to DDR4-2400 consistent with the
    SIMDRAM evaluation methodology (Table 2 of ASPLOS'21).
    """

    # Organization
    num_channels:             int = 1
    num_ranks_per_channel:    int = 1
    num_banks_per_rank:       int = 16
    num_subarrays_per_bank:   int = 64
    num_rows_per_subarray:    int = 1024
    row_size_bytes:           int = 8192      # 8 KB
    num_mats_per_subarray:    int = 16
    num_chips_per_rank:       int = 4
    dq_width_per_chip:        int = 16        # x16 chips
    prefetch_bits_per_chip:   int = 128       # 8n prefetch for DDR4 x16

    # Subarray row groups (Ambit / SIMDRAM)
    num_d_group_rows:         int = 1006
    num_c_group_rows:         int = 2         # C0 (all-0) and C1 (all-1)
    num_b_group_rows:         int = 16        # T0-T3, DCC0/1 + aliases

    # Timing (ns)
    t_AAP:  float = 46.16    # latency of one AAP command sequence
    t_AP:   float = 46.16    # latency of one AP  command sequence
    t_RAS:  float = 32.0     # row activation latency
    t_RCD:  float = 14.16    # activate-to-column-access delay
    t_RP:   float = 14.16    # precharge latency
    t_CK:   float = 0.833    # clock period (DDR4-2400)

    # Energy (nJ per full-row operation)
    e_AAP:  float = 2.21     # energy per AAP
    e_AP:   float = 1.52     # energy per AP  (+22% for TRA)

    # PipeDRAM inter-mat interconnect (FIGARO-style)
    t_reloc:    float = 0.57   # ns, charge-sharing latency per hff_width bits
    hff_width:  int   = 4      # bits transferred per relocation step

    # LISA inter-subarray transfer (DDR4-2400 scaled)
    # Standard LISA 1-hop: 3*t_RAS + 2*t_RBM + 2*t_RP
    # DDR4-2400: 3*32 + 2*8 + 2*15 = 142 ns
    # Energy: DDR3 90 nJ * (1.2/1.5)^2 = 57.6 nJ
    t_LISA:     float = 142.0  # ns, standard LISA 1-hop (source row not yet sensed)
    t_RBM:      float = 8.0    # ns, row buffer move latency (LISA component)
    e_LISA:     float = 57.6   # nJ, energy per LISA transfer (full-row operation)

    # Transposition energy
    e_transp_pj_per_bit: float = 8.44  # pJ per bit for DRAM read/write

    # Derived
    @property
    def simd_lanes(self) -> int:
        """Bits per DRAM row = SIMD lanes per subarray."""
        return self.row_size_bytes * 8

    @property
    def mat_columns(self) -> int:
        """Columns (bits) per DRAM mat."""
        return self.simd_lanes // self.num_mats_per_subarray

    @property
    def mats_per_chip(self) -> int:
        """Mats per chip = mats_per_subarray / chips_per_rank."""
        return self.num_mats_per_subarray // self.num_chips_per_rank

    @property
    def pipedram_simd_lanes(self) -> int:
        """Effective SIMD width for PipeDRAM: mat_columns x num_chips."""
        return self.mat_columns * self.num_chips_per_rank

    @property
    def t_inter_mat_copy(self) -> float:
        """Inter-mat carry propagation latency (ns), standard copy.

        FIGARO-style mechanism: activate source and destination rows
        in parallel (t_RAS), transfer mat_columns bits through a
        narrow (hff_width) interconnect at t_reloc per step, then
        precharge (t_RP).
        """
        num_steps = math.ceil(self.mat_columns / self.hff_width)
        return self.t_RAS + num_steps * self.t_reloc + self.t_RP

    @property
    def t_inter_mat_copy_sensed(self) -> float:
        """Inter-mat copy latency when source mat data is already sensed (ns).

        After LISA deposits data into the copying subarray, the source
        mat's sense amplifiers already hold the data, so the source
        activation (t_RAS) is skipped.  No precharge is needed
        mid-pipeline: the data stays in the sense amplifiers for the next
        step (LISA back).

        HFF relay only: ceil(mat_columns / hff_width) * t_reloc.
        """
        num_steps = math.ceil(self.mat_columns / self.hff_width)
        return num_steps * self.t_reloc

    @property
    def t_LISA_sensed(self) -> float:
        """LISA row buffer movement for transient carry data (ns).

        In the hierarchical carry pipeline, the carry is never written to
        a DRAM row; it stays in the sense amplifiers:

        Standard LISA (persistent data): 3*t_RAS + 2*t_RBM + 2*t_RP = 142 ns
          - ACT source, RBM, ACT destination to write back, PRE both sides

        Carry transfer: t_RBM = 8 ns
          - Source data already sensed (from MAJ or HFF)
          - No destination ACT (data stays in the destination SA)
          - No precharge (the next step operates on the SA directly)

        The single t_RP at the end of the round trip is accounted for in
        t_hier_round_trip.
        """
        return self.t_RBM

    @property
    def t_hier_round_trip(self) -> float:
        """Full hierarchical carry round-trip latency (ns).

        Three steps on sense-amplifier-resident data plus a single final
        precharge (values for presets.pipedram_config()):

        1. LISA out  (compute SA -> copy SA):  t_RBM       =  8 ns
        2. HFF       (mat j -> mat j+1):       relay       = 73 ns
        3. LISA back (copy SA -> compute SA):  t_RBM       =  8 ns
        4. Final PRE (release copy SA):        t_RP        = 15 ns
                                               ---------------------
                                               Total       ~104 ns
        """
        return (self.t_LISA_sensed           # LISA out (RBM only)
                + self.t_inter_mat_copy_sensed  # HFF relay (no ACT/PRE)
                + self.t_LISA_sensed           # LISA back (RBM only)
                + self.t_RP)                   # single final precharge

    @property
    def e_LISA_mat(self) -> float:
        """LISA energy when only one mat is activated (fine-grained PipeDRAM).

        PipeDRAM uses fine-grained DRAM activation: only the target mat's
        sense amplifiers are activated, not the full row.  Energy scales
        proportionally to the fraction of the row activated.
        """
        return self.e_LISA / self.num_mats_per_subarray

    @property
    def bus_bandwidth_GBps(self) -> float:
        """Peak DRAM bus bandwidth in GB/s.

        DDR transfers on both clock edges, so effective rate = 2 / t_CK.
        Bus width = num_chips x dq_width_per_chip bits.
        """
        bus_width_bytes = (self.num_chips_per_rank * self.dq_width_per_chip) // 8
        transfers_per_ns = 2.0 / self.t_CK   # DDR double data rate
        return bus_width_bytes * transfers_per_ns  # bytes/ns = GB/s

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @classmethod
    def from_json(cls, path: str) -> "DRAMConfig":
        with open(path) as f:
            return cls(**json.load(f))

    def summary(self) -> str:
        lines = [
            f"Channels={self.num_channels}  Ranks={self.num_ranks_per_channel}  "
            f"Banks={self.num_banks_per_rank}",
            f"Subarrays/bank={self.num_subarrays_per_bank}  "
            f"Rows/sub={self.num_rows_per_subarray}  "
            f"Row={self.row_size_bytes}B ({self.simd_lanes:,} bits)",
            f"Mats/sub={self.num_mats_per_subarray}  "
            f"Cols/mat={self.mat_columns:,}  "
            f"Chips={self.num_chips_per_rank}×{self.dq_width_per_chip}",
            f"t_AAP={self.t_AAP}ns  t_AP={self.t_AP}ns  "
            f"e_AAP={self.e_AAP}nJ  e_AP={self.e_AP}nJ",
        ]
        return "\n".join(lines)
