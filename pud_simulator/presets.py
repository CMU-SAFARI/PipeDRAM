"""
DRAM configuration presets.
"""

from .config import DRAMConfig


def pipedram_config() -> DRAMConfig:
    """Configuration matching the PipeDRAM evaluation parameters."""
    return DRAMConfig(
        # Organization
        num_channels             = 1,
        num_ranks_per_channel    = 1,
        num_banks_per_rank       = 32,        # banks per chip
        num_subarrays_per_bank   = 32,
        num_rows_per_subarray    = 1024,      # default; adjust if needed
        row_size_bytes           = 8192,      # 8 KB logical row
        num_mats_per_subarray    = 128,       # 8KB / 512 bits = 128 mats
        num_chips_per_rank       = 4,         # 128 mats / 32 mats-per-chip
        dq_width_per_chip        = 16,        # x16 assumed (64-bit bus / 4 chips)
        prefetch_bits_per_chip   = 128,

        # Subarray groups
        num_d_group_rows         = 1006,
        num_c_group_rows         = 2,
        num_b_group_rows         = 16,

        # Timing (ns)
        t_AAP   = 49.0,
        t_AP    = 42.0,
        t_RAS   = 32.0,
        t_RCD   = 15.0,
        t_RP    = 15.0,
        t_CK    = 0.833,

        # Energy (nJ)
        e_AAP   = 0.871,
        e_AP    = 0.871,
        e_transp_pj_per_bit = 8.44,

        # PipeDRAM inter-mat interconnect (FIGARO-style)
        t_reloc   = 0.57,   # ns, charge-sharing per hff_width bits
        hff_width = 4,       # bits per relocation step

        # LISA inter-subarray transfer (DDR4-2400 scaled)
        # Decomposition: 3*t_RAS + 2*t_RBM + 2*t_RP
        # DDR4-2400: 3*32 + 2*8 + 2*15 = 142 ns
        # Energy: DDR3 90 nJ * (1.2/1.5)^2 = 57.6 nJ
        t_LISA    = 142.0,   # ns, LISA 1-hop (from LISA HPCA'16 Table 1, scaled)
        e_LISA    = 57.6,    # nJ, per LISA transfer (V^2 scaled from DDR3)
    )


def proteus_config() -> DRAMConfig:
    """Configuration for the Proteus evaluation.

    Same DRAM organization as PipeDRAM, but Proteus distributes bits
    across subarrays (not mats) and uses LISA for inter-subarray carry.
    Operates at full-row granularity (65,536 SIMD lanes).
    Requires 64 subarrays for bit-level parallelism (up to 64-bit precision).
    """
    return DRAMConfig(
        # Organization
        num_channels             = 1,
        num_ranks_per_channel    = 1,
        num_banks_per_rank       = 32,
        num_subarrays_per_bank   = 64,       # Proteus needs >= bit-precision subarrays
        num_rows_per_subarray    = 1024,
        row_size_bytes           = 8192,
        num_mats_per_subarray    = 128,
        num_chips_per_rank       = 4,
        dq_width_per_chip        = 16,
        prefetch_bits_per_chip   = 128,

        # Subarray groups
        num_d_group_rows         = 1006,
        num_c_group_rows         = 2,
        num_b_group_rows         = 16,

        # Timing (ns)
        t_AAP   = 49.0,
        t_AP    = 42.0,
        t_RAS   = 32.0,
        t_RCD   = 15.0,
        t_RP    = 15.0,
        t_CK    = 0.833,

        # Energy (nJ)
        e_AAP   = 0.871,
        e_AP    = 0.871,
        e_transp_pj_per_bit = 8.44,

        # PipeDRAM inter-mat (not used by Proteus, but keep consistent)
        t_reloc   = 0.57,
        hff_width = 4,

        # LISA inter-subarray transfer (DDR4-2400 scaled)
        t_LISA    = 142.0,
        e_LISA    = 57.6,
    )


def simdram_config() -> DRAMConfig:
    """Original SIMDRAM ASPLOS'21 Table 2 configuration."""
    return DRAMConfig()  # all defaults
