"""
PUD Simulator: performance evaluation framework for Processing-Using-DRAM
=========================================================================

Supports:
  - SIMDRAM (bit-serial SIMD, vertical layout)
  - Proteus (bit-level parallelism across subarrays, vertical layout)
  - PipeDRAM (pipelined bit-serial/bit-parallel, horizontal layout)

The simulator walks each uProgram primitive (AAP/AP) individually,
preserving operand-level information needed for dependency analysis
and pipeline scheduling.
"""

__version__ = "0.4.0"

from .ddg import build_ddg, DependencyGraph  # noqa: F401
from .pipedram_simulator import PipeDRAMSimulator  # noqa: F401
from .proteus_simulator import ProteusSimulator  # noqa: F401
