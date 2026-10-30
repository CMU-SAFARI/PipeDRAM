"""
PUD Primitive representation.

Each primitive is a single AAP (row copy) or AP (majority) command
sequence with full operand information.  This is the fundamental unit
of execution that both SIMDRAM and PipeDRAM simulators operate on.

For DDG construction (PipeDRAM), each primitive records:
  - Which B-group / C-group / D-group rows it reads from
  - Which rows it writes to
  - The bit-position index it belongs to in the serial loop
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List


class OpType(str, Enum):
    AAP = "AAP"   # row copy: ACT src -> ACT dst -> PRE
    AP  = "AP"    # majority: TRA(row0, row1, row2) -> PRE


# B-group row symbolic names
# Multi-row aliases map to the set of physical rows that are
# simultaneously activated by the special B-group row decoder.

BGROUP_ALIASES = {
    "T0":           ["T0"],
    "T1":           ["T1"],
    "T2":           ["T2"],
    "T3":           ["T3"],
    "DCC0":         ["DCC0"],
    "DCC0N":        ["DCC0"],    # negated output of same DCC
    "DCC1":         ["DCC1"],
    "DCC1N":        ["DCC1"],
    "T0,T1,T2":     ["T0", "T1", "T2"],
    "T1,T2,T3":     ["T1", "T2", "T3"],
    "T2,T3":        ["T2", "T3"],
    "T0,T3":        ["T0", "T3"],
    "DCC0N,T0":     ["DCC0", "T0"],
    "DCC1N,T1":     ["DCC1", "T1"],
    "DCC0,T1,T2":   ["DCC0", "T1", "T2"],
    "DCC1,T0,T3":   ["DCC1", "T0", "T3"],
}

# C-group constants
C_ROWS = {"C0", "C1"}

# All B-group physical rows
BGROUP_ROWS = {"T0", "T1", "T2", "T3", "DCC0", "DCC1"}


def resolve_rows(name: str) -> List[str]:
    """Resolve a symbolic row name to the list of physical rows."""
    if name in BGROUP_ALIASES:
        return BGROUP_ALIASES[name]
    # D-group rows (SRC1[j], DST[j], etc.) or C-group
    return [name]


def is_negated(name: str) -> bool:
    """True if the operand uses the negated DCC wordline."""
    return "N" in name.split(",")[0] and "DCC" in name


@dataclass
class Primitive:
    """A single PUD primitive (AAP or AP) with full operand tracking.

    Attributes
    ----------
    op : OpType
        AAP (row copy) or AP (majority / TRA).
    dst : str
        Symbolic destination.  For AAP this is the copy target(s).
        For AP this is the three-row TRA address.
    src : str
        Symbolic source (AAP only; empty for AP).
    bit_idx : int
        Bit-position in the serial loop (0 = LSB).
        -1 for INIT / EPILOGUE primitives.
    phase : str
        Execution phase: "init", "loop", "epilogue", or custom.
    comment : str
        Human-readable description.
    """
    op:      OpType
    dst:     str = ""
    src:     str = ""
    bit_idx: int = -1
    phase:   str = ""
    comment: str = ""

    # Read / write sets (for DDG construction)

    @property
    def write_rows(self) -> List[str]:
        """Physical rows written by this primitive."""
        if self.op == OpType.AAP:
            rows = resolve_rows(self.dst)
            # Coalesced AP+AAP: if src is a 3-row alias, the AP part
            # also writes to those src rows before the copy.
            src_rows = resolve_rows(self.src)
            if len(src_rows) == 3 and all(r in BGROUP_ROWS for r in src_rows):
                return list(set(rows + src_rows))
            return rows
        else:
            # AP: TRA overwrites all three rows
            return resolve_rows(self.dst)

    @property
    def read_rows(self) -> List[str]:
        """Physical rows read by this primitive."""
        if self.op == OpType.AAP:
            src_rows = resolve_rows(self.src)
            # Coalesced: the AP part reads those 3 rows
            if len(src_rows) == 3 and all(r in BGROUP_ROWS for r in src_rows):
                return src_rows
            return src_rows
        else:
            # AP: reads all three rows to compute MAJ
            return resolve_rows(self.dst)

    def __repr__(self):
        if self.op == OpType.AAP:
            s = f"AAP {self.dst} ← {self.src}"
        else:
            s = f"AP  {self.dst}"
        if self.bit_idx >= 0:
            s += f"  [bit {self.bit_idx}]"
        if self.comment:
            s += f"  # {self.comment}"
        return s


@dataclass
class UProgram:
    """A complete uProgram: the ordered sequence of primitives for one
    PUD operation at a specific bit-precision.

    The simulator iterates ``self.primitives`` one by one.
    """
    name:        str = "unnamed"
    description: str = ""
    num_inputs:  int = 2
    num_outputs: int = 1
    bit_precision: int = 0

    primitives: List[Primitive] = field(default_factory=list)

    @property
    def num_aaps(self) -> int:
        return sum(1 for p in self.primitives if p.op == OpType.AAP)

    @property
    def num_aps(self) -> int:
        return sum(1 for p in self.primitives if p.op == OpType.AP)

    @property
    def total_primitives(self) -> int:
        return len(self.primitives)

    def summary(self) -> str:
        return (f"{self.name} (n={self.bit_precision}): "
                f"{self.num_aaps} AAP + {self.num_aps} AP = "
                f"{self.total_primitives} primitives")
