"""Comparison operations: greater, greater_equal, equal, min, max."""

from ..primitives import Primitive, OpType, UProgram
from . import register


def _aap(dst, src, bit_idx=-1, phase="", comment=""):
    return Primitive(OpType.AAP, dst, src, bit_idx, phase, comment)

def _ap(rows, bit_idx=-1, phase="", comment=""):
    return Primitive(OpType.AP, rows, "", bit_idx, phase, comment)


@register("greater")
def greater(n: int) -> UProgram:
    """S = (A > B)  (3n+2 primitives, linear)"""
    P = []
    P.append(_aap("T2", "C0", phase="init", comment="result ← 0"))
    for j in range(n):
        P.append(_aap("T1",    f"SRC1[{j}]", j, "loop", "copy A[j]"))
        P.append(_aap("DCC0N", f"SRC2[{j}]", j, "loop", "copy NOT B[j]"))
        P.append(_ap ("DCC0,T1,T2",          j, "loop", "MAJ(NOT B, A, accum)"))
    P.append(_aap("DST[0]", "T2", phase="epilogue", comment="store result"))
    return UProgram("Greater-Than", "S = (A > B)", 2, 1, n, P)


@register("greater_equal")
def greater_equal(n: int) -> UProgram:
    """S = (A >= B)  (3n+2 primitives, linear)"""
    P = []
    P.append(_aap("T2", "C1", phase="init", comment="result ← 1"))
    for j in range(n):
        P.append(_aap("T1",    f"SRC1[{j}]", j, "loop"))
        P.append(_aap("DCC0N", f"SRC2[{j}]", j, "loop"))
        P.append(_ap ("DCC0,T1,T2",          j, "loop"))
    P.append(_aap("DST[0]", "T2", phase="epilogue"))
    return UProgram("Greater-Equal", "S = (A >= B)", 2, 1, n, P)


@register("equal")
def equal(n: int) -> UProgram:
    """S = (A == B)  (4n+3 primitives, linear)"""
    P = []
    P.append(_aap("T2,T3", "C1", phase="init", comment="accum ← 1"))
    for j in range(n):
        P.append(_aap("DCC0N,T0", f"SRC1[{j}]", j, "loop"))
        P.append(_aap("DCC1N,T1", f"SRC2[{j}]", j, "loop"))
        P.append(_ap ("DCC0,T1,T2",              j, "loop", "XNOR bit (part 1)"))
        P.append(_ap ("DCC1,T0,T3",              j, "loop", "XNOR bit (part 2)"))
    P.append(_aap("T1",     "C0",       phase="epilogue"))
    P.append(_aap("DST[0]", "T0,T1,T2", phase="epilogue", comment="AND reduce"))
    return UProgram("Equality", "S = (A == B)", 2, 1, n, P)


def _greater_than_primitives(n: int, src1_fmt="SRC1[{}]", src2_fmt="SRC2[{}]") -> list:
    """Shared primitives for the 'greater-than' comparison, used by min/max."""
    P = []
    P.append(_aap("T2", "C0", phase="init", comment="gt accum ← 0"))
    for j in range(n):
        P.append(_aap("T1",    src1_fmt.format(j), j, "loop"))
        P.append(_aap("DCC0N", src2_fmt.format(j), j, "loop"))
        P.append(_ap ("DCC0,T1,T2",                j, "loop"))
    return P


def _less_than_primitives(n: int) -> list:
    """Primitives for 'less-than', swaps operand order."""
    return _greater_than_primitives(n, src1_fmt="SRC2[{}]", src2_fmt="SRC1[{}]")


def _if_then_else_primitives(n: int, sel_name="S") -> list:
    """Shared if-then-else primitives used by min/max."""
    P = []
    for j in range(n):
        P.append(_aap("DCC0N,T0", sel_name,       j, "mux"))
        P.append(_aap("DCC1N,T1", "C1",           j, "mux"))
        P.append(_aap("T3",       f"SRC1[{j}]",   j, "mux"))
        P.append(_aap("T2",       f"SRC2[{j}]",   j, "mux"))
        P.append(_ap ("T0,T1,T2",                 j, "mux", "MUX part 1"))
        P.append(_aap("T3",       "DCC0,T1,T2",   j, "mux", "coalesced"))
        P.append(_aap(f"DST[{j}]","DCC1,T0,T3",   j, "mux", "coalesced → out"))
    return P


@register("min")
def minimum(n: int) -> UProgram:
    """S = min(A, B)  (10n+2 primitives, linear)"""
    P = []
    # Phase 1: s = (A < B), using "less than" order
    P += _less_than_primitives(n)
    P.append(_aap("S", "T2", phase="phase1", comment="store comparison"))
    # Phase 2: output = s ? A : B
    P += _if_then_else_primitives(n, "S")
    return UProgram("Minimum", "S = min(A, B)", 2, 1, n, P)


@register("max")
def maximum(n: int) -> UProgram:
    """S = max(A, B)  (10n+2 primitives, linear)"""
    P = []
    # Phase 1: s = (A > B)
    P += _greater_than_primitives(n)
    P.append(_aap("S", "T2", phase="phase1", comment="store comparison"))
    # Phase 2: output = s ? A : B
    P += _if_then_else_primitives(n, "S")
    return UProgram("Maximum", "S = max(A, B)", 2, 1, n, P)
