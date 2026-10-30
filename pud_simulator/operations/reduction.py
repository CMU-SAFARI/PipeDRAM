"""Reduction operations: and/or/xor across N input bits, comparisons to zero."""

from ..primitives import Primitive, OpType, UProgram
from . import register


def _aap(dst, src, bit_idx=-1, phase="", comment=""):
    return Primitive(OpType.AAP, dst, src, bit_idx, phase, comment)

def _ap(rows, bit_idx=-1, phase="", comment=""):
    return Primitive(OpType.AP, rows, "", bit_idx, phase, comment)


@register("and_reduction")
def and_reduction(n: int) -> UProgram:
    """Y = AND of N input bits  (5*floor(n/2)+2 primitives)"""
    P = []
    if n % 2 == 0:
        P.append(_aap("T3", "C1", phase="init", comment="accum ← 1"))
        j = 0
    else:
        P.append(_aap("T3", "SRC1[0]", phase="init", comment="accum ← input[0]"))
        j = 1

    while j < n - 1:
        P.append(_aap("T0,T1,T2", "C0",           j, "loop"))
        P.append(_aap("DCC1",     f"SRC1[{j}]",   j, "loop"))
        P.append(_aap("T2",       "DCC1,T0,T3",   j, "loop", "AND(input[j], accum)"))
        P.append(_aap("DCC0",     f"SRC1[{j+1}]", j, "loop"))
        P.append(_aap("T3",       "DCC0,T1,T2",   j, "loop", "AND(input[j+1], result)"))
        j += 2

    P.append(_aap("DST[0]", "T3", phase="epilogue", comment="store result"))
    return UProgram("AND Reduction", "Y = AND of N inputs", 1, 1, n, P)


@register("or_reduction")
def or_reduction(n: int) -> UProgram:
    """Y = OR of N input bits  (5*floor(n/2)+2 primitives)"""
    P = []
    if n % 2 == 0:
        P.append(_aap("T3", "C0", phase="init", comment="accum ← 0"))
        j = 0
    else:
        P.append(_aap("T3", "SRC1[0]", phase="init"))
        j = 1

    while j < n - 1:
        P.append(_aap("T0,T1,T2", "C1",           j, "loop"))
        P.append(_aap("DCC1",     f"SRC1[{j}]",   j, "loop"))
        P.append(_aap("T2",       "DCC1,T0,T3",   j, "loop", "OR step"))
        P.append(_aap("DCC0",     f"SRC1[{j+1}]", j, "loop"))
        P.append(_aap("T3",       "DCC0,T1,T2",   j, "loop", "OR step"))
        j += 2

    P.append(_aap("DST[0]", "T3", phase="epilogue"))
    return UProgram("OR Reduction", "Y = OR of N inputs", 1, 1, n, P)


@register("xor_reduction")
def xor_reduction(n: int) -> UProgram:
    """Y = XOR of N input bits  (6*floor(n/2)+2 primitives)"""
    P = []
    if n % 2 == 0:
        P.append(_aap("DCC1N,T1", "C0", phase="init"))
        j = 0
    else:
        P.append(_aap("DCC1N,T1", "SRC1[0]", phase="init"))
        j = 1

    while j < n - 1:
        P.append(_aap("DCC0N,T0", f"SRC1[{j}]",   j, "loop"))
        P.append(_aap("T2",       f"SRC1[{j+1}]",  j, "loop"))
        P.append(_aap("T3",       "DCC0,T1,T2",    j, "loop", "coalesced"))
        P.append(_aap("T2",       "DCC1,T0,T3",    j, "loop", "coalesced"))
        P.append(_aap("DCC0N",    f"SRC1[{j+1}]",  j, "loop"))
        P.append(_aap("DCC1N",    "DCC0,T1,T2",    j, "loop", "coalesced XOR"))
        j += 2

    P.append(_aap("DST[0]", "T1", phase="epilogue"))
    return UProgram("XOR Reduction", "Y = XOR of N inputs", 1, 1, n, P)


@register("greater_equal_0")
def greater_equal_0(n: int) -> UProgram:
    """S = (A >= 0)  (2 primitives, constant)"""
    P = [
        _aap("DCC0N", f"SRC1[{n-1}]", phase="init", comment="negate MSB"),
        _aap("DST[0]", "DCC0",         phase="epilogue", comment="store NOT(MSB)"),
    ]
    return UProgram("Greater-Equal Zero", "S = (A >= 0)", 1, 1, n, P)


@register("greater_than_0")
def greater_than_0(n: int) -> UProgram:
    """S = (A > 0)  (OR-reduce lower bits, AND with NOT MSB)"""
    P = []
    # OR-reduction of bits 0..n-2
    if n % 2 == 1:
        P.append(_aap("T3", "C0", phase="init"))
        j = 0
    else:
        P.append(_aap("T3", "SRC1[0]", phase="init"))
        j = 1

    while j < n - 2:
        P.append(_aap("T0,T1,T2", "C1",          j, "loop"))
        P.append(_aap("DCC1",     f"SRC1[{j}]",  j, "loop"))
        P.append(_aap("T2",       "DCC1,T0,T3",  j, "loop"))
        P.append(_aap("DCC0",     f"SRC1[{j+1}]",j, "loop"))
        P.append(_aap("T3",       "DCC0,T1,T2",  j, "loop"))
        j += 2

    # AND with negated MSB
    P.append(_aap("DCC1N", f"SRC1[{j}]", phase="epilogue", comment="NOT MSB"))
    P.append(_aap("T0",    "C0",          phase="epilogue"))
    P.append(_aap("DST[0]","DCC1,T0,T3",  phase="epilogue", comment="AND(or_reduced, NOT MSB)"))
    return UProgram("Greater-Than Zero", "S = (A > 0)", 1, 1, n, P)
