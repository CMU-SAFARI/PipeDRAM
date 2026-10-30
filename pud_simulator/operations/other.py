"""Other operations: abs, relu, if_then_else, bitweave, bitcount."""

import math
from ..primitives import Primitive, OpType, UProgram
from . import register


def _aap(dst, src, bit_idx=-1, phase="", comment=""):
    return Primitive(OpType.AAP, dst, src, bit_idx, phase, comment)

def _ap(rows, bit_idx=-1, phase="", comment=""):
    return Primitive(OpType.AP, rows, "", bit_idx, phase, comment)


@register("abs")
def absolute_value(n: int) -> UProgram:
    """S = |A|  (10n-1 primitives, linear)"""
    P = []
    # Phase 1: compute carry chain for conditional negate
    P.append(_aap("T0", f"SRC1[{n-1}]", phase="init", comment="sign ← MSB"))
    P.append(_aap("CARRY[0]", "T0",      phase="init"))

    for j in range(n - 1):
        P.append(_aap("T3",    "C0",                j, f"carry_{j}"))
        P.append(_aap("DCC1N", f"SRC1[{j}]",        j, f"carry_{j}"))
        P.append(_aap(f"CARRY[{j+1}]", "DCC1,T0,T3",j, f"carry_{j}", "carry propagation"))

    # Phase 2: conditional negate + mux
    for j in range(n):
        P.append(_aap("DCC0N,T0", f"SRC1[{n-1}]",  j, "mux", "sign"))
        P.append(_aap("DCC1N,T1", f"CARRY[{j}]",   j, "mux"))
        P.append(_aap("T3",       f"SRC1[{j}]",    j, "mux"))
        P.append(_aap("T3",       "DCC0,T1,T2",    j, "mux", "coalesced"))
        P.append(_aap("T2",       "DCC1,T0,T3",    j, "mux", "coalesced"))
        P.append(_aap("DCC0N",    f"SRC1[{j}]",    j, "mux"))
        P.append(_aap(f"DST[{j}]","DCC0,T1,T2",    j, "mux", "coalesced → out"))

    return UProgram("Absolute Value", "S = |A|", 1, 1, n, P)


@register("relu")
def relu(n: int) -> UProgram:
    """S = max(A, 0) = ReLU(A)  (3n + ((n-1)%2) primitives, linear)"""
    P = []
    P.append(_aap("DCC0N", f"SRC1[{n-1}]", phase="init", comment="negate MSB"))
    P.append(_aap("SIGN",  "DCC0",          phase="init", comment="sign = NOT(MSB)"))

    j = 0
    while j < n - 2:
        P.append(_aap("T0,T1,T2", "SIGN",        j, "loop", "mask"))
        P.append(_aap("T2,T3",    "C0",          j, "loop"))
        P.append(_aap("DCC0",     f"SRC1[{j}]",  j, "loop"))
        P.append(_aap(f"DST[{j}]","DCC0,T1,T2",  j, "loop", "coalesced AND"))
        P.append(_aap("DCC1",     f"SRC1[{j+1}]",j, "loop"))
        P.append(_aap(f"DST[{j+1}]","DCC1,T0,T3",j, "loop", "coalesced AND (pair)"))
        j += 2

    if j == n - 2:
        P.append(_aap("T0", "SIGN",          j, "tail"))
        P.append(_aap("T1", "C0",            j, "tail"))
        P.append(_aap("T2", f"SRC1[{j}]",    j, "tail"))
        P.append(_aap(f"DST[{j}]", "T0,T1,T2", j, "tail", "second-to-MSB"))

    P.append(_aap(f"DST[{n-1}]", "C0", phase="epilogue", comment="MSB ← 0"))
    return UProgram("ReLU", "S = max(A, 0)", 1, 1, n, P)


@register("if_then_else")
def if_then_else(n: int) -> UProgram:
    """S = sel ? A : B  (7n primitives, linear)"""
    P = []
    for j in range(n):
        P.append(_aap("DCC0N,T0", "SEL",          j, "loop", "copy select"))
        P.append(_aap("DCC1N,T1", "C1",           j, "loop"))
        P.append(_aap("T3",       f"SRC1[{j}]",   j, "loop"))
        P.append(_aap("T2",       f"SRC2[{j}]",   j, "loop"))
        P.append(_ap ("T0,T1,T2",                 j, "loop", "MUX part 1"))
        P.append(_aap("T3",       "DCC0,T1,T2",   j, "loop", "coalesced"))
        P.append(_aap(f"DST[{j}]","DCC1,T0,T3",   j, "loop", "coalesced → out"))
    return UProgram("If-Then-Else", "S = sel ? A : B", 3, 1, n, P)


@register("bitweave")
def bitweave(n: int) -> UProgram:
    """BitWeaving predicate scan  (4n+4 primitives, linear)"""
    P = []
    P.append(_aap("DCC1",     "C1", phase="init"))
    P.append(_aap("T0,T1,T2", "C1", phase="init"))

    for j in range(n):
        P.append(_aap("DCC0N,T0", f"SRC1[{j}]", j, "loop", "load column bit"))
        # The second AAP depends on the comparison constants (data-dependent
        # in the C code). We model the worst case: 1 AAP always present.
        P.append(_aap("T2,T3",    "C0",          j, "loop", "predicate setup"))
        P.append(_ap ("DCC1,T0,T3",              j, "loop", "compare step 1"))
        P.append(_ap ("DCC0,T1,T2",              j, "loop", "compare step 2"))

    P.append(_aap("T1",     "C0",       phase="epilogue"))
    P.append(_aap("DST[0]", "T1,T2,T3", phase="epilogue", comment="store result"))
    return UProgram("BitWeaving Scan", "BitWeaving predicate", 1, 1, n, P)


@register("bitcount")
def bitcount(n: int) -> UProgram:
    """S = popcount(A)  (recursive divide-and-conquer, ~8n)"""

    def _foo(x):
        x |= x >> 1; x |= x >> 2; x |= x >> 4
        x |= x >> 8; x |= x >> 16
        return x >> 1

    P = []
    _id = [0]  # mutable counter for unique naming

    def _recurse(input_names, size, output_prefix):
        if size == 0:
            return [], 0
        if size == 1:
            P.append(_aap(f"{output_prefix}[0]", input_names[0],
                          phase="bitcount", comment="copy 1-bit"))
            return [f"{output_prefix}[0]"], 1
        if size <= 3:
            v3 = "C0" if size == 2 else input_names[2]
            _id[0] += 1
            tag = f"fa{_id[0]}"
            P.append(_aap("T0,T1,T2", input_names[0],       phase=tag))
            P.append(_aap("T2,T3",    input_names[1],       phase=tag))
            P.append(_aap("DCC1",     v3,                   phase=tag))
            P.append(_aap(f"{output_prefix}[1]", "DCC1,T0,T3", phase=tag, comment="carry"))
            P.append(_aap("T0,T3",    "DCC1N",              phase=tag))
            P.append(_ap ("T0,T1,T2",                       phase=tag, comment="sum"))
            P.append(_aap("T1",       v3,                   phase=tag))
            P.append(_aap(f"{output_prefix}[0]", "T1,T2,T3",phase=tag, comment="sum bit"))
            return [f"{output_prefix}[0]", f"{output_prefix}[1]"], 2

        left = _foo(size)
        right = size - left - 1
        _id[0] += 1
        lpfx = f"{output_prefix}_L{_id[0]}"
        _id[0] += 1
        rpfx = f"{output_prefix}_R{_id[0]}"

        louts, b1 = _recurse(input_names[:left], left, lpfx)
        routs, b2 = _recurse(input_names[left+1:left+1+right], right, rpfx)

        # Addition phase: add left + right results with carry-in
        carry_in = input_names[left]
        _id[0] += 1
        atag = f"add{_id[0]}"

        P.append(_aap("DCC1", carry_in, phase=atag, comment="load carry_in"))
        for i in range(b1):
            v2 = routs[i] if i < b2 else "C0"
            P.append(_aap("T0,T1,T2", "DCC1",       phase=atag))
            P.append(_aap("T2,T3",    louts[i],     phase=atag))
            P.append(_aap("DCC1",     v2,           phase=atag))
            P.append(_ap ("DCC1,T0,T3",             phase=atag, comment="carry"))
            P.append(_aap("T0,T3",    "DCC1N",      phase=atag))
            P.append(_ap ("T0,T1,T2",               phase=atag, comment="sum"))
            P.append(_aap("T1",       v2,           phase=atag))
            P.append(_aap(louts[i],   "T1,T2,T3",   phase=atag, comment="store sum"))
        P.append(_aap(f"{output_prefix}[{b1}]", "DCC1", phase=atag, comment="carry out"))

        result_names = louts + [f"{output_prefix}[{b1}]"]
        return result_names, 1 + b1

    input_names = [f"SRC1[{i}]" for i in range(n)]
    _recurse(input_names, n, "DST")

    return UProgram("Bit Count", "S = popcount(A)", 1, 1, n, P)
