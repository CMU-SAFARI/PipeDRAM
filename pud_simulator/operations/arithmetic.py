"""Arithmetic operations: addition, subtraction, multiplication, division."""

from ..primitives import Primitive, OpType, UProgram
from . import register


def _aap(dst, src, bit_idx=-1, phase="", comment=""):
    return Primitive(OpType.AAP, dst, src, bit_idx, phase, comment)

def _ap(rows, bit_idx=-1, phase="", comment=""):
    return Primitive(OpType.AP, rows, "", bit_idx, phase, comment)


@register("addition")
def addition(n: int) -> UProgram:
    """S = A + B  (8n+1 primitives, linear)"""
    P = []
    # INIT: carry_in <- 0
    P.append(_aap("DCC1", "C0", phase="init", comment="carry_in ← 0"))

    for j in range(n):
        P.append(_aap("T0,T1,T2", "DCC1",      j, "loop", "copy carry"))
        P.append(_aap("T2,T3",    f"SRC1[{j}]", j, "loop", "copy A[j]"))
        P.append(_aap("DCC1",     f"SRC2[{j}]", j, "loop", "copy B[j]"))
        P.append(_ap ("DCC1,T0,T3",             j, "loop", "MAJ → carry_out"))
        P.append(_aap("T0,T3",    "DCC1N",      j, "loop", "copy NOT(carry_out)"))
        P.append(_ap ("T0,T1,T2",               j, "loop", "MAJ → partial sum"))
        P.append(_aap("T1",       f"SRC2[{j}]", j, "loop", "copy B[j]"))
        P.append(_aap(f"DST[{j}]","T1,T2,T3",   j, "loop", "coalesced → S[j]"))

    up = UProgram("Addition", "S = A + B", 2, 1, n, P)
    return up


@register("subtraction")
def subtraction(n: int) -> UProgram:
    """S = A - B  (8n+1 primitives, linear)"""
    P = []
    P.append(_aap("DCC0", "C1", phase="init", comment="borrow_in ← 1"))

    for j in range(n):
        P.append(_aap("T0,T1,T2",  "DCC0",           j, "loop"))
        P.append(_aap("T2,T3",     f"SRC1[{j}]",     j, "loop", "copy A[j]"))
        P.append(_aap("DCC1N",     f"SRC2[{j}]",     j, "loop", "copy B[j] (negated)"))
        P.append(_aap("DCC0",      "DCC1,T0,T3",     j, "loop", "coalesced MAJ → borrow"))
        P.append(_aap("T0,T3",     "DCC1N",           j, "loop"))
        P.append(_ap ("T0,T1,T2",                     j, "loop", "MAJ → partial diff"))
        P.append(_aap("DCC1N",     f"SRC2[{j}]",     j, "loop", "copy B[j]"))
        P.append(_aap(f"DST[{j}]", "DCC1,T0,T3",     j, "loop", "coalesced → S[j]"))

    return UProgram("Subtraction", "S = A - B", 2, 1, n, P)


@register("multiplication")
def multiplication(n: int) -> UProgram:
    """S = A x B  (11n^2-5n-1 primitives, quadratic)"""
    P = []
    # Phase 1: initialize partial products  output[k] = AND(vals1[0], vals2[k])
    for k in range(n):
        P.append(_aap("T0", f"SRC1[0]",  -1, "init", f"AND init: A[0]"))
        P.append(_aap("T1", f"SRC2[{k}]",-1, "init", f"AND init: B[{k}]"))
        P.append(_aap("T2", "C0",        -1, "init"))
        P.append(_aap(f"DST[{k}]", "T0,T1,T2", -1, "init", f"out[{k}] = A[0]&B[{k}]"))
    P.append(_aap(f"DST[{n}]", "C0", -1, "init", "high bit ← 0"))

    # Phase 2: j=1..n-1, add shifted partial products
    for j in range(1, n):
        P.append(_aap("DCC1", "C0", -1, f"outer_{j}", f"carry ← 0 (outer j={j})"))
        for k in range(n):
            jk = j + k
            P.append(_aap("T0",      f"SRC1[{j}]",  jk, f"inner_{j}_{k}", "copy A[j]"))
            P.append(_aap("T1",      f"SRC2[{k}]",  jk, f"inner_{j}_{k}", "copy B[k]"))
            P.append(_aap("T2",      "C0",          jk, f"inner_{j}_{k}"))
            P.append(_ap ("T0,T1,T2",               jk, f"inner_{j}_{k}", "AND(A[j],B[k])"))
            P.append(_aap("T2,T3",   "DCC1",        jk, f"inner_{j}_{k}", "copy carry"))
            P.append(_aap("DCC1",    f"DST[{jk}]",  jk, f"inner_{j}_{k}", "copy partial sum"))
            P.append(_ap ("DCC1,T0,T3",              jk, f"inner_{j}_{k}", "MAJ → new carry"))
            P.append(_aap("T0,T3",   "DCC1N",       jk, f"inner_{j}_{k}"))
            P.append(_ap ("T0,T1,T2",               jk, f"inner_{j}_{k}", "MAJ → sum bit"))
            P.append(_aap("T1",      f"DST[{jk}]",  jk, f"inner_{j}_{k}"))
            P.append(_aap(f"DST[{jk}]", "T1,T2,T3", jk, f"inner_{j}_{k}", "store sum"))
        P.append(_aap(f"DST[{j+n}]", "DCC1", -1, f"outer_{j}", "carry out"))

    return UProgram("Multiplication", "S = A × B", 2, 1, n, P)


@register("division")
def division(n: int) -> UProgram:
    """S = A / B  (~8n^2+12n primitives, quadratic)"""
    P = []

    for j in range(n - 1, -1, -1):
        nj = n - j  # rest length

        # Step 1: copy input bit into rest
        P.append(_aap("T0",       f"SRC1[{j}]", j, f"div_s1_j{j}", "load dividend bit"))
        P.append(_aap(f"REST[{j}]", "T0",        j, f"div_s1_j{j}", f"rest[{j}] ← A[{j}]"))

        # Step 2: comparison (is rest >= divisor?)
        P.append(_aap("T1", "C1", j, f"div_s2_j{j}", "init subtraction carry"))

        # Step 2, Loop A: compare rest vs divisor
        for k in range(nj):
            P.append(_aap("T2",     f"REST[{j+k}]",  j, f"div_s2a_j{j}", f"rest[{j+k}]"))
            P.append(_aap("DCC0N",  f"SRC2[{k}]",    j, f"div_s2a_j{j}", f"NOT B[{k}]"))
            P.append(_ap ("DCC0,T1,T2",               j, f"div_s2a_j{j}", "subtract step"))

        # Step 2, Loop B: extend comparison with zero padding
        k = nj
        while k < n - 1:
            P.append(_aap("T2,T3",  "C0",            j, f"div_s2b_j{j}"))
            P.append(_aap("DCC0N",  f"SRC2[{k}]",    j, f"div_s2b_j{j}"))
            P.append(_aap("T0",     "DCC0,T1,T2",    j, f"div_s2b_j{j}", "coalesced"))
            P.append(_aap("DCC1N",  f"SRC2[{k+1}]",  j, f"div_s2b_j{j}"))
            P.append(_aap("T1",     "DCC1,T0,T3",    j, f"div_s2b_j{j}", "coalesced"))
            k += 2

        # Step 2, Tail
        if j % 2 == 1:
            P.append(_aap("T2",    "C0",                    j, f"div_s2t_j{j}"))
            P.append(_aap("DCC0N", f"SRC2[{n-1}]",          j, f"div_s2t_j{j}"))
            P.append(_aap("M",     "DCC0,T1,T2",            j, f"div_s2t_j{j}", "m ← compare result"))
        else:
            P.append(_aap("M",     "T1",                    j, f"div_s2t_j{j}", "m ← compare result"))

        # Step 3: store quotient bit
        P.append(_aap(f"DST[{j}]", "M", j, f"div_s3_j{j}", f"quotient[{j}] ← m"))

        # Step 4: partial product (tmp = m AND divisor)
        k = 0
        while k < nj - 1:
            P.append(_aap("T0,T1,T2", "C0", j, f"div_s4_j{j}"))
            P.append(_aap("T2,T3",    "M",  j, f"div_s4_j{j}"))
            P.append(_aap("DCC0",     f"SRC2[{k}]",   j, f"div_s4_j{j}"))
            P.append(_aap(f"TMP[{k}]","DCC0,T1,T2",   j, f"div_s4_j{j}", "coalesced AND"))
            P.append(_aap("DCC1",     f"SRC2[{k+1}]", j, f"div_s4_j{j}"))
            P.append(_aap(f"TMP[{k+1}]","DCC1,T0,T3", j, f"div_s4_j{j}", "coalesced AND"))
            k += 2

        if nj % 2 == 1:
            kk = nj - 1
            P.append(_aap("T0", "C0",              j, f"div_s4t_j{j}"))
            P.append(_aap("T1", "M",               j, f"div_s4t_j{j}"))
            P.append(_aap("T2", f"SRC2[{kk}]",     j, f"div_s4t_j{j}"))
            P.append(_aap(f"TMP[{kk}]", "T0,T1,T2",j, f"div_s4t_j{j}", "AND tail"))

        # Step 5: rest = rest - tmp (subtraction)
        P.append(_aap("DCC0", "C1", j, f"div_s5_j{j}", "init sub carry"))
        for k in range(nj):
            P.append(_aap("T0,T1,T2",  "DCC0",          j, f"div_s5_j{j}", "copy carry"))
            P.append(_aap("T2,T3",     f"REST[{j+k}]",  j, f"div_s5_j{j}"))
            P.append(_aap("DCC1N",     f"TMP[{k}]",     j, f"div_s5_j{j}"))
            P.append(_aap("DCC0",      "DCC1,T0,T3",    j, f"div_s5_j{j}", "coalesced sub"))
            P.append(_aap("T0,T3",     "DCC1N",         j, f"div_s5_j{j}"))
            P.append(_ap ("T0,T1,T2",                   j, f"div_s5_j{j}", "sub sum"))
            P.append(_aap("DCC1N",     f"TMP[{k}]",     j, f"div_s5_j{j}"))
            P.append(_aap(f"REST[{j+k}]", "DCC1,T0,T3", j, f"div_s5_j{j}", "store rest"))

    return UProgram("Division", "S = A / B", 2, 1, n, P)
