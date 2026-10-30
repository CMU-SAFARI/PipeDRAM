"""
Data Dependency Graph (DDG) for PUD uPrograms
==============================================

Builds a dependency graph from a UProgram's primitive sequence and
classifies each primitive as **bit-dependent** (on the carry chain /
critical path between bit positions) or **bit-independent** (can be
overlapped across pipeline stages in PipeDRAM).

Usage::

    from pud_simulator.ddg import build_ddg
    from pud_simulator.operations import get_operation

    uprogram = get_operation("addition")(n=8)
    ddg = build_ddg(uprogram)
    print(ddg.summary())
"""

from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple

from .primitives import UProgram, Primitive, BGROUP_ROWS, C_ROWS


# Enums

class DepType(str, Enum):
    """Dependency type between two primitives."""
    RAW = "RAW"   # read-after-write
    WAW = "WAW"   # write-after-write
    WAR = "WAR"   # write-after-read


class BitRelation(str, Enum):
    """Relationship between two primitives' bit positions."""
    INTRA_BIT    = "intra_bit"      # same bit_idx (both >= 0)
    INTER_BIT    = "inter_bit"      # different bit_idx (both >= 0)
    INIT_TO_LOOP = "init_to_loop"   # bit_idx -1 -> bit_idx >= 0
    WITHIN_INIT  = "within_init"    # both bit_idx == -1


class PrimClass(str, Enum):
    """Primitive classification for PipeDRAM scheduling."""
    BIT_DEPENDENT   = "bit_dependent"    # on the carry chain
    BIT_INDEPENDENT = "bit_independent"  # can be overlapped
    UNCLASSIFIED    = "unclassified"     # init/epilogue or no bit structure


# Edge

@dataclass(frozen=True)
class DepEdge:
    """A single dependency edge in the DDG."""
    src: int                    # source primitive index (earlier)
    dst: int                    # destination primitive index (later)
    dep_type: DepType           # RAW, WAW, WAR
    row: str                    # physical row causing the dependency
    bit_relation: BitRelation   # intra-bit, inter-bit, etc.


# Dependency Graph

@dataclass
class DependencyGraph:
    """Data dependency graph for a UProgram."""

    uprogram: UProgram
    edges: List[DepEdge] = field(default_factory=list)

    # Adjacency lists
    _adj:  Dict[int, List[DepEdge]] = field(
        default_factory=lambda: defaultdict(list))
    _radj: Dict[int, List[DepEdge]] = field(
        default_factory=lambda: defaultdict(list))

    # Primitives grouped by bit_idx
    bit_groups: Dict[int, List[int]] = field(
        default_factory=lambda: defaultdict(list))

    # Per-primitive classification
    classifications: Dict[int, PrimClass] = field(default_factory=dict)

    # Query methods

    def successors(self, idx: int) -> List[DepEdge]:
        """Outgoing edges from primitive *idx*."""
        return self._adj[idx]

    def predecessors(self, idx: int) -> List[DepEdge]:
        """Incoming edges into primitive *idx*."""
        return self._radj[idx]

    def inter_bit_edges(self) -> List[DepEdge]:
        """All edges crossing bit boundaries (carry chain)."""
        return [e for e in self.edges
                if e.bit_relation == BitRelation.INTER_BIT]

    def bit_dependent_primitives(self, bit_idx: Optional[int] = None) -> List[int]:
        """Primitive indices classified as bit-dependent."""
        if bit_idx is not None:
            return [i for i in self.bit_groups.get(bit_idx, [])
                    if self.classifications.get(i) == PrimClass.BIT_DEPENDENT]
        return [i for i, c in self.classifications.items()
                if c == PrimClass.BIT_DEPENDENT]

    def bit_independent_primitives(self, bit_idx: Optional[int] = None) -> List[int]:
        """Primitive indices classified as bit-independent."""
        if bit_idx is not None:
            return [i for i in self.bit_groups.get(bit_idx, [])
                    if self.classifications.get(i) == PrimClass.BIT_INDEPENDENT]
        return [i for i, c in self.classifications.items()
                if c == PrimClass.BIT_INDEPENDENT]

    @property
    def num_bit_positions(self) -> int:
        """Number of distinct non-negative bit positions."""
        return sum(1 for k in self.bit_groups if k >= 0)

    def critical_path_length(self) -> int:
        """Longest path in the DAG (number of edges)."""
        n = len(self.uprogram.primitives)
        if n == 0:
            return 0
        dist = [0] * n
        for i in range(n):
            for e in self._radj[i]:
                dist[i] = max(dist[i], dist[e.src] + 1)
        return max(dist)

    def per_bit_summary(self) -> Dict[int, Dict[str, int]]:
        """Per bit-position breakdown of dependent/independent counts."""
        result = {}
        for bit_idx in sorted(self.bit_groups):
            if bit_idx < 0:
                continue
            prims = self.bit_groups[bit_idx]
            dep = sum(1 for i in prims
                      if self.classifications.get(i) == PrimClass.BIT_DEPENDENT)
            indep = sum(1 for i in prims
                        if self.classifications.get(i) == PrimClass.BIT_INDEPENDENT)
            result[bit_idx] = {
                "total": len(prims),
                "bit_dependent": dep,
                "bit_independent": indep,
            }
        return result

    def summary(self) -> str:
        """Human-readable summary."""
        n_prims = len(self.uprogram.primitives)
        n_edges = len(self.edges)
        n_inter = len(self.inter_bit_edges())
        n_dep   = len(self.bit_dependent_primitives())
        n_indep = len(self.bit_independent_primitives())
        n_uncl  = sum(1 for c in self.classifications.values()
                      if c == PrimClass.UNCLASSIFIED)

        lines = [
            f"DDG for {self.uprogram.name} (n={self.uprogram.bit_precision})",
            f"  Primitives: {n_prims}",
            f"  Edges: {n_edges} ({n_inter} inter-bit)",
            f"  Bit positions: {self.num_bit_positions}",
            f"  Bit-dependent: {n_dep}  Bit-independent: {n_indep}  "
            f"Unclassified: {n_uncl}",
            f"  Critical path: {self.critical_path_length()} edges",
        ]
        return "\n".join(lines)


# Builder

def build_ddg(uprogram: UProgram) -> DependencyGraph:
    """Build a data dependency graph from a UProgram.

    1. Group primitives by bit_idx.
    2. Walk primitives in program order, tracking last writer / readers
       per row.  Create RAW, WAW, WAR edges.  Exclude C-group rows
       (constants).
    3. Classify primitives as bit-dependent / bit-independent using
       reachability analysis within each bit position.
    """
    graph = DependencyGraph(uprogram=uprogram)
    prims = uprogram.primitives

    # Phase 1: group by bit_idx
    for i, p in enumerate(prims):
        graph.bit_groups[p.bit_idx].append(i)

    # Phase 2: build edges
    last_writer: Dict[str, int] = {}
    last_readers: Dict[str, Set[int]] = defaultdict(set)

    for i, p in enumerate(prims):
        reads  = _dep_rows(p, "read")
        writes = _dep_rows(p, "write")

        seen: Set[Tuple[int, int, str]] = set()

        # RAW: I read from a row that was last written by j
        for r in reads:
            if r in last_writer:
                j = last_writer[r]
                key = (j, i, "RAW")
                if key not in seen:
                    seen.add(key)
                    _add_edge(graph, j, i, DepType.RAW, r, prims)

        # WAW + WAR: I write to a row
        for r in writes:
            if r in last_writer:
                j = last_writer[r]
                key = (j, i, "WAW")
                if key not in seen:
                    seen.add(key)
                    _add_edge(graph, j, i, DepType.WAW, r, prims)

            for j in last_readers.get(r, set()):
                if j == i:
                    continue
                key = (j, i, "WAR")
                if key not in seen:
                    seen.add(key)
                    _add_edge(graph, j, i, DepType.WAR, r, prims)

        # Update tracking
        for r in writes:
            last_writer[r] = i
            last_readers[r] = set()
        for r in reads:
            last_readers[r].add(i)

    # Phase 3: classify primitives
    _classify_primitives(graph)

    return graph


# Internal helpers

def _dep_rows(prim: Primitive, mode: str) -> List[str]:
    """Rows for dependency tracking (B-group + D-group, excluding C-group)."""
    rows = prim.read_rows if mode == "read" else prim.write_rows
    return [r for r in rows if r not in C_ROWS]


def _classify_bit_relation(p1: Primitive, p2: Primitive) -> BitRelation:
    b1, b2 = p1.bit_idx, p2.bit_idx
    if b1 == -1 and b2 == -1:
        return BitRelation.WITHIN_INIT
    if b1 == -1 and b2 >= 0:
        return BitRelation.INIT_TO_LOOP
    if b1 >= 0 and b2 == -1:
        return BitRelation.WITHIN_INIT  # loop -> epilogue, treat conservatively
    if b1 == b2:
        return BitRelation.INTRA_BIT
    return BitRelation.INTER_BIT


def _add_edge(graph: DependencyGraph, src: int, dst: int,
              dep_type: DepType, row: str, prims: list):
    edge = DepEdge(src, dst, dep_type, row,
                   _classify_bit_relation(prims[src], prims[dst]))
    graph.edges.append(edge)
    graph._adj[src].append(edge)
    graph._radj[dst].append(edge)


def _classify_primitives(graph: DependencyGraph):
    """Classify each primitive as bit-dependent, bit-independent, or unclassified.

    For each bit position j:
      1. Find *entries*: primitives at bit j with incoming inter-bit or
         init-to-loop edges through B-group rows.
      2. Find *exits*: primitives at bit j with outgoing inter-bit edges
         through B-group rows.
      3. Forward reachability from entries (intra-bit B-group edges only).
      4. Backward reachability from exits (intra-bit B-group edges only).
      5. Intersection = bit-dependent.  Rest = bit-independent.
    """
    prims = graph.uprogram.primitives

    # All bit_idx=-1 -> UNCLASSIFIED
    for i, p in enumerate(prims):
        if p.bit_idx == -1:
            graph.classifications[i] = PrimClass.UNCLASSIFIED

    for bit_idx in sorted(graph.bit_groups):
        if bit_idx < 0:
            continue

        bit_set = set(graph.bit_groups[bit_idx])

        # Build intra-bit adjacency restricted to B-group edges
        intra_fwd:  Dict[int, Set[int]] = defaultdict(set)
        intra_rev:  Dict[int, Set[int]] = defaultdict(set)
        for i in bit_set:
            for e in graph._adj[i]:
                if e.dst in bit_set and e.row in BGROUP_ROWS:
                    intra_fwd[i].add(e.dst)
                    intra_rev[e.dst].add(i)

        # Entries: incoming RAW inter-bit / init-to-loop edges via B-group.
        # Only RAW edges represent true data flow between bit positions;
        # WAW/WAR are resource conflicts that don't exist in PipeDRAM
        # (each mat has its own B-group rows).
        entries: Set[int] = set()
        for i in bit_set:
            for e in graph._radj[i]:
                if (e.src not in bit_set and e.row in BGROUP_ROWS
                        and e.dep_type == DepType.RAW):
                    entries.add(i)
                    break

        # Exits: outgoing RAW inter-bit edges via B-group
        exits: Set[int] = set()
        for i in bit_set:
            for e in graph._adj[i]:
                if (e.dst not in bit_set and e.row in BGROUP_ROWS
                        and e.dep_type == DepType.RAW):
                    exits.add(i)
                    break

        # Forward reachability from entries
        fwd_reach = _reachable(entries, intra_fwd)
        # Backward reachability from exits
        bwd_reach = _reachable(exits, intra_rev)

        bit_dependent = fwd_reach & bwd_reach

        for i in bit_set:
            if i in bit_dependent:
                graph.classifications[i] = PrimClass.BIT_DEPENDENT
            else:
                graph.classifications[i] = PrimClass.BIT_INDEPENDENT


def _reachable(seeds: Set[int], adj: Dict[int, Set[int]]) -> Set[int]:
    """BFS/DFS reachability from a set of seed nodes."""
    visited: Set[int] = set()
    stack = list(seeds)
    while stack:
        node = stack.pop()
        if node in visited:
            continue
        visited.add(node)
        for succ in adj.get(node, set()):
            if succ not in visited:
                stack.append(succ)
    return visited
