"""Pick a connected n-qubit patch of a heavy-hex device.

Qubits are removed one at a time, worst first, never disconnecting the patch. The removal
order is fixed, so patches are nested: the 100-qubit patch contains the 80-qubit patch.
"""
from __future__ import annotations

import networkx as nx
import numpy as np


def device_graph(backend) -> nx.Graph:
    g = nx.Graph()
    g.add_nodes_from(range(backend.num_qubits))
    g.add_edges_from((a, b) for a, b in backend.coupling_map.get_edges())
    return g


def qubit_costs(backend) -> dict[int, float]:
    """Readout error plus mean two-qubit error on the qubit's couplers (from the target)."""
    t = backend.target
    two_q = next((g for g in ("cz", "ecr", "cx") if g in t.operation_names), None)
    cost = {}
    for q in range(backend.num_qubits):
        ro = t["measure"][(q,)].error if "measure" in t.operation_names and t["measure"].get((q,)) else 0.0
        e2 = [p.error for qs, p in t[two_q].items() if q in qs and p is not None and p.error is not None] if two_q else []
        cost[q] = float((ro or 0.0) + (np.mean(e2) if e2 else 1.0))
    return cost


def removal_order(g: nx.Graph, cost: dict[int, float]) -> list[int]:
    """Order in which qubits are dropped, worst removable qubit first."""
    g = g.subgraph(max(nx.connected_components(g), key=len)).copy()
    order = []
    while g.number_of_nodes() > 1:
        art = set(nx.articulation_points(g))
        q = max((v for v in g.nodes if v not in art), key=lambda v: (cost[v], v))
        g.remove_node(q)
        order.append(q)
    return order


def patch(backend, n: int, cost: dict[int, float] | None = None):
    """Return (physical qubits, edges in local indices) for a connected n-qubit patch."""
    g = device_graph(backend)
    cost = qubit_costs(backend) if cost is None else cost
    drop = removal_order(g, cost)
    big = max(nx.connected_components(g), key=len)
    if n > len(big):
        raise ValueError(f"backend has only {len(big)} connected qubits")
    keep = sorted(set(big) - set(drop[: len(big) - n]))
    loc = {p: i for i, p in enumerate(keep)}
    edges = sorted((loc[a], loc[b]) for a, b in g.subgraph(keep).edges)
    edges = [tuple(sorted(e)) for e in edges]
    return keep, sorted(set(edges))


def edge_layers(n: int, edges) -> list[list[tuple[int, int]]]:
    """Split edges into layers of disjoint gates.

    Uses the alternating-path edge coloring for bipartite graphs, so a heavy-hex patch
    (bipartite, degree <= 3) always gets at most 3 layers.
    """
    g = nx.Graph(); g.add_nodes_from(range(n)); g.add_edges_from(edges)
    if not nx.is_bipartite(g):
        col = nx.coloring.greedy_color(nx.line_graph(g), strategy="saturation_largest_first")
        out = [[] for _ in range(max(col.values(), default=-1) + 1)]
        for e, c in col.items():
            out[c].append(tuple(sorted(e)))
        return [sorted(l) for l in out]
    D = max((d for _, d in g.degree), default=0)
    at = {v: {} for v in g.nodes}                 # at[v][color] = neighbor joined by that color
    for u, v in sorted(tuple(sorted(e)) for e in g.edges):
        a = next(c for c in range(D) if c not in at[u])
        b = next(c for c in range(D) if c not in at[v])
        if a != b:                                 # free color a at v by flipping the a/b path from v
            path, w, c = [], v, a
            while c in at[w]:
                x = at[w][c]; path.append((w, x, c)); w, c = x, (b if c == a else a)
            for w, x, c in path:
                del at[w][c]; del at[x][c]
            for w, x, c in path:
                c2 = b if c == a else a
                at[w][c2] = x; at[x][c2] = w
        at[u][a] = v; at[v][a] = u
    out = [[] for _ in range(D)]
    for v in g.nodes:
        for c, w in at[v].items():
            if v < w:
                out[c].append((v, w))
    return [sorted(l) for l in out]
