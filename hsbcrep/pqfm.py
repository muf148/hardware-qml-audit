"""Projected quantum feature maps (PQFM) of the HSBC/IBM bond-trading paper (Ciceri et al., arXiv:2509.17715, Sec. 3.2),
written as gate lists so that hardware circuits and classical simulations come from one object.

Their circuit: a random product fiducial state, then a Trotterized Heisenberg evolution
    U = (prod_odd U_j  prod_even U_j)^M,   U_j = exp(-i alpha/(4M) H_j),   H_j = Jx XX + Jy YY + Jz ZZ,
with couplings J set by features, applied as RZZ, RYY, RXX on nearest-neighbor pairs of a chain. Readout is <X>, <Y>, <Z>
on every qubit. "Shorter": 1 block (M = 2), alpha = 1, fiducial seed 1. "Longer": 2 blocks (M = 4), alpha = 0.1, seed 0.

The paper calls its assignment of features to gates "arbitrarily chosen" and does not give it. Ours is explicit:
features are used cyclically, in gate order (step, parity, pair, then ZZ, YY, XX). The angle of a gate is
    theta = 2 * alpha / (4 M) * J,   J = 2 pi tanh(z / 3)   (their preprocessing; see cryptofill.hsbc_scale),
since R_PP(theta) = exp(-i theta/2 PP).

`heavy_hex=True` replaces the chain by every coupler of a heavy-hex patch, in its three edge colors (a 2D variant
used for the 156-qubit arm; it is our extension, not the paper's circuit).
"""
from __future__ import annotations

import numpy as np

from .featuremap import Op
from .lattice import edge_layers

CONFIGS = {"shorter": {"alpha": 1.0, "blocks": 1, "seed": 1}, "longer": {"alpha": 0.1, "blocks": 2, "seed": 0}}


def fiducial(n, seed):
    """Uniformly random pure single-qubit states: RY(theta) then RZ(phi) on |0>."""
    rng = np.random.default_rng(seed)
    ops = []
    for q in range(n):
        th = float(np.arccos(1 - 2 * rng.random())); ph = float(2 * np.pi * rng.random())
        ops += [Op("ry", (q,), th), Op("rz", (q,), ph)]
    return ops


def pair_layers(n, edges=None, heavy_hex=False):
    """Layers of disjoint pairs: odd then even bonds of the chain 0-1-...-(n-1), or the 3 colors of a heavy-hex graph."""
    if heavy_hex:
        return edge_layers(n, edges)
    return [[(j, j + 1) for j in range(0, n - 1, 2)], [(j, j + 1) for j in range(1, n - 1, 2)]]


def pqfm(n, n_feat, config="longer", edges=None, heavy_hex=False, null=False):
    """Op list. Data gates carry feature index f (angle = coeff * J_f). null=True keeps every gate but zeroes the data
    (coefficient 0), so on hardware it runs the same physical gates with no dependence on the input."""
    cfg = CONFIGS[config]
    M = 2 * cfg["blocks"]
    c = 0.0 if null else 2 * cfg["alpha"] / (4 * M)
    ops = fiducial(n, cfg["seed"])
    layers = pair_layers(n, edges, heavy_hex)
    f = 0
    for _ in range(M):
        for layer in layers:
            for (i, j) in layer:
                for g in ("rzz", "ryy", "rxx"):
                    ops.append(Op(g, (i, j), c, (f % n_feat,)))
                    f += 1
    return ops


def n_data_gates(ops):
    return sum(1 for o in ops if o.feats)


def to_circuit(n, ops, n_feat):
    """Qiskit circuit with one parameter per feature (J[f]). The null map is this same circuit run with J = 0."""
    from qiskit import QuantumCircuit
    from qiskit.circuit import ParameterVector
    J = ParameterVector("J", n_feat)
    qc = QuantumCircuit(n)
    for op in ops:
        if op.gate == "h":
            qc.h(op.qubits[0]); continue
        if op.gate == "cz":
            qc.cz(*op.qubits); continue
        ang = op.coeff
        for f in op.feats:
            ang = ang * J[f]
        getattr(qc, op.gate)(ang, *op.qubits)
    return qc, J


def find_chain(coupling_edges, n_target, avoid=(), starts=None, time_per_start=3.0, rng=None):
    """Longest simple path (up to n_target) in the coupling graph without the `avoid` qubits (backtracking DFS)."""
    import sys, time
    import networkx as nx
    sys.setrecursionlimit(100000)
    g = nx.Graph(); g.add_edges_from(coupling_edges); g.remove_nodes_from(list(avoid))
    best = []
    starts = starts or ([v for v in g.nodes if g.degree(v) == 1] + sorted(g.nodes)[::5])
    for s in starts:
        path, seen, t0 = [s], {s}, time.time()
        cur_best = [s]
        def rec():
            nonlocal cur_best
            if time.time() - t0 > time_per_start or len(cur_best) >= n_target:
                return
            if len(path) > len(cur_best):
                cur_best = list(path)
            nb = [u for u in g.neighbors(path[-1]) if u not in seen]
            jitter = {u: (rng.random() if rng is not None else 0.0) for u in nb}
            for u in sorted(nb, key=lambda u: (sum(1 for w in g.neighbors(u) if w not in seen), jitter[u])):
                path.append(u); seen.add(u); rec(); path.pop(); seen.discard(u)
        rec()
        if len(cur_best) > len(best):
            best = cur_best
        if len(best) >= n_target:
            break
    return best[:n_target]
