"""Feature maps as a small gate list, so the hardware circuit and the classical
simulation are built from the same object.

An op is (gate, qubits, coeff, feats):
    angle = coeff * product(x[f] for f in feats)        feats = () means a fixed angle
Gates: h, rx, ry, rz (1 qubit); rzz, rxx, ryy, cz (2 qubits).

To plug in a map from your NAS, return a list of such ops from a function with the same
signature as `zz_map` and register it in FEATURE_MAPS.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector

from .lattice import edge_layers

ONE_Q = {"h", "rx", "ry", "rz"}
TWO_Q = {"rzz", "rxx", "ryy", "cz"}


@dataclass(frozen=True)
class Op:
    gate: str
    qubits: tuple
    coeff: float = 0.0
    feats: tuple = ()


def zz_map(n, edges, reps=2, pair_scale=1.0, **_):
    """IQP-style map native to the lattice: (H, RZ(x_q), RZZ(pair_scale * x_i * x_j) on edges) x reps."""
    layers = edge_layers(n, edges)
    ops = []
    for _r in range(reps):
        ops += [Op("h", (q,)) for q in range(n)]
        ops += [Op("rz", (q,), 1.0, (q,)) for q in range(n)]
        for layer in layers:
            ops += [Op("rzz", (i, j), pair_scale, (i, j)) for i, j in layer]
    return ops


def ry_cz_map(n, edges, reps=2, **_):
    """Hardware-efficient map: (RY(x_q), RZ(x_q), CZ on edges) x reps. No data in the two-qubit gates."""
    layers = edge_layers(n, edges)
    ops = []
    for _r in range(reps):
        ops += [Op("ry", (q,), 1.0, (q,)) for q in range(n)]
        ops += [Op("rz", (q,), 1.0, (q,)) for q in range(n)]
        for layer in layers:
            ops += [Op("cz", (i, j)) for i, j in layer]
    return ops


FEATURE_MAPS = {"zz": zz_map, "ry_cz": ry_cz_map}


def strip_entanglers(ops):
    """Same map with every two-qubit gate removed: a product state, classical by construction.

    This is the control that matters. Any gain over a classical kernel that survives this
    ablation comes from the bounded single-qubit encoding, not from entanglement."""
    return [o for o in ops if o.gate in ONE_Q]


def to_circuit(n, ops) -> tuple[QuantumCircuit, ParameterVector]:
    """Qiskit circuit with one parameter per feature (x[q] is the angle of gene q)."""
    x = ParameterVector("x", n)
    qc = QuantumCircuit(n)
    for op in ops:
        if op.gate == "h":
            qc.h(op.qubits[0]); continue
        if op.gate == "cz":
            qc.cz(*op.qubits); continue
        ang = op.coeff
        for f in op.feats:
            ang = ang * x[f]
        getattr(qc, op.gate)(ang, *op.qubits)
    return qc, x


DIAGONAL = {"rz", "rzz", "cz"}


def cone(ops, k):
    """Ops in the backward light cone of qubit k, in circuit order, and the qubits they touch.

    Walks the circuit backward in the Heisenberg picture. `gen` holds qubits where the
    evolved observable can be any Pauli, `dia` holds qubits where it is only I or Z. A
    diagonal gate (rz, rzz, cz) that touches no `gen` qubit commutes with the observable and
    is dropped, so commuting ZZ layers grow the cone by one lattice step per repetition,
    not one step per layer. Dropped gates cancel exactly; nothing is approximated.
    """
    gen, dia, keep = {k}, set(), []
    for op in reversed(ops):
        q = set(op.qubits)
        if op.gate in DIAGONAL:
            if not q & gen:
                continue
            dia |= q - gen
        else:
            if not q & (gen | dia):
                continue
            gen |= q
            dia -= q
        keep.append(op)
    return keep[::-1], sorted(gen | dia)


CZ_COST = {"cz": 1, "rzz": 2, "rxx": 2, "ryy": 2}     # CZ gates per op when the basis is CZ


def cone_stats(n, ops):
    """Per measured qubit: light-cone size, two-qubit ops in the cone, and their CZ count."""
    size, n2q, ncz = np.zeros(n, int), np.zeros(n, int), np.zeros(n, int)
    for k in range(n):
        c, s = cone(ops, k)
        size[k] = len(s)
        n2q[k] = sum(o.gate in TWO_Q for o in c)
        ncz[k] = sum(CZ_COST.get(o.gate, 0) for o in c)
    return size, n2q, ncz
