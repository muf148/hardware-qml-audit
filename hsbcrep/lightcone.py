"""Exact single-qubit expectation values by light-cone simulation.

<P_k> depends only on the gates in the backward light cone of qubit k. For a shallow map on
heavy-hex that cone holds a handful of qubits, so the 1-RDMs of a 156-qubit circuit are exact
statevector calculations on about ten qubits each. All samples are simulated together.
"""
from __future__ import annotations

import numpy as np

from .featuremap import cone

_H = np.array([[1, 1], [1, -1]], dtype=np.complex128) / np.sqrt(2)


def _angles(op, X):
    a = np.full(len(X), op.coeff, dtype=np.float64)
    for f in op.feats:
        a = a * X[:, f]
    return a


def _bshape(N, m, qs):
    """Shape that broadcasts an (N, 2, ...) per-qubit table over the state."""
    sh = [N] + [1] * m
    for q in qs:
        sh[q + 1] = 2
    return sh


def _apply_1q(psi, U, q):
    """General one-qubit gate. U: (2, 2) or (N, 2, 2). Acts on axis q + 1."""
    v = np.moveaxis(psi, q + 1, 0)
    a, b = v[0].copy(), v[1].copy()
    if U.ndim == 2:
        v[0] = U[0, 0] * a + U[0, 1] * b
        v[1] = U[1, 0] * a + U[1, 1] * b
    else:
        ex = (slice(None),) + (None,) * (psi.ndim - 2)
        v[0] = U[:, 0, 0][ex] * a + U[:, 0, 1][ex] * b
        v[1] = U[:, 1, 0][ex] * a + U[:, 1, 1][ex] * b
    return psi


def _apply_diag(psi, phase, qs):
    """Diagonal gate. phase: (N, 2) for one qubit or (N, 2, 2) for two, indexed by qs order."""
    N, m = psi.shape[0], psi.ndim - 1
    if len(qs) == 2 and qs[0] > qs[1]:
        phase, qs = phase.transpose(0, 2, 1), qs[::-1]
    psi *= phase.reshape(_bshape(N, m, qs))
    return psi


def _apply_2q(psi, U, q0, q1):
    """General two-qubit gate. U: (N, 2, 2, 2, 2) indexed [n, a0, a1, b0, b1]."""
    psi = np.moveaxis(psi, (q0 + 1, q1 + 1), (1, 2))
    psi = np.einsum("nabcd,ncd...->nab...", U, psi)
    return np.ascontiguousarray(np.moveaxis(psi, (1, 2), (q0 + 1, q1 + 1)))


def _rot(axis, a):
    c, s = np.cos(a / 2), np.sin(a / 2)
    U = np.zeros((len(a), 2, 2), dtype=np.complex128)
    if axis == "rx":
        U[:, 0, 0] = c; U[:, 1, 1] = c; U[:, 0, 1] = -1j * s; U[:, 1, 0] = -1j * s
    else:  # ry
        U[:, 0, 0] = c; U[:, 1, 1] = c; U[:, 0, 1] = -s; U[:, 1, 0] = s
    return U


_P = {"rxx": np.array([[0, 1], [1, 0]]), "ryy": np.array([[0, -1j], [1j, 0]])}
_ZZ = np.array([[1.0, -1.0], [-1.0, 1.0]])
_CZ = np.array([[1.0, 1.0], [1.0, -1.0]], dtype=np.complex128)


def _two(gate, a):
    PP = np.kron(_P[gate], _P[gate]).astype(np.complex128)
    U = np.cos(a / 2)[:, None, None] * np.eye(4) - 1j * np.sin(a / 2)[:, None, None] * PP
    return U.reshape(len(a), 2, 2, 2, 2)


def _expect(psi, q):
    """<X>, <Y>, <Z> of qubit q for each sample."""
    N = psi.shape[0]
    v = np.moveaxis(psi, q + 1, 1).reshape(N, 2, -1)
    r01 = np.einsum("nk,nk->n", v[:, 0], v[:, 1].conj())
    z = (np.abs(v[:, 0]) ** 2).sum(1) - (np.abs(v[:, 1]) ** 2).sum(1)
    return 2 * r01.real, -2 * r01.imag, z


def run_ops(ops, m, X):
    """Statevector of an m-qubit op list for every row of X. Returns (N, 2, ..., 2)."""
    N = len(X)
    psi = np.zeros((N,) + (2,) * m, dtype=np.complex128)
    psi[(slice(None),) + (0,) * m] = 1.0
    for op in ops:
        g = op.gate
        if g == "h":
            psi = _apply_1q(psi, _H, op.qubits[0])
        elif g == "rz":
            a = _angles(op, X)
            psi = _apply_diag(psi, np.stack([np.exp(-0.5j * a), np.exp(0.5j * a)], 1), op.qubits)
        elif g == "rzz":
            a = _angles(op, X)
            psi = _apply_diag(psi, np.exp(-0.5j * a[:, None, None] * _ZZ), op.qubits)
        elif g == "cz":
            psi = _apply_diag(psi, np.broadcast_to(_CZ, (N, 2, 2)), op.qubits)
        elif g in ("rx", "ry"):
            psi = _apply_1q(psi, _rot(g, _angles(op, X)), op.qubits[0])
        else:
            psi = _apply_2q(psi, _two(g, _angles(op, X)), *op.qubits)
    return psi


def projections(n, ops, X, max_cone=24, chunk_amps=2 ** 16):
    """Exact <X_k>, <Y_k>, <Z_k> for all qubits. Returns (N, 3, n), plus the cone sizes.

    Raises if a cone exceeds max_cone qubits; use a tensor-network simulation there.
    """
    X = np.asarray(X, dtype=np.float64)
    out = np.zeros((len(X), 3, n))
    sizes = np.zeros(n, int)
    for k in range(n):
        c, sup = cone(ops, k)
        m = len(sup); sizes[k] = m
        if m > max_cone:
            raise ValueError(f"light cone of qubit {k} has {m} qubits (> {max_cone})")
        loc = {q: i for i, q in enumerate(sup)}
        local = [type(o)(o.gate, tuple(loc[q] for q in o.qubits), o.coeff, o.feats) for o in c]
        chunk = max(1, chunk_amps // 2 ** m)
        for s in range(0, len(X), chunk):
            psi = run_ops(local, m, X[s:s + chunk])
            out[s:s + chunk, :, k] = np.stack(_expect(psi, loc[k]), axis=1)
    return out, sizes
