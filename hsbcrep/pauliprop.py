"""Heisenberg-picture Pauli propagation for <X_k>, <Y_k>, <Z_k>, vectorized over samples.

Used where a light cone is too large for a statevector (5 or more repetitions on the
156-qubit lattice). Each observable is evolved backward through the circuit as a sum of
Pauli strings whose coefficients are vectors over the samples. A rotation exp(-i a Q / 2)
that anticommutes with a string P splits it: P -> cos(a) P + sin(a) (-i P Q). Strings whose
largest |coefficient| over the samples falls below `tol` are dropped; that is the only
approximation, and the dropped weight is returned so the error can be bounded.

Supports the gates of the feature maps in featuremap.py: h, rx, ry, rz, rzz, rxx, ryy, cz.
"""
from __future__ import annotations

import numpy as np

from .featuremap import cone


def _pc(v: int) -> int:
    return v.bit_count()


def _phase_exp(x1, z1, x2, z2):
    """g such that sigma(x1,z1) sigma(x2,z2) = i^g sigma(x1^x2, z1^z2) (Hermitian convention)."""
    Y1, X1, Z1 = x1 & z1, x1 & ~z1, ~x1 & z1
    g = (_pc(Y1 & z2 & ~x2) - _pc(Y1 & x2 & ~z2)
         + _pc(X1 & z2 & x2) - _pc(X1 & z2 & ~x2)
         + _pc(Z1 & x2 & ~z2) - _pc(Z1 & x2 & z2))
    return g % 4


def _mul_minus_i(x1, z1, x2, z2):
    """-i P Q for anticommuting Hermitian P, Q: returns (x, z, sign) with sign = +-1."""
    g = _phase_exp(x1, z1, x2, z2)            # odd for anticommuting pairs
    s = {1: 1.0, 3: -1.0}[g]                  # -i * i^g
    return x1 ^ x2, z1 ^ z2, s


def _anticommute(x1, z1, x2, z2):
    return (_pc(x1 & z2) + _pc(z1 & x2)) & 1


def _angles(op, X):
    a = np.full(len(X), op.coeff, dtype=np.float64)
    for f in op.feats:
        a = a * X[:, f]
    return a


def _gen(op):
    """Pauli generator (x, z) of a rotation gate, as bitmasks on local qubits."""
    g, q = op.gate, op.qubits
    b = lambda i: 1 << i
    if g == "rz":  return 0, b(q[0])
    if g == "rx":  return b(q[0]), 0
    if g == "ry":  return b(q[0]), b(q[0])
    if g == "rzz": return 0, b(q[0]) | b(q[1])
    if g == "rxx": return b(q[0]) | b(q[1]), 0
    if g == "ryy": return b(q[0]) | b(q[1]), b(q[0]) | b(q[1])
    raise ValueError(g)


def _h(terms, q):
    m = 1 << q
    out = {}
    for (x, z), c in terms.items():
        xb, zb = bool(x & m), bool(z & m)
        nx = (x & ~m) | (m if zb else 0)
        nz = (z & ~m) | (m if xb else 0)
        out[(nx, nz)] = -c if (xb and zb) else c   # Y -> -Y
    return out


def _cz(terms, a, b):
    """CZ conjugation: X_a -> X_a Z_b, X_b -> Z_a X_b (with Y signs)."""
    ma, mb = 1 << a, 1 << b
    out = {}
    for (x, z), c in terms.items():
        xa, xb = bool(x & ma), bool(x & mb)
        nz = z ^ (mb if xa else 0) ^ (ma if xb else 0)
        # sign: CZ maps X_a X_b -> Y_a Y_b etc; track via explicit product
        sign = 1.0
        if xa and xb:
            # X_a X_b -> (X_a Z_b)(Z_a X_b) = (X_a Z_a)(Z_b X_b) = (-iY_a)(iY_b) = Y_a Y_b
            # general case handled by computing the phase of the product of images
            sign = _cz_sign(x, z, a, b)
        out[(x, nz)] = out.get((x, nz), 0) + sign * c
    return out


def _cz_sign(x, z, a, b):
    # image of sigma(x,z) under CZ conjugation, as product of images of single-qubit factors on a, b
    ma, mb = 1 << a, 1 << b
    img = []
    for q, m, o in ((a, ma, mb), (b, mb, ma)):
        xq, zq = bool(x & m), bool(z & m)
        if xq and zq:  img.append((m, m | o))       # Y_q -> Y_q Z_o
        elif xq:       img.append((m, o))           # X_q -> X_q Z_o
        elif zq:       img.append((0, m))
        else:          img.append((0, 0))
    (x1, z1), (x2, z2) = img
    return {0: 1.0, 2: -1.0}[_phase_exp(x1, z1, x2, z2)]


def evolve(ops, P, X, tol=1e-6):
    """Backward-evolve Pauli string P = (x, z) through ops for every sample row of X.

    Returns (expectations in |0...0>, dropped weight bound, peak number of strings).
    """
    N = len(X)
    terms = {P: np.ones(N)}
    dropped = np.zeros(N)
    peak = 1
    for op in reversed(ops):
        g = op.gate
        if g == "h":
            terms = _h(terms, op.qubits[0]); continue
        if g == "cz":
            terms = _cz(terms, *op.qubits); continue
        qx, qz = _gen(op)
        a = _angles(op, X)
        ca, sa = np.cos(a), np.sin(a)
        out = {}
        for (x, z), c in terms.items():
            if _anticommute(x, z, qx, qz):
                nx, nz, s = _mul_minus_i(x, z, qx, qz)
                k1, k2 = (x, z), (nx, nz)
                out[k1] = out.get(k1, 0) + c * ca
                out[k2] = out.get(k2, 0) + (s * sa) * c
            else:
                out[(x, z)] = out.get((x, z), 0) + c
        terms = {}
        for k, c in out.items():
            if np.abs(c).max() >= tol:
                terms[k] = c
            else:
                dropped += np.abs(c)
        peak = max(peak, len(terms))
    ev = np.zeros(N)
    for (x, z), c in terms.items():
        if x == 0:                       # only I and Z survive in <0|.|0>
            ev += c
    return ev, dropped, peak


def projections_pp(n, ops, X, tol=1e-6, qubits=None):
    """<X_k>, <Y_k>, <Z_k> for all (or the listed) qubits by Pauli propagation in each light cone.

    Returns (P with shape (N, 3, n), dropped weight with the same shape, peak strings per qubit).
    """
    X = np.asarray(X, dtype=np.float64)
    qs = range(n) if qubits is None else qubits
    out = np.zeros((len(X), 3, n)); drop = np.zeros_like(out); peak = np.zeros(n, int)
    for k in qs:
        c, sup = cone(ops, k)
        loc = {q: i for i, q in enumerate(sup)}
        local = [type(o)(o.gate, tuple(loc[q] for q in o.qubits), o.coeff, o.feats) for o in c]
        kk = loc[k]; m = 1 << kk
        for j, P in enumerate(((m, 0), (m, m), (0, m))):        # X, Y, Z
            ev, dr, pk = evolve(local, P, X, tol)
            out[:, j, k] = ev; drop[:, j, k] = dr; peak[k] = max(peak[k], pk)
    return out, drop, peak
