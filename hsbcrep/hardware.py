"""Hardware helpers: chain choice, ISA circuits, estimator options, job bookkeeping.

Estimator settings follow the HSBC/IBM paper (Sec. 3.2): Pauli twirling on gates and readout (TREX), no ZNE.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from qiskit.quantum_info import SparsePauliOp
from qiskit.transpiler import generate_preset_pass_manager

from . import pqfm


def chain_score(chain, target):
    """Sum of readout errors on the chain plus CZ errors on its bonds (lower is better)."""
    s = 0.0
    for q in chain:
        p = target["measure"].get((q,)) if "measure" in target.operation_names else None
        s += (p.error or 0.0) if p is not None and p.error is not None else 0.05
    for a, b in zip(chain[:-1], chain[1:]):
        p = target["cz"].get((a, b)) or target["cz"].get((b, a))
        s += (p.error or 0.0) if p is not None and p.error is not None else 0.05
    return s


def choose_chain(backend, candidates):
    """Best of the precomputed n-qubit chains (outputs/chains129.json), scored on today's calibration.
    Chains whose bonds are not all couplers of this backend are skipped."""
    cm = {tuple(sorted(e)) for e in backend.coupling_map.get_edges()}
    scored = []
    for c in candidates:
        if all(tuple(sorted((a, b))) in cm for a, b in zip(c[:-1], c[1:])):
            scored.append((chain_score(c, backend.target), c))
    if not scored:
        raise RuntimeError("no precomputed chain fits this backend's coupling map")
    scored.sort(key=lambda t: t[0])
    return scored[0][1], scored[0][0], len(scored)


def build_isa(n, ops, n_feat, phys, backend, seed=0):
    """Transpile once onto the given physical qubits. Returns (isa, parameter column order, observables)."""
    qc, J = pqfm.to_circuit(n, ops, n_feat)
    pm = generate_preset_pass_manager(optimization_level=1, backend=backend, initial_layout=list(phys), seed_transpiler=seed)
    isa = pm.run(qc)
    if list(isa.layout.final_index_layout()) != list(phys):
        raise RuntimeError("routing moved qubits: the chain bonds are not all native couplers")
    cols = [p.index for p in isa.parameters]
    obs = [[SparsePauliOp.from_sparse_list([(P, [k], 1.0)], num_qubits=n).apply_layout(isa.layout)]
           for P in "XYZ" for k in range(n)]
    return isa, cols, obs


def two_q_count(isa):
    return int(isa.size(lambda inst: inst.operation.num_qubits == 2)), int(isa.depth(lambda inst: inst.operation.num_qubits == 2))


def make_estimator(backend, shots, hardware=True):
    from qiskit_ibm_runtime import EstimatorV2
    est = EstimatorV2(mode=backend)
    est.options.default_shots = shots
    if hardware:
        est.options.resilience_level = 1
        est.options.resilience.measure_mitigation = True       # TREX
        est.options.resilience.zne_mitigation = False
        est.options.twirling.enable_gates = True              # Pauli twirling, as in the HSBC paper
        est.options.twirling.enable_measure = True
        est.options.dynamical_decoupling.enable = False
    return est


def calibration_stamp(backend):
    """Time of the backend's last calibration update (string), or '' if unavailable."""
    try:
        p = backend.properties()
        return str(p.last_update_date)
    except Exception:
        return ""


def save_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=1, default=str))


def load_json(path, default=None):
    p = Path(path)
    return json.loads(p.read_text()) if p.exists() else default


def unpack(result, n_pubs_qubits):
    """EstimatorV2 result -> list of (evs, stds) arrays shaped (rows, 3, n) per PUB."""
    out = []
    for pub, n in zip(result, n_pubs_qubits):
        evs = np.asarray(pub.data.evs).reshape(3, n, -1).transpose(2, 0, 1)
        std = np.asarray(pub.data.stds).reshape(3, n, -1).transpose(2, 0, 1)
        out.append((evs, std))
    return out
