"""Where circuits run. One place to change the device or the saved account."""
from __future__ import annotations

ACCOUNT = "Faryad-free-ibm"      # name of the saved IBM Quantum account
DEVICE = "ibm_fez"               # 156-qubit Heron r2


def local_backend():
    """FakeFez: the ibm_fez lattice and a calibration snapshot, for local checks."""
    from qiskit_ibm_runtime.fake_provider import FakeFez
    return FakeFez()
