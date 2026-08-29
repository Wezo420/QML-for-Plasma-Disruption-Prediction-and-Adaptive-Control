"""
circuits.py
===========
Parameterised quantum-circuit building blocks used to assemble the QRC
encoding / reservoir / readout-prep stages shown on slide 9 of the project
presentation ("Quantum Circuit Demonstration"):

    1. Data encoding & superposition  -> Hadamard layer + RY rotation encoding
    2. Quantum reservoir evolution    -> fixed entangling unitary (CNOTs)
    3. Measurement & feature vector   -> handled in quantum_reservoir.py
    4. Classical readout              -> handled in quantum_reservoir.py

The block design (RY-rotation encoding followed by a layer of CNOT
entanglers, repeated for the input block X, a fixed random block A, and
optionally a trainable-but-frozen block P) follows the general pattern
used in Ahmed, Tennie & Magri, "Robust quantum reservoir computers for
forecasting chaotic dynamics" and the accompanying MagriLab
Stability_QRC_GS reference implementation that this project is based on
(https://github.com/MagriLab/Stability_QRC_GS). The functions below are
an independent, simplified re-implementation for this project (fewer,
better-documented block types; no Qiskit-version-specific label kwargs)
rather than a direct copy.

Every block function takes a Qiskit ``QuantumCircuit`` and mutates it in
place (matching the reference repo's convention), then returns it for
convenience.
"""
from __future__ import annotations

from itertools import combinations
from typing import Sequence

from qiskit import QuantumCircuit


def linear_entangler(n_qubits: int, params: Sequence[float], qc: QuantumCircuit) -> QuantumCircuit:
    """RY rotation on every qubit (cycling through ``params`` if there are
    more parameters than qubits) followed by a nearest-neighbour CNOT
    chain (0->1->2->...->n-1->0). This is the cheapest entangling block
    ("Linear" in the presentation / reference repo naming) -- O(n) gates,
    good default for larger qubit counts."""
    for j, p in enumerate(params):
        qc.ry(p, j % n_qubits)
    for i in range(n_qubits):
        target = 0 if i == n_qubits - 1 else i + 1
        qc.cx(i, target)
    return qc


def fully_entangled(n_qubits: int, params: Sequence[float], qc: QuantumCircuit,
                     symmetric: bool = False) -> QuantumCircuit:
    """RY rotation encoding followed by an all-to-all CNOT entangling
    layer (every qubit pair connected once). This is the "Fully
    Entangled" block -- O(n^2) gates, richer feature map, matches the
    fully-connected circuit shown in Figure 1 of the presentation.

    If ``symmetric``, a second RY layer using the remaining parameters
    (or the same parameters again if there are exactly n_qubits of them)
    is applied after the entangling layer, mirroring the reference
    repo's ``Unitary_FullyEntSym`` block used for the fixed random-angle
    block ``A``.
    """
    params = list(params)
    for j in range(min(len(params), n_qubits)):
        qc.ry(params[j], j)

    for i, k in combinations(range(n_qubits), 2):
        qc.cx(i, k)

    if symmetric:
        second_half = params[n_qubits:] if len(params) > n_qubits else params
        for j, p in enumerate(second_half):
            qc.ry(p, j % n_qubits)
    return qc


def feature_map_products(n_qubits: int, params: Sequence[float], qc: QuantumCircuit) -> QuantumCircuit:
    """RZ rotation encoding followed by pairwise ZZ-like interactions
    RY(x_i * x_j) sandwiched between CNOTs on every qubit pair. This
    directly encodes pairwise *products* of input features into the
    entangling angles (a nonlinear feature map), matching the
    reference repo's ``Unitary_Feature`` block."""
    params = list(params)
    n_use = min(len(params), n_qubits)
    for j in range(n_use):
        qc.rz(params[j], j)

    pairs = combinations(range(n_use), 2)
    for i, k in pairs:
        qc.cx(i, k)
        qc.ry(params[i] * params[k], k)
        qc.cx(i, k)
    return qc


# Registry so callers (e.g. an ensemble sweep) can select an encoder by name.
ENCODER_REGISTRY = {
    "linear": linear_entangler,
    "full": lambda n, p, qc: fully_entangled(n, p, qc, symmetric=False),
    "full_sym": lambda n, p, qc: fully_entangled(n, p, qc, symmetric=True),
    "feature_product": feature_map_products,
}


def build_reservoir_circuit(
    n_qubits: int,
    x_params: Sequence[float],
    alpha_params: Sequence[float],
    x_encoder: str = "full",
    alpha_encoder: str = "full_sym",
    hadamard: bool = True,
) -> QuantumCircuit:
    """Assemble the full per-timestep reservoir circuit:

        H^(x n_had)  ->  x_encoder(X)  ->  alpha_encoder(alpha)

    where ``X`` carries the (scaled) plasma-state input for this timestep
    and ``alpha`` are the fixed random angles that define the reservoir's
    (untrained) internal dynamics -- exactly Blocks 1 and 2 of the
    "Quantum Circuit Demonstration" on slide 9. The classical-readout
    weights are trained separately (Block 4); this circuit itself is
    never trained.
    """
    qc = QuantumCircuit(n_qubits)
    if hadamard:
        n_had = min(len(x_params), n_qubits)
        qc.h(range(n_had))

    ENCODER_REGISTRY[x_encoder](n_qubits, x_params, qc)
    ENCODER_REGISTRY[alpha_encoder](n_qubits, alpha_params, qc)
    return qc


# A handful of named "configurations" analogous to the reference repo's
# config 1-9, expressed as (x_encoder, alpha_encoder) pairs so an ensemble
# sweep can iterate over them by name.
CIRCUIT_CONFIGS = {
    "linear_linear": ("linear", "linear"),
    "linear_fullsym": ("linear", "full_sym"),
    "full_full": ("full", "full"),
    "full_fullsym": ("full", "full_sym"),
    "feature_linear": ("feature_product", "linear"),
    "feature_fullsym": ("feature_product", "full_sym"),
}
