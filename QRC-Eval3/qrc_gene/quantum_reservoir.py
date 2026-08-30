"""
quantum_reservoir.py
=====================
Quantum Reservoir Computer (QRC) -- Block 3A of the project block diagram.


CIRCUIT LIBRARY
---------------
Eight named circuit "recipes" are registered in ``CIRCUIT_LIBRARY`` below, each
built from one or more of the block-appliers in the "Block appliers" section.
Five of them (``magri_*``) are a faithful re-implementation of the five
configurations in ``QuantumReservoirNetwork.gen_quantumcircuit`` from
https://github.com/MagriLab/Stability_QRC_GS ``src/QRC/qrc.py`` (verified
directly against that file and its ``src/QRC/unitaryblock.py`` -- the block
names in the docstrings below (``Unitary4``, ``Unitary_FullyEnt``,
``Unitary_FullyEntSym``, ``Unitary_Feature``, ``Unitary_ReverseEnt``,
``Unitary_C``) are that file's own names, kept here purely so the mapping is
traceable). Two more (``magri_reverse_ent``, ``magri_bidirectional``) port two
block types that exist in that repo's ``unitaryblock.py`` but were never wired
into any of its five official configs. The eighth (``qasm_reupload_dual_a``) is a new configuration.

``ReservoirCircuit_Nonlinear_GENE_Config1.qasm`` circuit
    magri_full_fullsym      P:chain-ladder -> X:full-ent        -> A:full-ent-sym   [recurrent]
    magri_linear            P:chain-ladder -> X:chain-ladder     -> A:chain-ladder   [recurrent]
    magri_linear_x2         X:chain-ladder x2 (no P)                -> A:chain-ladder
    magri_full_x2           X:full-ent x2 (no P)                    -> A:full-ent-sym
    magri_feature_x2        X:feature-product x2 (no P)             -> A:chain-ladder
    magri_reverse_ent       X:reverse-sandwich x2 (no P)             -> A:chain-ladder    [NEW]
    magri_bidirectional     X:bidirectional-linear x2 (no P)         -> A:full-ent-sym    [NEW]
    qasm_reupload_dual_a    X:chain-ladder x n_qubits (no P)         -> A:full-ent-sym x2 [NEW, from QASM]

"P" (recurrence) re-encodes the reservoir's own previous measurement (all
2**n_qubits basis probabilities) as rotation angles -- genuine *quantum*
memory, not just a classical leaky average. "X" encodes the current (scaled)
plasma-state input. "A" is a fixed, never-trained random-angle entangling
unitary that gives the reservoir its fixed internal dynamics. Configs without
a P block still get a (classical) memory mechanism via the leaky update on the
measured-probability vector, ``epsilon_q`` -- same mechanism used for every
config, see ``QuantumReservoirComputer.step``.

For speed, each reservoir builds ONE parameterised circuit *template* per
instance (Qiskit ``ParameterVector`` placeholders for P and X; the fixed
random block A is baked in as plain numbers immediately, since it never
changes), then rebinds only the numeric P/X values every timestep via
``assign_parameters`` instead of rebuilding circuit topology from scratch.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Callable, Dict, NamedTuple, Optional, Sequence

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import Statevector

# ==========================================================================
# Block appliers -- each mutates ``qc`` in place and returns it.
# ==========================================================================


def apply_chain_ladder(qc: QuantumCircuit, n: int, params: Sequence[float], name: str) -> QuantumCircuit:
    """RY-then-CNOT ladder that walks up the qubit chain and bounces back at
    the far end (0->1->2->...->n-1->n-2->...), cycling through ``params`` if
    there are more of them than qubits. This is a fresh implementation of the
    same gate sequence as Stability_QRC_GS's ``Unitary4`` -- used there for
    every "linear"-labelled block (P, X or A depending on the config)."""
    for j, theta in enumerate(params):
        q = j % n
        qc.ry(theta, q, label=f"$R_Y$({name})")
        if q <= 1:
            qc.cx(q, q + 1)
        elif q % (n - 1) == 0:
            qc.cx(q, q - 1)
        else:
            qc.cx(q, q + 1)
        if q == n - 1:
            qc.barrier()
    return qc


def apply_dense_entangler(qc: QuantumCircuit, n: int, params: Sequence[float], name: str,
                           mirror: bool = False) -> QuantumCircuit:
    """RY on every qubit from the first ``n`` params, then an all-to-all CNOT
    layer (every qubit pair connected once, O(n^2) gates -- the "fully
    connected" circuit shown in Figure 1 of the presentation), then either a
    second RY layer re-using the SAME params (``mirror=True`` -- Stability_QRC
    GS's ``Unitary_FullyEntSym``, used for the fixed A block) or an RY layer
    consuming any params left over past the first n (``mirror=False`` --
    ``Unitary_FullyEnt``, used for the X block)."""
    params = list(params)
    for q in range(min(len(params), n)):
        qc.ry(params[q], q, label=f"$R_Y$({name})")
    for a, b in combinations(range(n), 2):
        qc.cx(a, b)
    if mirror:
        for q in range(min(len(params), n)):
            qc.ry(params[q], q, label=f"$R_Y$({name})")
    else:
        for q in range(n, len(params)):
            qc.ry(params[q], q % n, label=f"$R_Y$({name})")
    return qc


def apply_pairwise_product_encoder(qc: QuantumCircuit, n: int, params: Sequence[float],
                                    name: str) -> QuantumCircuit:
    """RZ-encode each input, then for every qubit pair (i, k) sandwich a
    CNOT-RY(param_i*param_k)-CNOT around it, injecting the pairwise *product*
    of two input features as an entangling angle -- a genuinely nonlinear
    (quadratic) feature map. Matches Stability_QRC_GS's ``Unitary_Feature``."""
    params = list(params)
    n_use = min(len(params), n)
    for q in range(n_use):
        qc.rz(params[q], q)
    for a, b in combinations(range(n_use), 2):
        qc.cx(a, b)
        qc.ry(params[a] * params[b], b, label=f"$R_Y$({name}\u00b7{name})")
        qc.cx(a, b)
    return qc


def apply_reverse_sandwich_entangler(qc: QuantumCircuit, n: int, params: Sequence[float],
                                      name: str) -> QuantumCircuit:
    """Forward nearest-neighbour CNOT chain (0-1, 1-2, ..., n-2 - n-1), then
    an RY layer on every qubit, then the SAME chain of CNOTs applied again
    but walked in reverse iteration order (n-2 - n-1 first, down to 0-1
    last). Because CNOTs on overlapping qubit pairs don't commute through an
    intervening rotation layer, running the chain forward then backward
    (rather than forward twice) produces a genuinely different entangled
    state, not merely the identity -- this is a fresh implementation of
    Stability_QRC_GS's ``Unitary_ReverseEnt``, which exists in that repo's
    ``unitaryblock.py`` but (unlike Unitary4/Unitary_FullyEnt/Unitary_Feature)
    was never wired into any of its five named configs, so this is a new
    ensemble member rather than a re-statement of what's already covered."""
    for q in range(n - 1):
        qc.cx(q, q + 1)
    for j, theta in enumerate(params):
        qc.ry(theta, j % n, label=f"$R_Y$({name})")
    for q in reversed(range(n - 1)):
        qc.cx(q, q + 1)
    qc.barrier()
    return qc


def apply_bidirectional_entangler(qc: QuantumCircuit, n: int, params: Sequence[float],
                                   name: str) -> QuantumCircuit:
    """For each qubit i in turn: RY(param_i), then CNOT(i, i-1) if i>0 and
    CNOT(i, i+1) if i<n-1. Every interior qubit ends up entangled with BOTH
    neighbours by the time the sweep passes it, giving denser local
    connectivity than ``apply_chain_ladder`` while staying O(n) (no all-to-all
    cost). Fresh implementation of Stability_QRC_GS's ``Unitary_C``, which --
    like ``Unitary_ReverseEnt`` above -- is defined in that repo but not used
    by any of its five configs, so this is a new ensemble member."""
    for q in range(n):
        theta = params[q % len(params)]
        qc.ry(theta, q, label=f"$R_Y$({name})")
        if q > 0:
            qc.cx(q, q - 1)
        if q < n - 1:
            qc.cx(q, q + 1)
    qc.barrier()
    return qc


def apply_reupload_ladder(qc: QuantumCircuit, n: int, params: Sequence[float], name: str,
                           repeats: int) -> QuantumCircuit:
    """Apply ``apply_chain_ladder`` ``repeats`` times in a row, re-cycling
    through the same ``params`` each pass -- "data re-uploading" (Perez-
    Salinas et al., 2020): re-injecting the same classical data at multiple
    points in a fixed-depth circuit increases the expressivity of the
    resulting feature map without adding qubits. This is the structural
    signature observed in the user-supplied
    ``ReservoirCircuit_Nonlinear_GENE_Config1.qasm`` circuit (see
    ``qasm_reupload_dual_a`` below) -- not part of Stability_QRC_GS."""
    for _ in range(repeats):
        apply_chain_ladder(qc, n, params, name)
    return qc


# ==========================================================================
# Circuit recipes
# ==========================================================================

class CircuitSpec(NamedTuple):
    """One ensemble-selectable circuit family."""

    build: Callable[[QuantumCircuit, int, Optional[ParameterVector], ParameterVector, np.ndarray], None]
    recurrent: bool          # uses the P (previous-measurement) block?
    source: str              # provenance, for the gallery notebook / report
    description: str


def _build_magri_full_fullsym(qc, n, P, X, alpha):
    apply_chain_ladder(qc, n, P, "P")
    qc.barrier()
    apply_dense_entangler(qc, n, X, "X", mirror=False)
    qc.barrier()
    apply_dense_entangler(qc, n, alpha, "A", mirror=True)


def _build_magri_linear(qc, n, P, X, alpha):
    apply_chain_ladder(qc, n, P, "P")
    qc.barrier()
    apply_chain_ladder(qc, n, X, "X")
    qc.barrier()
    apply_chain_ladder(qc, n, alpha, "A")


def _build_magri_linear_x2(qc, n, P, X, alpha):
    apply_chain_ladder(qc, n, X, "X")
    qc.barrier()
    apply_chain_ladder(qc, n, X, "X")
    qc.barrier()
    apply_chain_ladder(qc, n, alpha, "A")


def _build_magri_full_x2(qc, n, P, X, alpha):
    apply_dense_entangler(qc, n, X, "X", mirror=False)
    qc.barrier()
    apply_dense_entangler(qc, n, X, "X", mirror=False)
    qc.barrier()
    apply_dense_entangler(qc, n, alpha, "A", mirror=True)


def _build_magri_feature_x2(qc, n, P, X, alpha):
    apply_pairwise_product_encoder(qc, n, X, "X")
    qc.barrier()
    apply_pairwise_product_encoder(qc, n, X, "X")
    qc.barrier()
    apply_chain_ladder(qc, n, alpha, "A")


def _build_magri_reverse_ent(qc, n, P, X, alpha):
    apply_reverse_sandwich_entangler(qc, n, X, "X")
    apply_reverse_sandwich_entangler(qc, n, X, "X")
    apply_chain_ladder(qc, n, alpha, "A")


def _build_magri_bidirectional(qc, n, P, X, alpha):
    apply_bidirectional_entangler(qc, n, X, "X")
    apply_bidirectional_entangler(qc, n, X, "X")
    apply_dense_entangler(qc, n, alpha, "A", mirror=True)


def _build_qasm_reupload_dual_a(qc, n, P, X, alpha):
    apply_reupload_ladder(qc, n, X, "X", repeats=n)
    qc.barrier()
    apply_dense_entangler(qc, n, alpha, "A", mirror=True)
    qc.barrier()
    apply_dense_entangler(qc, n, alpha, "A", mirror=True)


CIRCUIT_LIBRARY: Dict[str, CircuitSpec] = {
    "magri_full_fullsym": CircuitSpec(
        _build_magri_full_fullsym, True,
        "Stability_QRC_GS qrc.py config 1 (Unitary4 -> Unitary_FullyEnt -> Unitary_FullyEntSym)",
        "H -> P:chain-ladder (recurrent) -> X:dense-entangle -> A:dense-entangle-mirrored",
    ),
    "magri_linear": CircuitSpec(
        _build_magri_linear, True,
        "Stability_QRC_GS qrc.py config 2 (Unitary4 x3)",
        "H -> P:chain-ladder (recurrent) -> X:chain-ladder -> A:chain-ladder",
    ),
    "magri_linear_x2": CircuitSpec(
        _build_magri_linear_x2, False,
        "Stability_QRC_GS qrc.py config 3 (Unitary4 x3, no P)",
        "H -> X:chain-ladder x2 -> A:chain-ladder",
    ),
    "magri_full_x2": CircuitSpec(
        _build_magri_full_x2, False,
        "Stability_QRC_GS qrc.py config 4 (Unitary_FullyEnt x2 -> Unitary_FullyEntSym)",
        "H -> X:dense-entangle x2 -> A:dense-entangle-mirrored",
    ),
    "magri_feature_x2": CircuitSpec(
        _build_magri_feature_x2, False,
        "Stability_QRC_GS qrc.py config 5 (Unitary_Feature x2 -> Unitary4)",
        "H -> X:pairwise-product x2 -> A:chain-ladder",
    ),
    "magri_reverse_ent": CircuitSpec(
        _build_magri_reverse_ent, False,
        "Stability_QRC_GS unitaryblock.py Unitary_ReverseEnt (defined but unused by any qrc.py config) [NEW to ensemble]",
        "H -> X:reverse-sandwich x2 -> A:chain-ladder",
    ),
    "magri_bidirectional": CircuitSpec(
        _build_magri_bidirectional, False,
        "Stability_QRC_GS unitaryblock.py Unitary_C (defined but unused by any qrc.py config) [NEW to ensemble]",
        "H -> X:bidirectional-linear x2 -> A:dense-entangle-mirrored",
    ),
    "qasm_reupload_dual_a": CircuitSpec(
        _build_qasm_reupload_dual_a, False,
        "H -> X:chain-ladder x n_qubits (data re-upload) -> A:dense-entangle-mirrored x2",
    ),
}

CIRCUIT_CONFIG_INFO = {k: v.description for k, v in CIRCUIT_LIBRARY.items()}
RECURRENT_CONFIGS = {k for k, v in CIRCUIT_LIBRARY.items() if v.recurrent}


@dataclass
class QRCConfig:
    """Hyperparameters of a quantum-reservoir configuration."""

    n_qubits: int = 4
    dim: int = 3                                # input/output (plasma-state) dimension
    circuit_config: str = "magri_full_x2"       # key into CIRCUIT_LIBRARY
    epsilon_q: float = 0.3                      # leak rate on the measured-probability update
    input_scale: float = 2 * np.pi              # maps scaled input in [0,1] -> [0, input_scale)
    alpha_range: float = 2 * np.pi              # range of the fixed random reservoir (A) angles
    tikhonov: float = 1e-6
    seed: int = 0

    def __post_init__(self):
        if self.circuit_config not in CIRCUIT_LIBRARY:
            raise ValueError(
                f"circuit_config must be one of {sorted(CIRCUIT_LIBRARY)}, got {self.circuit_config!r}"
            )

    @property
    def recurrent(self) -> bool:
        return self.circuit_config in RECURRENT_CONFIGS


class QuantumReservoirComputer:
    def __init__(self, cfg: QRCConfig):
        self.cfg = cfg
        self.n_qubits = cfg.n_qubits
        self.dim = cfg.dim
        self.N_units = 2 ** cfg.n_qubits          # measured basis-probability vector size
        self.bias_out = np.array([1.0])
        self.spec = CIRCUIT_LIBRARY[cfg.circuit_config]
        self.recurrent = self.spec.recurrent

        self.alpha = self._draw_fixed_angles(cfg.seed)          # A block: fixed, baked in
        self.param_P = ParameterVector("P", self.N_units) if self.recurrent else None
        self.param_X = ParameterVector("X", cfg.dim)
        self.template = self._assemble_template()

    def _draw_fixed_angles(self, seed: int) -> np.ndarray:
        rng = np.random.RandomState(seed)
        return rng.uniform(0, self.cfg.alpha_range, size=self.n_qubits)

    # -- circuit construction ------------------------------------------------
    def _assemble_template(self) -> QuantumCircuit:
        """Build the per-instance parameterised circuit template: P and X
        are live ``Parameter`` placeholders (rebound every step via
        ``assign_parameters``); A (``self.alpha``, fixed for this instance's
        lifetime) is baked in as plain numbers."""
        n = self.n_qubits
        qc = QuantumCircuit(n)
        qc.h(range(min(self.dim, n)))
        self.spec.build(qc, n, self.param_P, self.param_X, self.alpha)
        return qc

    def bind(self, x_in: np.ndarray, prob_prev: Optional[np.ndarray] = None) -> QuantumCircuit:
        """Return the template with concrete numeric values substituted for
        P (if this config is recurrent) and X -- the per-timestep circuit
        actually simulated."""
        values = {p: float(x_in[j] * self.cfg.input_scale) for j, p in enumerate(self.param_X)}
        if self.recurrent:
            values.update({p: float(prob_prev[j]) for j, p in enumerate(self.param_P)})
        return self.template.assign_parameters(values)

    # -- one reservoir step ------------------------------------------------
    def step(self, prob_prev: np.ndarray, x_in: np.ndarray) -> np.ndarray:
        """Advance the reservoir by one step: bind + simulate the circuit,
        then apply the leaky update on the measured probability vector
        (``epsilon_q``) and append the readout bias."""
        qc = self.bind(x_in, prob_prev if self.recurrent else None)
        probs_new = Statevector.from_instruction(qc).probabilities()
        blended = (1 - self.cfg.epsilon_q) * prob_prev + self.cfg.epsilon_q * probs_new
        return np.hstack((blended, self.bias_out))

    def open_loop(self, U: np.ndarray, x0: np.ndarray) -> np.ndarray:
        """Teacher-forced rollout. Returns augmented state trajectory,
        shape (len(U)+1, N_units+1)."""
        N = U.shape[0]
        Xa = np.empty((N + 1, self.N_units + 1))
        Xa[0] = np.concatenate((x0, self.bias_out))
        for i in range(1, N + 1):
            Xa[i] = self.step(Xa[i - 1, :self.N_units], U[i - 1])
        return Xa

    def closed_loop(self, N: int, x0: np.ndarray, Wout: np.ndarray):
        """Autonomous rollout: the reservoir's own prediction is fed back in
        as the next timestep's input."""
        xa = x0.copy()
        Yh = np.empty((N + 1, self.dim))
        Yh[0] = np.dot(xa, Wout)
        for i in range(1, N + 1):
            xa = self.step(xa[:self.N_units], Yh[i - 1])
            Yh[i] = np.dot(xa, Wout)
        return Yh, xa

    # -- training (ridge regression readout, Block 4) -----------------------
    def train(self, U_washout: np.ndarray, U_train: np.ndarray, Y_train: np.ndarray,
              tikhonov: Optional[np.ndarray] = None):
        tikhonov = np.atleast_1d(tikhonov if tikhonov is not None else self.cfg.tikhonov)

        xf_washout = self.open_loop(U_washout, np.zeros(self.N_units))[-1, :self.N_units]
        Xa = self.open_loop(U_train, xf_washout)

        LHS = np.dot(Xa[1:].T, Xa[1:])
        RHS = np.dot(Xa[1:].T, Y_train)

        Wout = np.zeros((tikhonov.size, self.N_units + 1, self.dim))
        for j, tk in enumerate(tikhonov):
            Wout[j] = np.linalg.solve(LHS + tk * np.eye(self.N_units + 1), RHS)
        return Xa, Wout, xf_washout

    # -- convenience for the circuit-display notebook -----------------------
    def example_bound_circuit(self, seed: int = 0) -> QuantumCircuit:
        """A fully numeric example circuit (random example X in [0,1)^dim
        and, if recurrent, a random example probability vector for P) for
        drawing purposes -- exactly the structure run at every timestep,
        just with illustrative rather than live data."""
        rng = np.random.RandomState(seed)
        x_example = rng.uniform(0, 1, size=self.dim)
        p_example = rng.dirichlet(np.ones(self.N_units)) if self.recurrent else None
        return self.bind(x_example, p_example)
