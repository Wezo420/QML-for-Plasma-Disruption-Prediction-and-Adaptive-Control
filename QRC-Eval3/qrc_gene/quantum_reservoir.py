"""
quantum_reservoir.py
=====================
Quantum Reservoir Computer (QRC) -- Block 3A of the project block diagram.

Circuit design ported/adapted from the unitary-block library and 5 named
circuit configurations used in
https://github.com/MagriLab/OptimalTraining_RFQRC (``src/QRC/unitaryblock.py``,
``src/QRC/qrc.py``), which is the more advanced sibling of
https://github.com/MagriLab/Stability_QRC_GS this project originally
referenced. Each reservoir step applies up to three labelled blocks after an
initial Hadamard layer:

    P  (recurrence)  -- re-encodes the reservoir's OWN previous measurement
                         (all 2**n_qubits basis probabilities) as rotation
                         angles -- true *quantum* memory, not just a
                         classical leaky average. Only configs 1-2 use it.
    X  (input)        -- encodes the current (scaled) plasma-state input.
    A  (reservoir)     -- a fixed, never-trained random-angle entangling
                          unitary that gives the reservoir its fixed
                          internal dynamics.

    Config 1: H -> P:linear-ent -> X:full-ent      -> A:full-ent-symmetric
    Config 2: H -> P:linear-ent -> X:linear-ent     -> A:linear-ent
    Config 3: H ->       (no P) -> X:linear-ent x2  -> A:linear-ent
    Config 4: H ->       (no P) -> X:full-ent x2    -> A:full-ent-symmetric
    Config 5: H ->       (no P) -> X:feature-product x2 -> A:linear-ent

("x2" = the input block is applied twice in sequence, deepening the
encoding.) Configs 1-2 give the richest (quantum-recurrent) memory but cost
O(2**n_qubits) extra gates per step for the P block; configs 3-5 use
classical-leaky memory only (via epsilon_q on the measured probabilities,
same mechanism as before) and are much cheaper for larger qubit counts.

For speed, each reservoir builds ONE parameterised circuit *template* per
instance (Qiskit ``ParameterVector`` placeholders for P and X; the fixed
random block A is baked in as plain numbers immediately, since it never
changes), then rebinds only the numeric P/X values every timestep via
``assign_parameters`` instead of rebuilding circuit topology from scratch.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Optional

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import Statevector

# --------------------------------------------------------------------------
# Unitary building blocks (ported from OptimalTraining_RFQRC/src/QRC/unitaryblock.py)
# --------------------------------------------------------------------------


def _unitary_linear(n: int, params, qc: QuantumCircuit, name: str) -> QuantumCircuit:
    """'Unitary4': RY rotations cycling through qubits 0..n-1 (wrapping if
    there are more params than qubits), each followed by a nearest-neighbour
    CNOT that zig-zags 0->1->...->n-1->n-2->... at the boundary. Used for
    the P (recurrence) block and as a cheap X/A block in configs 2-3."""
    for j, p in enumerate(params):
        i = j % n
        qc.ry(p, i, label=f"$R_Y$({name})")
        if i <= 1:
            qc.cx(i, i + 1)
        elif i % (n - 1) == 0:
            qc.cx(i, i - 1)
        else:
            qc.cx(i, i + 1)
        if i == n - 1:
            qc.barrier()
    return qc


def _unitary_full_ent(n: int, params, qc: QuantumCircuit, name: str,
                       symmetric: bool = False) -> QuantumCircuit:
    """'Unitary_FullyEnt' / 'Unitary_FullyEntSym': RY on every qubit using
    the first n params, an all-to-all CNOT entangling layer (every pair
    connected once), then -- if ``symmetric`` (the 'A' block variant) -- a
    second RY layer re-using the same params, or -- if not symmetric (the
    'X' block variant) -- an RY layer using any *remaining* params beyond
    the first n."""
    params = list(params)
    for j in range(min(len(params), n)):
        qc.ry(params[j], j, label=f"$R_Y$({name})")
    for i, k in combinations(range(n), 2):
        qc.cx(i, k)
    if symmetric:
        for j in range(min(len(params), n)):
            qc.ry(params[j], j, label=f"$R_Y$({name})")
    else:
        for j in range(n, len(params)):
            qc.ry(params[j], j % n, label=f"$R_Y$({name})")
    return qc


def _unitary_feature(n: int, params, qc: QuantumCircuit, name: str) -> QuantumCircuit:
    """'Unitary_Feature': RZ encoding of each input, then for every qubit
    pair (i,k) a CNOT-RY(param_i*param_k)-CNOT sandwich that injects the
    pairwise *product* of two input features as an entangling angle -- a
    genuinely nonlinear (quadratic) feature map baked into the circuit."""
    params = list(params)
    n_use = min(len(params), n)
    for j in range(n_use):
        qc.rz(params[j], j)
    for i, k in combinations(range(n_use), 2):
        qc.cx(i, k)
        qc.ry(params[i] * params[k], k, label=f"$R_Y$({name}_i{name}_j)")
        qc.cx(i, k)
    return qc


# Human-readable description of each of the 5 configs, used by the display
# notebook and error messages.
CIRCUIT_CONFIG_INFO = {
    1: "H -> P:linear (recurrent) -> X:full-ent -> A:full-ent-symmetric",
    2: "H -> P:linear (recurrent) -> X:linear -> A:linear",
    3: "H -> X:linear x2 (no recurrence) -> A:linear",
    4: "H -> X:full-ent x2 (no recurrence) -> A:full-ent-symmetric",
    5: "H -> X:feature-product x2 (no recurrence) -> A:linear",
}
RECURRENT_CONFIGS = {1, 2}


@dataclass
class QRCConfig:
    """Hyperparameters of a quantum-reservoir configuration."""

    n_qubits: int = 4
    dim: int = 3                      # input/output (plasma-state) dimension
    circuit_config: int = 4           # 1-5, see CIRCUIT_CONFIG_INFO
    epsilon_q: float = 0.3            # leak rate on the measured-probability update
    input_scale: float = 2 * np.pi    # maps scaled input in [0,1] -> [0, input_scale)
    alpha_range: float = 2 * np.pi    # range of the fixed random reservoir (A) angles
    tikhonov: float = 1e-6
    seed: int = 0

    def __post_init__(self):
        if self.circuit_config not in CIRCUIT_CONFIG_INFO:
            raise ValueError(f"circuit_config must be one of {sorted(CIRCUIT_CONFIG_INFO)}")

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
        self.recurrent = cfg.recurrent

        self.alpha = self._gen_fixed_angles(cfg.seed)          # A block: fixed, baked in
        self.param_P = ParameterVector("P", self.N_units) if self.recurrent else None
        self.param_X = ParameterVector("X", cfg.dim)
        self.template = self._build_template()

    def _gen_fixed_angles(self, seed: int) -> np.ndarray:
        rnd = np.random.RandomState(seed)
        return rnd.uniform(0, self.cfg.alpha_range, size=self.n_qubits)

    # -- circuit construction ------------------------------------------------
    def _build_template(self) -> QuantumCircuit:
        """Build the per-instance parameterised circuit template: P and X
        are live ``Parameter`` placeholders (rebound every step via
        ``assign_parameters``); A (``self.alpha``, fixed for this
        instance's lifetime) is baked in as plain numbers."""
        n, cfg = self.n_qubits, self.cfg
        qc = QuantumCircuit(n)
        qc.h(range(min(cfg.dim, n)))

        if cfg.circuit_config == 1:
            _unitary_linear(n, self.param_P, qc, "P")
            qc.barrier()
            _unitary_full_ent(n, self.param_X, qc, "X", symmetric=False)
            qc.barrier()
            _unitary_full_ent(n, self.alpha, qc, "A", symmetric=True)
        elif cfg.circuit_config == 2:
            _unitary_linear(n, self.param_P, qc, "P")
            qc.barrier()
            _unitary_linear(n, self.param_X, qc, "X")
            qc.barrier()
            _unitary_linear(n, self.alpha, qc, "A")
        elif cfg.circuit_config == 3:
            _unitary_linear(n, self.param_X, qc, "X")
            qc.barrier()
            _unitary_linear(n, self.param_X, qc, "X")
            qc.barrier()
            _unitary_linear(n, self.alpha, qc, "A")
        elif cfg.circuit_config == 4:
            _unitary_full_ent(n, self.param_X, qc, "X", symmetric=False)
            qc.barrier()
            _unitary_full_ent(n, self.param_X, qc, "X", symmetric=False)
            qc.barrier()
            _unitary_full_ent(n, self.alpha, qc, "A", symmetric=True)
        elif cfg.circuit_config == 5:
            _unitary_feature(n, self.param_X, qc, "X")
            qc.barrier()
            _unitary_feature(n, self.param_X, qc, "X")
            qc.barrier()
            _unitary_linear(n, self.alpha, qc, "A")
        return qc

    def bind(self, x_in: np.ndarray, prob_prev: Optional[np.ndarray] = None) -> QuantumCircuit:
        """Return the template with concrete numeric values substituted for
        P (if this config is recurrent) and X -- the per-timestep circuit
        actually simulated."""
        values = {}
        for j, p in enumerate(self.param_X):
            values[p] = float(x_in[j] * self.cfg.input_scale)
        if self.recurrent:
            for j, p in enumerate(self.param_P):
                values[p] = float(prob_prev[j])
        return self.template.assign_parameters(values)

    # -- one reservoir step ------------------------------------------------
    def step(self, prob_prev: np.ndarray, x_in: np.ndarray) -> np.ndarray:
        """Advance the reservoir by one step: bind + simulate the circuit,
        then apply the leaky update on the measured probability vector
        (the reference repo's ``epsilon_q``) and append the readout bias."""
        qc = self.bind(x_in, prob_prev if self.recurrent else None)
        sv = Statevector.from_instruction(qc)
        prob_tilde = np.abs(np.asarray(sv.data)) ** 2
        prob_new = (1 - self.cfg.epsilon_q) * prob_prev + self.cfg.epsilon_q * prob_tilde
        return np.hstack((prob_new, self.bias_out))

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
        """Autonomous rollout: the reservoir's own prediction is fed back
        in as the next timestep's input."""
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
        rnd = np.random.RandomState(seed)
        x_example = rnd.uniform(0, 1, size=self.dim)
        if self.recurrent:
            p_example = rnd.dirichlet(np.ones(self.N_units))  # valid probability vector
            return self.bind(x_example, p_example)
        return self.bind(x_example, None)
