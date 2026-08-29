"""
quantum_reservoir.py
=====================
Quantum Reservoir Computer (QRC) -- Block 3A of the project block diagram,
implementing the four stages shown on slide 9 ("Quantum Circuit
Demonstration") of the presentation:

    1. Data encoding & superposition   (Hadamard + RY(theta) rotation, theta
                                         linearly maps the scaled plasma
                                         state x_i in [0,1] to [0, 2*pi))
    2. Quantum reservoir evolution     (fixed random-angle entangling
                                         unitary -- never trained)
    3. Measurement & feature vector    (computational-basis probabilities
                                         of the output statevector)
    4. Classical readout & prediction  (ridge-regression Wout, trained --
                                         handled the same way as the
                                         classical reservoir for a fair
                                         QRC-vs-CRC comparison)

Architecture follows the general design of Ahmed, Tennie & Magri, "Robust
quantum reservoir computers for forecasting chaotic dynamics" and the
accompanying MagriLab Stability_QRC_GS reference implementation this
project is based on (https://github.com/MagriLab/Stability_QRC_GS,
``src/QRC/qrc.py``). This is an independent, simplified re-implementation:
statevector-only (no shot-noise / fake-backend emulation, which the
reference repo supports but which is not needed for this project's scope),
and with the closed-loop-Jacobian / Lyapunov machinery removed.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from .circuits import CIRCUIT_CONFIGS, build_reservoir_circuit


@dataclass
class QRCConfig:
    """Hyperparameters of a quantum-reservoir configuration -- the knobs
    an ensemble sweep varies."""

    n_qubits: int = 4
    dim: int = 3                       # input/output (plasma-state) dimension
    circuit_config: str = "full_fullsym"   # key into circuits.CIRCUIT_CONFIGS
    epsilon_q: float = 0.3             # leak rate on the measured-probability update
    input_scale: float = 2 * np.pi     # maps scaled input in [0,1] -> [0, input_scale)
    alpha_range: float = 2 * np.pi     # range of the fixed random reservoir angles
    tikhonov: float = 1e-6
    seed: int = 0

    def __post_init__(self):
        if self.circuit_config not in CIRCUIT_CONFIGS:
            raise ValueError(
                f"Unknown circuit_config {self.circuit_config!r}; "
                f"choose from {list(CIRCUIT_CONFIGS)}"
            )


class QuantumReservoirComputer:
    def __init__(self, cfg: QRCConfig):
        self.cfg = cfg
        self.n_qubits = cfg.n_qubits
        self.dim = cfg.dim
        # number of measured reservoir "units" = size of the computational
        # basis probability vector
        self.N_units = 2 ** cfg.n_qubits
        self.bias_out = np.array([1.0])
        self.x_encoder, self.alpha_encoder = CIRCUIT_CONFIGS[cfg.circuit_config]
        self.alpha = self._gen_fixed_angles(cfg.seed)

    def _gen_fixed_angles(self, seed: int) -> np.ndarray:
        rnd = np.random.RandomState(seed)
        return rnd.uniform(0, self.cfg.alpha_range, size=self.n_qubits)

    # -- one reservoir step ------------------------------------------------
    def _measure_probabilities(self, x_scaled: np.ndarray) -> np.ndarray:
        """Build the per-timestep circuit, evolve the statevector, and
        return the computational-basis probability vector (length
        N_units) -- Blocks 1-3 of the quantum circuit demonstration."""
        theta = x_scaled * self.cfg.input_scale
        qc = build_reservoir_circuit(
            self.n_qubits, theta, self.alpha,
            x_encoder=self.x_encoder, alpha_encoder=self.alpha_encoder,
        )
        sv = Statevector.from_instruction(qc)
        probs = np.abs(np.asarray(sv.data)) ** 2
        return probs

    def step(self, prob_prev: np.ndarray, x_in: np.ndarray) -> np.ndarray:
        """Advance the reservoir by one step, applying a leaky update on
        the measured-probability vector (the quantum analogue of the ESN
        leaky-integrator update -- the reference repo's ``epsilon_q``)."""
        prob_tilde = self._measure_probabilities(x_in)
        prob_new = (1 - self.cfg.epsilon_q) * prob_prev + self.cfg.epsilon_q * prob_tilde
        return np.hstack((prob_new, self.bias_out))

    def open_loop(self, U: np.ndarray, x0: np.ndarray) -> np.ndarray:
        """Drive the reservoir with a known (teacher-forced) input
        sequence. Returns the augmented state trajectory, shape
        (len(U)+1, N_units+1)."""
        N = U.shape[0]
        Xa = np.empty((N + 1, self.N_units + 1))
        Xa[0] = np.concatenate((x0, self.bias_out))
        for i in range(1, N + 1):
            Xa[i] = self.step(Xa[i - 1, :self.N_units], U[i - 1])
        return Xa

    def closed_loop(self, N: int, x0: np.ndarray, Wout: np.ndarray):
        """Autonomous rollout: the reservoir's own prediction is fed back
        in as the next timestep's input (the "Autonomous Feedback State"
        mode from the presentation, used for the Valid-Prediction-Time
        experiment)."""
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
