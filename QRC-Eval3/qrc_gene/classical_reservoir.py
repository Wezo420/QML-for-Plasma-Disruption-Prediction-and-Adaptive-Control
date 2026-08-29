"""
classical_reservoir.py
=======================
Classical Echo State Network (ESN) reservoir -- Block 3B ("Classical
Reservoir CRC/ESN") of the project block diagram, used as the baseline
"linear layer" configuration that the QRC is benchmarked against
(Results & Discussion, slides 10-11 of the presentation: "QRC vs CRC").

This is a self-contained re-implementation of the standard leaky-integrator
ESN update

    x(t+1) = (1-eps) x(t) + eps * tanh( Win [u(t); 1] * sigma_in + rho * W x(t) )

used throughout reservoir-computing literature and in the reference
implementation this project is based on
(https://github.com/MagriLab/Stability_QRC_GS, ``src/QRC/crc.py``), with
the closed-loop-Jacobian / Lyapunov-exponent machinery removed since it is
out of scope for this project's deliverables (early-warning, noise
robustness and valid-prediction-time evaluation only).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.sparse import csr_matrix, lil_matrix
from scipy.sparse.linalg import eigs as sparse_eigs


@dataclass
class ESNConfig:
    """Hyperparameters of a classical reservoir ("linear layer")
    configuration -- the knobs an ensemble sweep varies."""

    N_units: int = 200          # reservoir size
    dim: int = 3                # input/output dimension
    rho: float = 0.6            # spectral radius scaling
    sigma_in: float = 0.5       # input scaling
    epsilon: float = 1.0        # leak rate (1.0 = no leaking)
    density: float = 0.2        # reservoir connectivity density
    tikhonov: float = 1e-6      # ridge-regression regularisation
    seed: int = 0                # RNG seed for Win / W


class EchoStateNetwork:
    def __init__(self, cfg: ESNConfig):
        self.cfg = cfg
        self.N_units = cfg.N_units
        self.dim = cfg.dim
        self.bias_in = np.array([1.0])
        self.bias_out = np.array([1.0])
        self.Win = self._gen_input_matrix(cfg.seed)
        self.W = self._gen_reservoir_matrix(cfg.seed + 1)

    # -- construction -----------------------------------------------------
    def _gen_input_matrix(self, seed: int):
        rnd = np.random.RandomState(seed)
        Win = lil_matrix((self.N_units, self.dim + 1))
        for j in range(self.N_units):
            Win[j, rnd.randint(0, self.dim + 1)] = rnd.uniform(-1, 1)
        return Win.tocsr()

    def _gen_reservoir_matrix(self, seed: int):
        rnd = np.random.RandomState(seed)
        W = csr_matrix(
            rnd.uniform(-1, 1, (self.N_units, self.N_units))
            * (rnd.rand(self.N_units, self.N_units) < self.cfg.density)
        )
        spectral_radius = np.abs(sparse_eigs(W, k=1, which="LM", return_eigenvectors=False))[0]
        return (1.0 / spectral_radius) * W

    # -- dynamics -----------------------------------------------------------
    def step(self, x_pre: np.ndarray, u: np.ndarray) -> np.ndarray:
        u_aug = np.hstack((u, self.bias_in))
        pre_activation = self.Win.dot(u_aug * self.cfg.sigma_in) + self.W.dot(self.cfg.rho * x_pre)
        x_post = (1 - self.cfg.epsilon) * x_pre + self.cfg.epsilon * np.tanh(pre_activation)
        return np.hstack((x_post, self.bias_out))

    def open_loop(self, U: np.ndarray, x0: np.ndarray) -> np.ndarray:
        """Drive the reservoir with a known input sequence ``U``
        (teacher-forced). Returns the augmented state trajectory,
        shape (len(U)+1, N_units+1)."""
        N = U.shape[0]
        Xa = np.empty((N + 1, self.N_units + 1))
        Xa[0] = np.concatenate((x0, self.bias_out))
        for i in range(1, N + 1):
            Xa[i] = self.step(Xa[i - 1, :self.N_units], U[i - 1])
        return Xa

    def closed_loop(self, N: int, x0: np.ndarray, Wout: np.ndarray):
        """Autonomous rollout: the reservoir's own prediction from the
        previous step is fed back in as the next input (the "Autonomous
        Feedback State" mode from the presentation)."""
        xa = x0.copy()
        Yh = np.empty((N + 1, self.dim))
        Yh[0] = np.dot(xa, Wout)
        for i in range(1, N + 1):
            xa = self.step(xa[:self.N_units], Yh[i - 1])
            Yh[i] = np.dot(xa, Wout)
        return Yh, xa

    # -- training (ridge regression readout) ---------------------------
    def train(self, U_washout: np.ndarray, U_train: np.ndarray, Y_train: np.ndarray,
              tikhonov: Optional[np.ndarray] = None):
        """Washout then open-loop train, solving the ridge-regression
        readout for one or more Tikhonov regularisation values at once
        (Block 4 "Ridge regression / Tikhonov regularization").

        Returns:
            Xa: augmented training-state trajectory
            Wout: array of shape (len(tikhonov), N_units+1, dim)
        """
        tikhonov = np.atleast_1d(tikhonov if tikhonov is not None else self.cfg.tikhonov)

        xf_washout = self.open_loop(U_washout, np.zeros(self.N_units))[-1, :self.N_units]
        Xa = self.open_loop(U_train, xf_washout)

        LHS = np.dot(Xa[1:].T, Xa[1:])
        RHS = np.dot(Xa[1:].T, Y_train)

        Wout = np.zeros((tikhonov.size, self.N_units + 1, self.dim))
        for j, tk in enumerate(tikhonov):
            Wout[j] = np.linalg.solve(LHS + tk * np.eye(self.N_units + 1), RHS)
        return Xa, Wout, xf_washout
