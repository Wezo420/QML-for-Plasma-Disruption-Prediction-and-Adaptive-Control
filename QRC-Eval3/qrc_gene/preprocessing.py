"""
preprocessing.py
=================
Turns raw GENE diagnostic time series (as returned by ``gene_io``) into the
washout / train / val / test tensors expected by the reservoir classes,
mirroring Block 2 ("Pre-processing") of the project block diagram:

    transient removal -> normalization -> (optional) noise injection
    -> washout split

This module also builds the *targets* used by the three evaluation
experiments described in the project presentation ("Overview of
Implementation", slide 7):

    Y_clean   : clean future state           -> forecasting target
    Y_warning : scalar "critical proximity"  -> early-warning target
    U_noisy   : sensor-corrupted input       -> noise-robustness input

Because this GENE run is a single *statistically stationary* nonlinear
saturated-turbulence simulation (not a swept-parameter run like the
Lorenz-63 rho-drift toy problem in the reference repo), there is no
explicit ground-truth bifurcation parameter to build Y_warning from.
Instead we build a physically motivated proxy from the run itself: the
electrostatic heat flux Q_es (the quantity that actually characterises
the level of turbulent transport / "how unstable" the plasma is at a
given instant) and the free-energy drive term. Both are documented in
the GENE manual (Sec. 4.1, 4.6) as the physical quantities that grow
during the transition from linear growth to nonlinear saturation - the
closest real-physics analogue of "approaching the critical point".
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np


# --------------------------------------------------------------------------
# Scaling
# --------------------------------------------------------------------------

@dataclass
class MinMaxScaler:
    """Per-feature min-max scaler to [0, 1], fit on a training window only
    (never on val/test) -- matches "Min-Max scaled to a [0,1] range" from
    the project presentation (slide 7)."""

    feature_min: Optional[np.ndarray] = None
    feature_max: Optional[np.ndarray] = None
    eps: float = 1e-12

    def fit(self, U: np.ndarray) -> "MinMaxScaler":
        self.feature_min = U.min(axis=0)
        self.feature_max = U.max(axis=0)
        return self

    def transform(self, U: np.ndarray) -> np.ndarray:
        rng = np.maximum(self.feature_max - self.feature_min, self.eps)
        return (U - self.feature_min) / rng

    def fit_transform(self, U: np.ndarray) -> np.ndarray:
        return self.fit(U).transform(U)

    def inverse_transform(self, Us: np.ndarray) -> np.ndarray:
        rng = np.maximum(self.feature_max - self.feature_min, self.eps)
        return Us * rng + self.feature_min


# --------------------------------------------------------------------------
# Feature construction from nrg.dat / energy.dat
# --------------------------------------------------------------------------

def nrg_to_feature_matrix(
    nrg: np.ndarray,
    columns: Sequence[str],
    species_index: int = 0,
    use_columns: Optional[Sequence[str]] = None,
    log_transform: bool = True,
) -> Tuple[np.ndarray, list]:
    """Turn the (n_t, n_spec, n_cols) nrg array into a (n_t, n_features)
    state matrix U(t), analogous to the 3D Lorenz state used in the
    reference repository.

    Fluctuation-amplitude columns (n1_sq, u1par_sq, T1par_sq, T1perp_sq)
    and flux columns (Gamma_es, Q_es, ...) are strictly non-negative and
    typically span several orders of magnitude as the simulation
    transitions from linear growth to nonlinear saturation, so a log10
    transform (log10(x + eps)) is applied by default before scaling --
    this is the "Physics-Informed Feature Extraction" step from the
    Data Set Description slide, adapted for nrg quantities.

    Args:
        nrg: array from ``gene_io.read_nrg``.
        columns: column names returned alongside ``nrg``.
        species_index: which species row to use (0 for a single-species run).
        use_columns: subset of column names to keep as features; defaults
            to all available columns.
        log_transform: apply log10(x + eps) before returning.

    Returns:
        (U, feature_names)
    """
    use_columns = list(use_columns) if use_columns is not None else list(columns)
    idx = [columns.index(c) for c in use_columns]
    U = nrg[:, species_index, idx].astype(float)
    if log_transform:
        eps = 1e-30
        U = np.log10(np.abs(U) + eps)
    return U, use_columns


def resample_to_grid(t_src: np.ndarray, y_src: np.ndarray, t_target: np.ndarray) -> np.ndarray:
    """Linear interpolation of a (possibly coarser/finer sampled) series
    onto a target time grid -- the "Temporal Alignment" step from the
    Data Set Description slide, needed because nrg.dat (istep_nrg) and
    energy.dat (istep_energy) are written at different cadences."""
    return np.interp(t_target, t_src, y_src)


def merge_energy_features(
    t_nrg: np.ndarray,
    U_nrg: np.ndarray,
    energy: Dict[str, np.ndarray],
    energy_columns: Sequence[str] = ("Etot", "dEdt_drive"),
) -> Tuple[np.ndarray, list]:
    """Resample selected energy.dat columns onto the nrg.dat time grid and
    append them as extra features."""
    extra = []
    names = []
    for col in energy_columns:
        y = resample_to_grid(energy["time"], energy[col], t_nrg)
        extra.append(y)
        names.append(col)
    extra = np.stack(extra, axis=1) if extra else np.empty((len(t_nrg), 0))
    U_full = np.concatenate([U_nrg, extra], axis=1)
    return U_full, names


# --------------------------------------------------------------------------
# Transient removal
# --------------------------------------------------------------------------

def remove_transient(t: np.ndarray, U: np.ndarray, n_transient: int) -> Tuple[np.ndarray, np.ndarray]:
    """Discard the first ``n_transient`` samples (linear growth phase,
    before the turbulence has saturated) -- Block 2 "transient removal"
    step. See also ``estimate_saturation_index`` to pick n_transient
    automatically from the data."""
    return t[n_transient:], U[n_transient:]


def estimate_saturation_index(Etot: np.ndarray, frac_of_max_growth: float = 0.9) -> int:
    """Heuristic estimate of when the simulation exits the linear growth
    phase and enters nonlinear saturation, using the total free energy
    Etot: returns the first index at which dEtot/dt drops below
    ``frac_of_max_growth`` of its peak value (i.e. growth has slowed down
    substantially, indicating saturation)."""
    dEdt = np.gradient(Etot)
    peak = dEdt.max()
    if peak <= 0:
        return 0
    below = np.where(dEdt < frac_of_max_growth * peak)[0]
    candidates = below[below > np.argmax(dEdt)]
    return int(candidates[0]) if len(candidates) else 0


# --------------------------------------------------------------------------
# Noise injection ("Corrupted State Vector" from the presentation)
# --------------------------------------------------------------------------

def add_sensor_noise(U: np.ndarray, eta: float, seed: Optional[int] = None) -> np.ndarray:
    """U_noisy(t) = U_clean(t) + N(0, eta * sigma_U), matching the
    "Corrupted State Vector" definition on slide 7 of the presentation.
    ``eta`` is the noise level (e.g. 0.01 for 1%, 0.10 for 10%)."""
    rng = np.random.RandomState(seed)
    sigma = U.std(axis=0, keepdims=True)
    noise = rng.normal(loc=0.0, scale=eta * sigma, size=U.shape)
    return U + noise


# --------------------------------------------------------------------------
# Early-warning proxy target
# --------------------------------------------------------------------------

def build_warning_signal(
    Q_es: np.ndarray,
    window: int = 25,
) -> np.ndarray:
    """Physically motivated proxy for the "Critical Proximity Signal"
    Y_warning from the presentation (slide 7), built from the
    electrostatic heat flux Q_es. Q_es rises sharply as the simulation
    transitions from linear growth into nonlinear turbulent saturation
    (the GENE-data analogue of "approaching the critical bifurcation
    point"). The signal is defined as the (smoothed) flux normalized to
    peak at 1.0 at the point of maximum growth rate of the flux itself,
    i.e. the moment of fastest transition into turbulence:

        Y(t) = smoothed(dQ_es/dt) / max(smoothed(dQ_es/dt))

    clipped to [0, 1]. This preserves the qualitative property requested
    in the presentation's Test Cases (TC-01/TC-02): the signal should
    spike near the transition and stay low & flat during quasi-steady
    turbulence.
    """
    Q = np.asarray(Q_es, dtype=float)
    if window > 1:
        kernel = np.ones(window) / window
        Q_smooth = np.convolve(Q, kernel, mode="same")
    else:
        Q_smooth = Q
    dQ = np.gradient(Q_smooth)
    dQ = np.clip(dQ, 0, None)  # only rising transport signals "danger"
    peak = dQ.max()
    if peak <= 0:
        return np.zeros_like(dQ)
    return np.clip(dQ / peak, 0.0, 1.0)


# --------------------------------------------------------------------------
# Supervised dataset construction (open-loop forecasting target)
# --------------------------------------------------------------------------

def make_forecast_pairs(U: np.ndarray, horizon: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """Build (U(t), Y(t)=U(t+horizon)) pairs for one-step-ahead
    forecasting, matching "Clean Future State" (Y_clean) from the
    presentation."""
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    X = U[:-horizon]
    Y = U[horizon:]
    return X, Y


@dataclass
class DatasetSplit:
    """Washout / train / val / test split of a (already scaled) state
    time series, following the reference repo's N_washout / N_train /
    N_val / N_test convention."""

    t: np.ndarray
    U: np.ndarray
    Y: np.ndarray
    n_washout: int
    n_train: int
    n_val: int
    n_test: int

    def __post_init__(self):
        total_needed = self.n_washout + self.n_train + self.n_val + self.n_test
        if total_needed > len(self.U):
            raise ValueError(
                f"Requested washout+train+val+test={total_needed} samples "
                f"but only {len(self.U)} are available. Reduce split sizes "
                f"or provide a longer run."
            )

    @property
    def washout(self):
        s = 0
        return self.U[s:s + self.n_washout], self.Y[s:s + self.n_washout]

    @property
    def train(self):
        s = self.n_washout
        return self.U[s:s + self.n_train], self.Y[s:s + self.n_train]

    @property
    def val(self):
        s = self.n_washout + self.n_train
        return self.U[s:s + self.n_val], self.Y[s:s + self.n_val]

    @property
    def test(self):
        s = self.n_washout + self.n_train + self.n_val
        return self.U[s:s + self.n_test], self.Y[s:s + self.n_test]

    def time_of(self, split: str) -> np.ndarray:
        bounds = {
            "washout": (0, self.n_washout),
            "train": (self.n_washout, self.n_washout + self.n_train),
            "val": (self.n_washout + self.n_train,
                     self.n_washout + self.n_train + self.n_val),
            "test": (self.n_washout + self.n_train + self.n_val,
                      self.n_washout + self.n_train + self.n_val + self.n_test),
        }
        s, e = bounds[split]
        return self.t[s:e]


def build_dataset(
    t: np.ndarray,
    U_raw: np.ndarray,
    frac_washout: float = 0.10,
    frac_train: float = 0.60,
    frac_val: float = 0.15,
    frac_test: float = 0.15,
    horizon: int = 1,
    scaler: Optional[MinMaxScaler] = None,
) -> Tuple[DatasetSplit, MinMaxScaler]:
    """Convenience wrapper: fits a MinMaxScaler on the *training portion
    only*, scales the whole series with it, builds one-step-ahead forecast
    pairs, and returns a ready-to-use ``DatasetSplit``.

    Split sizes are given as fractions of the total usable length
    (``len(U_raw) - horizon``) so the same call works regardless of how
    many transient samples were already removed upstream.
    """
    fracs = np.array([frac_washout, frac_train, frac_val, frac_test])
    if not np.isclose(fracs.sum(), 1.0, atol=1e-6):
        raise ValueError("fractions must sum to 1.0")

    X_all, Y_all = make_forecast_pairs(U_raw, horizon=horizon)
    t_all = t[:-horizon]
    n_total = len(X_all)

    n_washout = int(n_total * frac_washout)
    n_train = int(n_total * frac_train)
    n_val = int(n_total * frac_val)
    n_test = n_total - n_washout - n_train - n_val  # absorb rounding

    if scaler is None:
        scaler = MinMaxScaler()
        train_slice = slice(n_washout, n_washout + n_train)
        scaler.fit(X_all[train_slice])

    X_scaled = scaler.transform(X_all)
    Y_scaled = scaler.transform(Y_all)

    split = DatasetSplit(
        t=t_all, U=X_scaled, Y=Y_scaled,
        n_washout=n_washout, n_train=n_train, n_val=n_val, n_test=n_test,
    )
    return split, scaler
