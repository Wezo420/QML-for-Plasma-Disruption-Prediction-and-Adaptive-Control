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
Lorenz-63 rho-drift), there is no
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

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import gene_io


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


# --------------------------------------------------------------------------
# field.dat / mom_<species>.dat -> POD features (Eval-3 addition)
# --------------------------------------------------------------------------
#
# WHY THIS WASN'T USED BEFORE
# ----------------------------
# Every raw field/mom snapshot is a (nx0, nky0, nz0) complex array *per
# variable* -- for this run's grid (nx0=96, nky0=64, nz0=24) that's 147,456
# complex numbers = 2,359,296 bytes per variable per snapshot. field.dat has
# 1 variable/snapshot, mom_ions.dat has 6. Feeding that directly into a
# reservoir with a handful of qubits (or even a few hundred classical ESN
# units) makes no sense dimensionally, so *some* reduction step is required
# first -- consistent with the presentation's own "Physics-Informed Feature
# Extraction / PCA or POD" step.
#
# But dimensionality is a solvable problem; the actual blocker with the data
# supplied for this project is simpler and unconditional: the *chunk* files
# are smaller than a single snapshot. Verified directly against the supplied
# files (nx0=96, nky0=64, nz0=24 from parameters.dat):
#
#   field_chunk_1.dat:  943,718 bytes available; one full snapshot needs
#                       2,359,320 bytes (time record + 1 variable record).
#                       That's <40% of even the FIRST variable's payload.
#   mom_ions_chunk_1.dat: 3,879,731 bytes available; one full snapshot needs
#                       6 variable records = 14,155,840 bytes (+ time record).
#                       The chunk contains 1 complete variable plus ~64% of
#                       a second -- 0 of the 6 variables needed to call a
#                       snapshot "complete".
#
# So there are exactly ZERO complete 3D snapshots in either supplied chunk --
# not a code gap, a data-volume fact (see ``gene_io.snapshot_byte_requirements``
# for the arithmetic, and ``gene_io.read_gene_binary``, which detects this
# and reports ``n_complete=0, truncated=True`` rather than crashing).
#
# WHAT THIS SECTION ADDS
# -----------------------
# A real, tested reduction pipeline -- spectral energy by (kx, ky) mode,
# then POD (SVD-based PCA) down to a handful of coefficients per variable --
# that ``build_field_mom_features`` below will actually run and merge into
# the reservoir's input state the moment enough complete snapshots exist
# (either because a fuller ``field.dat``/``mom_ions.dat`` is supplied, or
# because more sequential chunks are concatenated with
# ``gene_io.concat_chunks``). Given only the single chunk of each file
# available today, this path reports *why* it's inactive (see
# ``FieldMomFeatureReport.reason``) and the rest of the pipeline proceeds on
# ``nrg.dat``/``energy.dat`` features alone, exactly as before.

def spectral_energy_by_k(snap: "gene_io.BinarySnapshots", var_index: int = 0) -> np.ndarray:
    """Reduce each complex (nx0, nky0, nz0) snapshot of one field/mom
    variable to a real (nx0, nky0) turbulent-energy-per-mode array by
    summing |amplitude|^2 over the parallel (z) direction -- the standard
    GENE k-spectrum diagnostic (GENE manual Sec. 4.2), and a physically
    meaningful reduction step rather than an arbitrary flatten. Returns
    shape (n_snap, nx0, nky0); ``n_snap`` may be 0 if ``snap`` has no
    complete snapshots."""
    if snap.n_complete == 0:
        nx0, nky0 = snap.data.shape[-3], snap.data.shape[-2]
        return np.empty((0, nx0, nky0))
    amp = snap.data[:, var_index]                    # (n_snap, nx0, nky0, nz0)
    return np.sum(np.abs(amp) ** 2, axis=-1)          # sum over z


def pod_features(spectra: np.ndarray, n_components: int = 3) -> Tuple[np.ndarray, Dict]:
    """POD (Proper Orthogonal Decomposition) of a (n_snap, ...) real
    spectrum time series via economy SVD of the mean-centred, flattened
    data matrix -- mathematically the same computation as PCA. Returns
    (coefficients, info) where ``coefficients`` has shape
    (n_snap, k) with ``k = min(n_components, n_snap-1, n_features)``, and
    ``info`` reports what was actually used (``k``, ``explained_variance_
    ratio``, whether ``n_components`` had to be reduced) so callers can
    log/display the true fidelity of the reduction rather than assuming it
    matched the request.

    If fewer than 2 snapshots are available, POD is not meaningful (you
    cannot estimate a covariance structure from 0 or 1 samples); an empty
    ``(n_snap, 0)`` coefficient array is returned with ``info["skipped"] =
    True`` and a human-readable ``info["reason"]``.
    """
    n_snap = spectra.shape[0]
    if n_snap < 2:
        return np.empty((n_snap, 0)), {
            "skipped": True,
            "reason": f"only {n_snap} complete snapshot(s) available; POD needs >= 2.",
            "k": 0,
        }

    flat = spectra.reshape(n_snap, -1).astype(float)
    mean = flat.mean(axis=0)
    centred = flat - mean

    k = min(n_components, n_snap - 1, flat.shape[1])
    U, S, Vt = np.linalg.svd(centred, full_matrices=False)
    coeffs = U[:, :k] * S[:k]
    total_var = (S ** 2).sum()
    explained = (S[:k] ** 2) / total_var if total_var > 0 else np.zeros(k)

    info = {
        "skipped": False,
        "k": k,
        "requested_components": n_components,
        "reduced_from_request": k < n_components,
        "explained_variance_ratio": explained,
        "mean": mean,
        "components": Vt[:k],
    }
    return coeffs, info


@dataclass
class FieldMomFeatureReport:
    """Diagnostic summary of one field/mom POD-feature-extraction attempt,
    surfaced to the notebook so the user sees *why* a source was or wasn't
    used rather than features silently appearing or not."""

    source: str                       # "field" or "mom"
    path: str
    used: bool
    n_complete: int
    truncated: bool
    bytes_available: int
    bytes_required_per_snapshot: int
    reason: str
    feature_names: List[str] = field(default_factory=list)


def build_field_mom_features(
    path: str | Path,
    run_info: "gene_io.GeneRunInfo",
    reader,                            # gene_io.read_field_file or read_mom_file
    source_label: str,
    n_components: int = 3,
    var_indices: Optional[Sequence[int]] = None,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], FieldMomFeatureReport]:
    """End-to-end attempt to build POD features from one field/mom file
    (or concatenated-chunk file): read -> per-variable spectral energy ->
    POD. Never raises on insufficient data; returns
    ``(times, features, report)`` with ``times=features=None`` when there
    aren't enough complete snapshots, and a filled-in ``report`` either
    way so the caller can log exactly what happened.
    """
    path = Path(path)
    bytes_available = path.stat().st_size
    snap = reader(path, run_info)

    n_vars = len(snap.var_names)
    req = gene_io.snapshot_byte_requirements(run_info, n_vars)
    var_indices = list(var_indices) if var_indices is not None else list(range(n_vars))

    if snap.n_complete == 0:
        reason = (
            f"0 complete snapshots in {path.name} ({bytes_available:,} bytes available; "
            f"one complete {source_label} snapshot needs "
            f"{req['snapshot_bytes']:,} bytes across {n_vars} variable(s))."
        )
        return None, None, FieldMomFeatureReport(
            source=source_label, path=str(path), used=False,
            n_complete=0, truncated=snap.truncated,
            bytes_available=bytes_available,
            bytes_required_per_snapshot=req["snapshot_bytes"],
            reason=reason,
        )

    all_coeffs, names = [], []
    for vi in var_indices:
        spectra = spectral_energy_by_k(snap, var_index=vi)
        coeffs, info = pod_features(spectra, n_components=n_components)
        if info.get("skipped") or coeffs.shape[1] == 0:
            continue
        all_coeffs.append(coeffs)
        vname = snap.var_names[vi]
        names.extend(f"{source_label}_{vname}_pod{j}" for j in range(coeffs.shape[1]))

    if not all_coeffs:
        reason = (
            f"{snap.n_complete} complete snapshot(s) in {path.name}, but that's still "
            f"below the minimum of 2 needed to fit POD."
        )
        return None, None, FieldMomFeatureReport(
            source=source_label, path=str(path), used=False,
            n_complete=snap.n_complete, truncated=snap.truncated,
            bytes_available=bytes_available,
            bytes_required_per_snapshot=req["snapshot_bytes"],
            reason=reason,
        )

    features = np.concatenate(all_coeffs, axis=1)
    report = FieldMomFeatureReport(
        source=source_label, path=str(path), used=True,
        n_complete=snap.n_complete, truncated=snap.truncated,
        bytes_available=bytes_available,
        bytes_required_per_snapshot=req["snapshot_bytes"],
        reason=f"used {snap.n_complete} complete snapshot(s).",
        feature_names=names,
    )
    return snap.times, features, report


def merge_field_mom_features(
    t_nrg: np.ndarray,
    U_nrg: np.ndarray,
    field_times: Optional[np.ndarray],
    field_features: Optional[np.ndarray],
    feature_names: Sequence[str],
) -> Tuple[np.ndarray, list]:
    """Resample POD field/mom features (built on the field/mom file's own,
    typically coarser, time grid -- e.g. istep_field=200 vs istep_nrg=10 in
    the supplied run) onto the nrg.dat time grid and append as extra
    columns, the same pattern as ``merge_energy_features``. If
    ``field_features`` is None (nothing usable was extracted), ``U_nrg`` is
    returned unchanged."""
    if field_features is None or field_features.shape[1] == 0:
        return U_nrg, []
    resampled = np.stack(
        [resample_to_grid(field_times, field_features[:, j], t_nrg)
         for j in range(field_features.shape[1])],
        axis=1,
    )
    return np.concatenate([U_nrg, resampled], axis=1), list(feature_names)
