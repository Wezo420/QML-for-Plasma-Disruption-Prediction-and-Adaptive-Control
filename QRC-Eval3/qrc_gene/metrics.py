"""
metrics.py
==========

    TC-03 / TC-04  Noise robustness  -> mse / nmse
    TC-05 / TC-06  Prediction time   -> valid_prediction_time (VPT)
    TC-01 / TC-02  Early warning     -> early_warning_score
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((y_true - y_pred) ** 2))


def nmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """MSE normalised by the variance of the ground truth (so it is
    comparable across differently-scaled features/runs)."""
    denom = np.var(y_true) + 1e-12
    return float(np.mean((y_true - y_pred) ** 2) / denom)


def valid_prediction_time(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    dt: float,
    threshold: float = 0.4,
    lyapunov_time: Optional[float] = None,
) -> Tuple[float, int]:
    """Valid Prediction Time (VPT), matching the presentation's
    ("Derived Metric (VPT Evaluation)", slide 7): the closed-loop
    forecast is considered to have "failed" the first time the
    (per-timestep) Euclidean error between prediction and ground truth
    exceeds ``threshold``.

    Args:
        y_true, y_pred: (N, dim) arrays, closed-loop rollout vs ground truth.
        dt: physical time between samples.
        threshold: Euclidean-error failure threshold (0.4 in the
            presentation, defined on min-max-scaled [0,1] data).
        lyapunov_time: if given, VPT is also reported in units of
            Lyapunov times (as in the presentation / reference repo);
            otherwise VPT is reported in raw simulation time only.

    Returns:
        (t_fail, i_fail): failure time (or the full trajectory length if
        the error never crosses the threshold) and the corresponding
        sample index.
    """
    err = np.linalg.norm(y_true - y_pred, axis=1)
    over = np.where(err > threshold)[0]
    i_fail = int(over[0]) if len(over) else len(err)
    t_fail = i_fail * dt
    if lyapunov_time:
        return t_fail / lyapunov_time, i_fail
    return t_fail, i_fail


def early_warning_score(
    y_true_signal: np.ndarray,
    y_pred_signal: np.ndarray,
    peak_time_tolerance_frac: float = 0.05,
) -> dict:
    """Compare a predicted early-warning signal against the ground-truth
    warning signal (TC-01: "Signal Y peaks within +/-5% of the tipping
    point"; TC-02: "Signal Y remains flat and low during stability").

    Returns a dict with:
        peak_error_frac : |t_pred_peak - t_true_peak| / len(signal)
        pred_peak_value : predicted signal's peak amplitude (should be
                           high for a real transition, TC-01)
        pred_baseline    : predicted signal's median value away from the
                           peak (should be low & flat for TC-02)
    """
    i_true = int(np.argmax(y_true_signal))
    i_pred = int(np.argmax(y_pred_signal))
    peak_error_frac = abs(i_pred - i_true) / max(len(y_true_signal), 1)

    mask = np.ones(len(y_pred_signal), dtype=bool)
    window = max(int(peak_time_tolerance_frac * len(y_pred_signal)), 1)
    lo, hi = max(0, i_true - window), min(len(y_pred_signal), i_true + window)
    mask[lo:hi] = False

    return {
        "peak_error_frac": peak_error_frac,
        "within_tolerance": peak_error_frac <= peak_time_tolerance_frac,
        "pred_peak_value": float(y_pred_signal[i_pred]),
        "pred_baseline_median": float(np.median(y_pred_signal[mask])) if mask.any() else float("nan"),
    }
