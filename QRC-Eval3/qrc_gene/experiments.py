"""
experiments.py
===============
Glue code shared by the two deliverable notebooks:

    01_QRC_GENE_Main.ipynb              -- single-configuration run + the
                                            three evaluation experiments
    02_QRC_Ensemble_Configuration.ipynb -- sweeps many QRC / ESN configs
                                            using the same building blocks

Keeping this logic here (rather than duplicated in notebook cells) means
both notebooks train/evaluate reservoirs identically, so ensemble results
are directly comparable to the main notebook's single-run results.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from . import gene_io, preprocessing as pp
from .classical_reservoir import EchoStateNetwork, ESNConfig
from .quantum_reservoir import QuantumReservoirComputer, QRCConfig
from .metrics import mse, nmse, valid_prediction_time


@dataclass
class GeneDataset:
    """Bundle of everything downstream code needs: the split/scaled
    state-forecasting dataset plus the raw ingredients (time grid,
    warning-signal proxy, run metadata) needed by the three experiments."""

    split: pp.DatasetSplit
    scaler: pp.MinMaxScaler
    feature_names: list
    run_info: gene_io.GeneRunInfo
    t_nrg: np.ndarray
    Q_es_raw: np.ndarray            # for the early-warning experiment
    dt: float


def load_and_prepare_gene_dataset(
    data_dir: str | Path,
    feature_columns=("n1_sq", "T1par_sq", "T1perp_sq"),
    energy_columns=(),
    frac_washout: float = 0.10,
    frac_train: float = 0.55,
    frac_val: float = 0.15,
    frac_test: float = 0.20,
    saturation_frac_of_max_growth: float = 0.9,
) -> GeneDataset:
    """End-to-end Block 1 + Block 2 pipeline: load the raw GENE files,
    remove the linear-growth transient, build the state-forecasting
    dataset and split it into washout/train/val/test, exactly as
    validated interactively while building this helper.
    """
    data_dir = Path(data_dir)

    run_info = gene_io.load_run_info(data_dir / "parameters.dat")
    t_nrg, nrg, cols = gene_io.read_nrg(
        data_dir / "nrg.dat", n_spec=run_info.n_spec, nrgcols=run_info.nrgcols
    )
    energy = gene_io.read_energy(data_dir / "energy.dat")

    sat_idx = pp.estimate_saturation_index(energy["Etot"], saturation_frac_of_max_growth)
    sat_time = energy["time"][sat_idx]
    n_transient = int(np.searchsorted(t_nrg, sat_time))

    U_nrg, feat_names = pp.nrg_to_feature_matrix(nrg, cols, use_columns=list(feature_columns))
    if energy_columns:
        U_nrg, extra_names = pp.merge_energy_features(t_nrg, U_nrg, energy, energy_columns)
        feat_names = feat_names + extra_names

    t_trim, U_trim = pp.remove_transient(t_nrg, U_nrg, n_transient)

    split, scaler = pp.build_dataset(
        t_trim, U_trim,
        frac_washout=frac_washout, frac_train=frac_train,
        frac_val=frac_val, frac_test=frac_test, horizon=1,
    )

    Q_es_raw = nrg[:, 0, cols.index("Q_es")]
    dt = float(np.mean(np.diff(split.t)))

    return GeneDataset(
        split=split, scaler=scaler, feature_names=feat_names, run_info=run_info,
        t_nrg=t_nrg, Q_es_raw=Q_es_raw, dt=dt,
    )


# --------------------------------------------------------------------------
# Train + closed-loop validation for a single configuration
# --------------------------------------------------------------------------

_SPLIT_ORDER = ["washout", "train", "val", "test"]


def _advance_state(reservoir, ds: GeneDataset, upto_split: str) -> np.ndarray:
    """Teacher-force the reservoir through every split *before*
    ``upto_split`` (in washout -> train -> val -> test order) and return
    its final raw (non-augmented) state.

    This matters whenever ``eval_split`` is not the split immediately
    after training: e.g. to evaluate on ``test`` the reservoir must first
    be driven through ``val`` too (using the true, teacher-forced ``val``
    data), otherwise the closed-loop rollout on ``test`` starts from a
    stale internal state left over from the end of training and fails
    immediately regardless of how good the configuration actually is.
    """
    idx = _SPLIT_ORDER.index(upto_split)
    state = np.zeros(reservoir.N_units)
    for split_name in _SPLIT_ORDER[:idx]:
        U_split, _ = getattr(ds.split, split_name)
        state = reservoir.open_loop(U_split, state)[-1, :reservoir.N_units]
    return state


def train_eval_esn(cfg: ESNConfig, ds: GeneDataset, eval_split: str = "val",
                    threshold: float = 0.4):
    """Train an ESN on washout+train, teacher-force it through any splits
    between training and ``eval_split`` (see ``_advance_state``), then run
    a closed-loop rollout over ``eval_split`` and score it. Returns a
    result dict (used both by the main notebook and by the ensemble
    sweep)."""
    Uw, Yw = ds.split.washout
    Ut, Yt = ds.split.train
    U_eval, Y_eval = getattr(ds.split, eval_split)

    esn = EchoStateNetwork(cfg)
    Xa, Wout, xf_train = esn.train(Uw, Ut, Yt, tikhonov=np.array([cfg.tikhonov]))

    xf = xf_train if eval_split == "val" else _advance_state(esn, ds, eval_split)
    x0 = np.concatenate([xf, [1.0]])
    Yh, _ = esn.closed_loop(len(U_eval) - 1, x0, Wout[0])

    return {
        "model": "ESN",
        "mse": mse(Y_eval, Yh),
        "nmse": nmse(Y_eval, Yh),
        "vpt_time": valid_prediction_time(Y_eval, Yh, dt=ds.dt, threshold=threshold)[0],
        "Y_true": Y_eval,
        "Y_pred": Yh,
        "reservoir": esn,
        "Wout": Wout[0],
        "cfg": cfg,
    }


def train_eval_qrc(cfg: QRCConfig, ds: GeneDataset, eval_split: str = "val",
                    threshold: float = 0.4, max_eval_len: Optional[int] = None):
    """Same as ``train_eval_esn`` but for a QuantumReservoirComputer.
    ``max_eval_len`` caps the closed-loop rollout length, since each step
    runs a statevector simulation and full-length rollouts (hundreds of
    steps) can be slow for larger qubit counts -- useful for quick
    ensemble screening before a full-length confirmation run."""
    Uw, Yw = ds.split.washout
    Ut, Yt = ds.split.train
    U_eval, Y_eval = getattr(ds.split, eval_split)
    if max_eval_len is not None:
        U_eval, Y_eval = U_eval[:max_eval_len], Y_eval[:max_eval_len]

    qrc = QuantumReservoirComputer(cfg)
    Xa, Wout, xf_train = qrc.train(Uw, Ut, Yt, tikhonov=np.array([cfg.tikhonov]))

    xf = xf_train if eval_split == "val" else _advance_state(qrc, ds, eval_split)
    x0 = np.concatenate([xf, [1.0]])
    Yh, _ = qrc.closed_loop(len(U_eval) - 1, x0, Wout[0])

    return {
        "model": "QRC",
        "mse": mse(Y_eval, Yh),
        "nmse": nmse(Y_eval, Yh),
        "vpt_time": valid_prediction_time(Y_eval, Yh, dt=ds.dt, threshold=threshold)[0],
        "Y_true": Y_eval,
        "Y_pred": Yh,
        "reservoir": qrc,
        "Wout": Wout[0],
        "cfg": cfg,
    }
