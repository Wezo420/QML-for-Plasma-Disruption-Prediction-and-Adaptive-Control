"""
ensemble_search.py
====================
Utilities for the Ensemble Configuration notebook
(``02_QRC_Ensemble_Configuration.ipynb``): randomised sampling of QRC /
ESN ("linear layer") hyperparameter configurations from a declared search
space, and a runner that trains + evaluates each one and collects the
results into a tidy ``pandas.DataFrame`` leaderboard.

Design note: a full grid over every hyperparameter combination grows
combinatorially (e.g. 4 qubit counts x 6 circuit types x 4 leak rates x 3
Tikhonov values = 288 QRC configs alone) and most of that grid is
uninformative. We therefore default to *random search* over declared
ranges (Bergstra & Bengio, 2012 -- random search is a strong, simple
baseline for this kind of hyperparameter space), while still exposing a
``mode="grid"`` option for a smaller, exhaustive sweep when that's what's
wanted (e.g. for the final confirmation run over a narrowed-down space).
"""
from __future__ import annotations

import itertools
import time
from dataclasses import asdict
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .classical_reservoir import ESNConfig
from .quantum_reservoir import QRCConfig
from .circuits import CIRCUIT_CONFIGS
from .experiments import GeneDataset, train_eval_esn, train_eval_qrc


# --------------------------------------------------------------------------
# Search-space definitions
# --------------------------------------------------------------------------

ESN_SEARCH_SPACE: Dict[str, Sequence] = {
    "N_units": [50, 100, 200, 400],
    "rho": [0.2, 0.4, 0.6, 0.8, 1.0, 1.2],
    "sigma_in": [0.05, 0.1, 0.3, 0.5, 1.0],
    "epsilon": [0.3, 0.5, 0.7, 0.9, 1.0],
    "density": [0.1, 0.2, 0.3],
    "tikhonov": [1e-8, 1e-6, 1e-4, 1e-2],
}

QRC_SEARCH_SPACE: Dict[str, Sequence] = {
    "n_qubits": [3, 4, 5, 6, 7, 8],
    "circuit_config": list(CIRCUIT_CONFIGS.keys()),
    "epsilon_q": [0.1, 0.2, 0.3, 0.5, 0.7, 1.0],
    "input_scale": [np.pi, 1.5 * np.pi, 2 * np.pi],
    "tikhonov": [1e-8, 1e-6, 1e-4, 1e-2],
}


def sample_esn_configs(
    n: int, dim: int, seed: int = 0, space: Optional[Dict[str, Sequence]] = None,
) -> List[ESNConfig]:
    """Randomly sample ``n`` distinct ESN configurations from ``space``
    (defaults to ``ESN_SEARCH_SPACE``). Each sampled config gets its own
    reservoir-construction seed (derived from ``seed``) so results are
    reproducible run-to-run."""
    space = space or ESN_SEARCH_SPACE
    rnd = np.random.RandomState(seed)
    configs, seen = [], set()
    attempts = 0
    while len(configs) < n and attempts < n * 20:
        attempts += 1
        choice = {k: rnd.choice(v) for k, v in space.items()}
        key = tuple(choice.items())
        if key in seen:
            continue
        seen.add(key)
        cfg = ESNConfig(
            N_units=int(choice["N_units"]), dim=dim, rho=float(choice["rho"]),
            sigma_in=float(choice["sigma_in"]), epsilon=float(choice["epsilon"]),
            density=float(choice["density"]), tikhonov=float(choice["tikhonov"]),
            seed=int(rnd.randint(0, 1_000_000)),
        )
        configs.append(cfg)
    return configs


def sample_qrc_configs(
    n: int, dim: int, seed: int = 0, space: Optional[Dict[str, Sequence]] = None,
) -> List[QRCConfig]:
    """Randomly sample ``n`` distinct QRC configurations from ``space``
    (defaults to ``QRC_SEARCH_SPACE``)."""
    space = space or QRC_SEARCH_SPACE
    rnd = np.random.RandomState(seed)
    configs, seen = [], set()
    attempts = 0
    while len(configs) < n and attempts < n * 20:
        attempts += 1
        choice = {k: rnd.choice(v) if k != "circuit_config" else rnd.choice(v)
                   for k, v in space.items()}
        key = tuple(choice.items())
        if key in seen:
            continue
        seen.add(key)
        cfg = QRCConfig(
            n_qubits=int(choice["n_qubits"]), dim=dim,
            circuit_config=str(choice["circuit_config"]),
            epsilon_q=float(choice["epsilon_q"]), input_scale=float(choice["input_scale"]),
            tikhonov=float(choice["tikhonov"]),
            seed=int(rnd.randint(0, 1_000_000)),
        )
        configs.append(cfg)
    return configs


def grid_esn_configs(dim: int, space: Dict[str, Sequence], seed: int = 0) -> List[ESNConfig]:
    """Exhaustive grid over a (small!) search space -- intended for a
    narrowed-down confirmation sweep, not the initial broad search."""
    keys = list(space.keys())
    configs = []
    for combo in itertools.product(*[space[k] for k in keys]):
        choice = dict(zip(keys, combo))
        configs.append(ESNConfig(
            N_units=int(choice["N_units"]), dim=dim, rho=float(choice["rho"]),
            sigma_in=float(choice["sigma_in"]), epsilon=float(choice["epsilon"]),
            density=float(choice.get("density", 0.2)), tikhonov=float(choice["tikhonov"]),
            seed=seed,
        ))
    return configs


def grid_qrc_configs(dim: int, space: Dict[str, Sequence], seed: int = 0) -> List[QRCConfig]:
    keys = list(space.keys())
    configs = []
    for combo in itertools.product(*[space[k] for k in keys]):
        choice = dict(zip(keys, combo))
        configs.append(QRCConfig(
            n_qubits=int(choice["n_qubits"]), dim=dim,
            circuit_config=str(choice["circuit_config"]),
            epsilon_q=float(choice["epsilon_q"]),
            input_scale=float(choice.get("input_scale", 2 * np.pi)),
            tikhonov=float(choice["tikhonov"]), seed=seed,
        ))
    return configs


# --------------------------------------------------------------------------
# Sweep runner
# --------------------------------------------------------------------------

def run_sweep(
    configs: Sequence,
    train_eval_fn: Callable,
    ds: GeneDataset,
    label: str,
    eval_split: str = "val",
    threshold: float = 0.4,
    verbose: bool = True,
    **train_eval_kwargs,
) -> pd.DataFrame:
    """Train + evaluate every config in ``configs`` (a list of ESNConfig or
    QRCConfig) and return a results DataFrame sorted by validation MSE.

    Rows that raise an exception (e.g. a pathological reservoir matrix)
    are recorded with ``mse=NaN`` rather than aborting the whole sweep,
    so one bad hyperparameter draw doesn't lose the rest of the sweep's
    results.
    """
    rows = []
    for i, cfg in enumerate(configs):
        t0 = time.time()
        try:
            res = train_eval_fn(cfg, ds, eval_split=eval_split, threshold=threshold,
                                 **train_eval_kwargs)
            row = {
                "family": label, "config_id": i,
                "mse": res["mse"], "nmse": res["nmse"], "vpt_time": res["vpt_time"],
                "wall_time_s": time.time() - t0, "error": None,
            }
        except Exception as exc:  # noqa: BLE001 - deliberately broad for a sweep
            row = {
                "family": label, "config_id": i,
                "mse": np.nan, "nmse": np.nan, "vpt_time": np.nan,
                "wall_time_s": time.time() - t0, "error": str(exc),
            }
        row.update({f"cfg.{k}": v for k, v in asdict(cfg).items()})
        rows.append(row)
        if verbose:
            status = "OK" if row["error"] is None else f"FAILED ({row['error']})"
            print(f"[{label} {i+1:>3}/{len(configs)}] mse={row['mse']!s:>10}  "
                  f"({row['wall_time_s']:.2f}s)  {status}")

    df = pd.DataFrame(rows)
    return df.sort_values("mse", na_position="last").reset_index(drop=True)
