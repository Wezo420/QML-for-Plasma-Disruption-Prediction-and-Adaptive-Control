# QRC on GENE Plasma Data

Helper library + two notebooks for running a Quantum Reservoir Computer (QRC) and a
classical Echo State Network on real GENE nonlinear gyrokinetic
turbulence output, for the **Quantum ML for Real-Time Plasma Instability Prediction**
project.

Built to replace the Lorenz-63 attractor used in the reference implementation
([MagriLab/Stability_QRC_GS](https://github.com/MagriLab/Stability_QRC_GS)) with the
project's actual GENE simulation data.

## Structure

```
qrc_gene/                     the helper library
    gene_io.py                 GENE file readers (namelists, nrg, energy, geometry, binary field/mom)
    preprocessing.py            transient removal, scaling, noise injection, dataset splitting
    circuits.py                  parameterised quantum-circuit building blocks
    quantum_reservoir.py          QuantumReservoirComputer + QRCConfig  (Block 3A)
    classical_reservoir.py        EchoStateNetwork + ESNConfig (Block 3B)
    metrics.py                    mse / nmse / valid_prediction_time / early_warning_score
    experiments.py                shared load-dataset / train-eval glue used by both notebooks
    ensemble_search.py            config sampling + sweep runner for the ensemble notebook

notebooks/
    01_QRC_GENE_Main.ipynb                 execution notebook
    02_QRC_Ensemble_Configuration.ipynb    hyperparameter/config ensemble sweep

data/                          GENE files (parameters.dat, nrg.dat,
                                energy.dat, circular.dat, field/mom binary chunks)

requirements.txt
```

## Quick start

```bash
pip install -r requirements.txt
jupyter notebook notebooks/01_QRC_GENE_Main.ipynb
```

Both notebooks add the repo root to `sys.path` and import `qrc_gene` directly — no
package installation step needed, just keep the folder structure intact.

## How the GENE parameter files are mapped and used

Everything the helper needs is **read out of `parameters.dat`'s Fortran namelists at
runtime** (via `gene_io.load_run_info`) — nothing about the run is hard-coded:

| Symbol read from `parameters.dat` | Namelist | Used for |
|---|---|---|
| `nx0`, `nky0`, `nz0` | `&box` | shape of every binary snapshot in `field.dat` / `mom_<species>.dat` |
| `n_spec` | `&box` | number of per-species blocks in `nrg.dat` |
| `n_fields` | `&info` | number of variables per field-file snapshot (1 = electrostatic-only, as in the supplied run) |
| `n_moms` | `&info` | number of variables per mom-file snapshot (6 in the current run: `n1, T1par, T1perp, q1par+1.5p0u1par, q1perp+p0u1par, u1par`) |
| `nrgcols` | `&info` | number of columns per `nrg.dat` species line (8 or 10; the supplied run has 10, i.e. toroidal-angular-momentum columns included) |
| `dt_max` | `&general` | base simulation timestep, used to convert sample counts to physical time (e.g. in the VPT metric) |

This run-info object is then passed to `gene_io.read_field_file` / `read_mom_file` /
`read_nrg` so every reader always uses the *actual* grid/species/column counts for
**this specific run**, not assumed defaults — the same helper works unmodified on a
different GENE run with a different resolution or species count.


## Known limitations/ Next steps

* Only `nrg.dat` (and optionally `energy.dat`) diagnostics are used as reservoir
  input; the high-dimensional field/moment data is not yet exercised end-to-end
  (see above).
* The early-warning experiment uses a physically motivated *proxy* target rather
  than a literal bifurcation-parameter sweep, since only one (statistically
  stationary) nonlinear run was supplied. A genuine drift experiment would need a
  short parameter scan (several GENE runs at different gradient-drive values).
* The ensemble notebook uses random search by default; `qrc_gene.ensemble_search`
  also exposes `grid_esn_configs` / `grid_qrc_configs` for an exhaustive sweep over
  a narrowed-down space once you have a promising region (and HPC time to spend —
  the QRC statevector simulation cost grows as $2^{n\_qubits}$).
