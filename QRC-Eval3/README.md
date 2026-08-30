# QRC on GENE Plasma Data

Helper library + two notebooks for running a Quantum Reservoir Computer (QRC) and a
classical Echo State Network ("linear layer") on real GENE nonlinear gyrokinetic
turbulence output, for the **Quantum ML for Real-Time Plasma Instability Prediction**
project.

Built to replace the toy Lorenz-63 problem used in the reference implementation
([MagriLab/Stability_QRC_GS](https://github.com/MagriLab/Stability_QRC_GS)) with the
project's actual GENE simulation data, following the block diagram and test cases in
`Major_Project_Presenation_1.pdf`.

## Structure

```
qrc_gene/                     the helper library (deliverable 1)
    gene_io.py                 GENE file readers (namelists, nrg, energy, geometry, binary field/mom)
    preprocessing.py            transient removal, scaling, noise injection, dataset splitting
    circuits.py                  parameterised quantum-circuit building blocks
    quantum_reservoir.py          QuantumReservoirComputer + QRCConfig  (Block 3A)
    classical_reservoir.py        EchoStateNetwork + ESNConfig ("linear layer", Block 3B)
    metrics.py                    mse / nmse / valid_prediction_time / early_warning_score
    experiments.py                shared load-dataset / train-eval glue used by both notebooks
    ensemble_search.py            config sampling + sweep runner for the ensemble notebook

notebooks/
    01_QRC_GENE_Main.ipynb                 deliverable 1's execution notebook
    02_QRC_Ensemble_Configuration.ipynb    deliverable 2: hyperparameter/config ensemble sweep

data/                          the GENE files (parameters.dat, nrg.dat,
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

### From raw diagnostics to the reservoir's input state `U(t)`

The reservoir's 3-D (or higher) input state is built from `nrg.dat`'s
spatially-averaged fluctuation-intensity columns (`n1_sq`, `T1par_sq`, `T1perp_sq` by
default — configurable), the GENE analogue of the reference repo's Lorenz-63 state
vector. `energy.dat`'s free-energy trace is used only to *automatically* detect the
end of the linear growth phase (`preprocessing.estimate_saturation_index`) so it can
be discarded as a transient, matching Block 2 ("transient removal") of the project's
block diagram.

### The binary `field.dat` / `mom_<species>.dat` files

`gene_io.read_gene_binary` (and the `read_field_file` / `read_mom_file` wrappers) is a
general Fortran-unformatted-sequential reader validated byte-for-byte against your
supplied `field_chunk_1.dat` / `mom_ions_chunk_1.dat` (record markers, snapshot size
`nx0*nky0*nz0*16` bytes, complex128 `(kx, ky, z)` layout). **The two chunk files you
provided are smaller than a single timestep** (a full snapshot is ~2.36 MB; the
supplied chunks are 0.9 MB and 3.7 MB respectively), so they were only usable to
validate the byte layout, not to build a time series — the reader detects this
correctly (`n_complete=0, truncated=True`) instead of crashing. It is ready to read
full field/mom files once available; a POD/PCA dimensionality-reduction step is
recommended before feeding that much higher-dimensional data into a reservoir (see
notebook 1's "Known limitations" section).

### The early-warning target (`Y_warning`)

The presentation's toy problem sweeps an explicit bifurcation parameter (Lorenz's
$\rho$: 15→30) to define "distance to the critical point". This GENE run is a single
**statistically stationary** nonlinear-saturated simulation — there is no swept
parameter — so `preprocessing.build_warning_signal` builds a physically motivated
proxy instead, from the rate of change of the ion heat flux $Q_{es}$ (the quantity
that characterises the actual transition from linear growth into turbulent
transport). This is documented in the module docstring and flagged clearly in the
notebook as a proxy, not a literal port of the toy problem's target.

## The two deliverables

1. **`qrc_gene/` + `01_QRC_GENE_Main.ipynb`** — the new helper library and its
   execution notebook: loads the real GENE run, preprocesses it, trains one QRC
   configuration and one classical-reservoir ("linear layer") configuration, and
   runs the three evaluation experiments from the presentation (early-warning
   signal, noise robustness, valid prediction time).
2. **`02_QRC_Ensemble_Configuration.ipynb`** — sweeps many QRC configurations
   (qubit count, circuit/encoding design, leak rate, Tikhonov regularisation) and
   many classical-reservoir configurations (reservoir size, spectral radius, input
   scaling, leak rate, density, Tikhonov regularisation) via random search, confirms
   the top candidates at full evaluation length, and reports the best layout and
   hyperparameters of each family plus an overall winner on a held-out test split.

Both notebooks were executed end-to-end in a Python 3.12 / Qiskit 2.5 environment
before being handed off, with outputs retained, so you can read the results without
re-running anything — but everything is fully reproducible (fixed seeds throughout).

## Known limitations / suggested next steps

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
