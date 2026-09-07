"""
qrc_gene
========
Helper library that maps GENE nonlinear gyrokinetic simulation output to a
Quantum Reservoir Computer (QRC) / classical Echo State Network (ESN)
pipeline, for the "Quantum Machine Learning for Real-Time Plasma
Instability Prediction & Adaptive Control" project.

Submodules:
    gene_io               GENE file readers (namelists, nrg, energy,
                           geometry, binary field/mom files); also
                           field/mom chunk-diagnostic + concat utilities
    preprocessing          transient removal, scaling, noise injection,
                           forecast-pair / washout-train-val-test splitting,
                           field/mom POD-feature extraction
    circuits               DEPRECATED -- thin compatibility shim over
                           quantum_reservoir.CIRCUIT_LIBRARY, see its docstring
    quantum_reservoir       QuantumReservoirComputer + QRCConfig +
                           CIRCUIT_LIBRARY (8 circuit families -- see its
                           module docstring for the full list and provenance)
    classical_reservoir     EchoStateNetwork + ESNConfig ("linear layer")
    metrics                 mse / nmse / valid_prediction_time / early_warning_score
    experiments             shared dataset-loading / train-eval glue
    ensemble_search         config sampling + sweep runner for the ensemble notebook
"""
# Note: the deprecated `circuits` submodule is intentionally NOT imported
# eagerly here (unlike the others) so that simply doing `import qrc_gene`
# doesn't emit its DeprecationWarning on every normal use of the package --
# it's still available via `from qrc_gene import circuits` for anything that
# still needs it, which is exactly when the warning should fire.
from . import gene_io, preprocessing, metrics, experiments, ensemble_search
from .quantum_reservoir import QuantumReservoirComputer, QRCConfig, CIRCUIT_LIBRARY
from .classical_reservoir import EchoStateNetwork, ESNConfig

__all__ = [
    "gene_io", "preprocessing", "metrics", "experiments", "ensemble_search",
    "QuantumReservoirComputer", "QRCConfig", "CIRCUIT_LIBRARY",
    "EchoStateNetwork", "ESNConfig",
]
