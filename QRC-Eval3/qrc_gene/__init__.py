"""
qrc_gene
========
Helper library that maps GENE nonlinear gyrokinetic simulation output to a
Quantum Reservoir Computer (QRC) / classical Echo State Network (ESN)
pipeline, for the "Quantum Machine Learning for Real-Time Plasma
Instability Prediction & Adaptive Control" project.

Submodules:
    gene_io               GENE file readers (namelists, nrg, energy,
                           geometry, binary field/mom files)
    preprocessing          transient removal, scaling, noise injection,
                           forecast-pair / washout-train-val-test splitting
    circuits               parameterised quantum-circuit building blocks
    quantum_reservoir       QuantumReservoirComputer + QRCConfig
    classical_reservoir     EchoStateNetwork + ESNConfig ("linear layer")
    metrics                 mse / nmse / valid_prediction_time / early_warning_score
"""
from . import gene_io, preprocessing, circuits, metrics, experiments, ensemble_search
from .quantum_reservoir import QuantumReservoirComputer, QRCConfig
from .classical_reservoir import EchoStateNetwork, ESNConfig

__all__ = [
    "gene_io", "preprocessing", "circuits", "metrics", "experiments", "ensemble_search",
    "QuantumReservoirComputer", "QRCConfig",
    "EchoStateNetwork", "ESNConfig",
]
