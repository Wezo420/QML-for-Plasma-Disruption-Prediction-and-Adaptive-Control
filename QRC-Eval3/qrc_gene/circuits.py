"""
circuits.py
===========
DEPRECATED as of Eval-3. This module used to define its own independent
circuit-block registry (``ENCODER_REGISTRY`` / ``CIRCUIT_CONFIGS``, string
keys like ``"linear_linear"``), separate from the one ``quantum_reservoir.py``
actually validates ``QRCConfig.circuit_config`` against. That split is
exactly what made ``ensemble_search.py``'s QRC sweep silently unusable
(``QRC_SEARCH_SPACE`` sampled keys from here, ``QRCConfig`` rejected all of
them) -- see the changelog comment at the top of ``ensemble_search.py``.

There is now exactly one circuit registry,
``quantum_reservoir.CIRCUIT_LIBRARY``, so it can't drift out of sync with
``QRCConfig`` again. This module is kept only so old notebook cells or
external scripts that still do ``from qrc_gene.circuits import
CIRCUIT_CONFIGS`` don't hard-crash; the names below are aliases onto the new
registry, not a second implementation. New code should import directly from
``quantum_reservoir`` instead.
"""
from __future__ import annotations

import warnings

from .quantum_reservoir import (
    CIRCUIT_LIBRARY,
    CIRCUIT_CONFIG_INFO,
    apply_chain_ladder,
    apply_dense_entangler,
    apply_pairwise_product_encoder,
)

warnings.warn(
    "qrc_gene.circuits is deprecated; import CIRCUIT_LIBRARY (and the "
    "apply_* block functions) from qrc_gene.quantum_reservoir instead.",
    DeprecationWarning,
    stacklevel=2,
)

# kept only for backward compatibility.
CIRCUIT_CONFIGS = CIRCUIT_CONFIG_INFO

# Old function names -> new implementations (signature differs slightly:
# the new versions take ``(qc, n, params, name)`` with ``qc`` first, matching
# every other Qiskit in-place-mutation convention; these thin wrappers
# restore the old ``(n, params, qc)`` argument order for any caller still
# using it).
def linear_entangler(n_qubits, params, qc):
    return apply_chain_ladder(qc, n_qubits, params, "X")


def fully_entangled(n_qubits, params, qc, symmetric=False):
    return apply_dense_entangler(qc, n_qubits, params, "X", mirror=symmetric)


def feature_map_products(n_qubits, params, qc):
    return apply_pairwise_product_encoder(qc, n_qubits, params, "X")


ENCODER_REGISTRY = {
    "linear": linear_entangler,
    "full": lambda n, p, qc: fully_entangled(n, p, qc, symmetric=False),
    "full_sym": lambda n, p, qc: fully_entangled(n, p, qc, symmetric=True),
    "feature_product": feature_map_products,
}
