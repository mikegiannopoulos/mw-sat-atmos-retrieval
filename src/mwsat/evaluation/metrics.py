from __future__ import annotations

import numpy as np


def compute_bias(reference: list[float], estimate: list[float]) -> float:
    """Compute mean bias for retrieval or forward-model evaluation."""
    if len(reference) != len(estimate):
        raise ValueError("Reference and estimate must have the same length")
    if not reference:
        raise ValueError("Reference and estimate must not be empty")

    reference_values = np.asarray(reference, dtype=float)
    estimate_values = np.asarray(estimate, dtype=float)
    return float(np.mean(estimate_values - reference_values))


def compute_rmse(reference: list[float], estimate: list[float]) -> float:
    """Compute RMSE for retrieval or forward-model evaluation."""
    if len(reference) != len(estimate):
        raise ValueError("Reference and estimate must have the same length")
    if not reference:
        raise ValueError("Reference and estimate must not be empty")

    reference_values = np.asarray(reference, dtype=float)
    estimate_values = np.asarray(estimate, dtype=float)
    return float(np.sqrt(np.mean((estimate_values - reference_values) ** 2)))
