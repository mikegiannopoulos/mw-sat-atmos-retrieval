from __future__ import annotations

import numpy as np

from mwsat.instrument.config import InstrumentConfig


def apply_noise(
    y_true: np.ndarray, instrument: InstrumentConfig, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    noise_std = np.array([channel.nedt_k for channel in instrument.channels]).reshape(
        y_true.shape
    )
    noise = rng.normal(0.0, noise_std, size=y_true.shape)
    return y_true + noise, noise
