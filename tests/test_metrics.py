from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from mwsat.evaluation.metrics import compute_bias, compute_rmse


def test_compute_bias() -> None:
    reference = [1, 2, 3]
    estimate = [2, 2, 4]

    result = compute_bias(reference, estimate)

    assert result == pytest.approx(2 / 3)


def test_compute_rmse() -> None:
    reference = [1, 2, 3]
    estimate = [2, 2, 4]

    result = compute_rmse(reference, estimate)

    assert result == pytest.approx(math.sqrt(2 / 3))


def test_metrics_mismatched_lengths() -> None:
    with pytest.raises(ValueError):
        compute_bias([1, 2], [1])

    with pytest.raises(ValueError):
        compute_rmse([1, 2], [1])


def test_metrics_empty_inputs() -> None:
    with pytest.raises(ValueError):
        compute_bias([], [])

    with pytest.raises(ValueError):
        compute_rmse([], [])
