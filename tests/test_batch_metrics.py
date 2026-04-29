from __future__ import annotations

import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from mwsat.evaluation.batch_metrics import aggregate_metrics


def test_aggregate_metrics_success() -> None:
    results = [
        {"metrics": {"bias": 1.0, "rmse": 2.0}},
        {"metrics": {"bias": 3.0, "rmse": 4.0}},
    ]

    summary = aggregate_metrics(results)

    assert summary["n_profiles"] == 2
    assert summary["mean_bias"] == 2.0
    assert summary["mean_rmse"] == 3.0


def test_aggregate_metrics_empty_input() -> None:
    with pytest.raises(ValueError):
        aggregate_metrics([])


def test_aggregate_metrics_missing_metrics() -> None:
    with pytest.raises(ValueError):
        aggregate_metrics([{}])


def test_aggregate_metrics_missing_metric_key() -> None:
    with pytest.raises(ValueError):
        aggregate_metrics([{"metrics": {"bias": 1.0}}])
