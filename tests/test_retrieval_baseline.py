from __future__ import annotations

import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from mwsat.retrieval.baseline import retrieve_temperature_profile


def test_retrieve_temperature_profile_success() -> None:
    observation = {"tb": [270.0, 271.0, 272.0]}

    result = retrieve_temperature_profile(observation, {})

    assert "temperature" in result
    assert len(result["temperature"]) == 3
    assert result["temperature"] == [271.0, 271.0, 271.0]


def test_retrieve_temperature_profile_with_configured_levels() -> None:
    observation = {"tb": [270.0, 271.0, 272.0]}
    retrieval_config = {
        "retrieval": {
            "vertical_grid": {
                "n_levels": 5,
            }
        }
    }

    result = retrieve_temperature_profile(observation, retrieval_config)

    assert len(result["temperature"]) == 5


def test_retrieve_temperature_profile_missing_tb() -> None:
    with pytest.raises(ValueError):
        retrieve_temperature_profile({}, {})


def test_retrieve_temperature_profile_empty_tb() -> None:
    with pytest.raises(ValueError):
        retrieve_temperature_profile({"tb": []}, {})
