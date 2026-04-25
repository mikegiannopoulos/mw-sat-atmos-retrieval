from __future__ import annotations

import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from mwsat.forward.simulator import simulate_brightness_temperature


def test_simulate_brightness_temperature_success() -> None:
    profile = {
        "pressure": [1000.0, 850.0, 700.0],
        "temperature": [290.0, 280.0, 270.0],
    }
    instrument_config = {
        "channels": {
            "center_frequencies_ghz": [50.3, 52.8, 54.4],
        }
    }

    result = simulate_brightness_temperature(profile, instrument_config)

    assert "tb" in result
    assert len(result["tb"]) == 3


def test_simulate_brightness_temperature_missing_profile_field() -> None:
    profile = {
        "pressure": [1000.0, 850.0, 700.0],
    }
    instrument_config = {
        "channels": {
            "center_frequencies_ghz": [50.3, 52.8],
        }
    }

    with pytest.raises(ValueError):
        simulate_brightness_temperature(profile, instrument_config)


def test_simulate_brightness_temperature_missing_instrument_channels() -> None:
    profile = {
        "pressure": [1000.0, 850.0, 700.0],
        "temperature": [290.0, 280.0, 270.0],
    }
    instrument_config = {}

    with pytest.raises(ValueError):
        simulate_brightness_temperature(profile, instrument_config)


def test_simulate_brightness_temperature_empty_channel_list() -> None:
    profile = {
        "pressure": [1000.0, 850.0, 700.0],
        "temperature": [290.0, 280.0, 270.0],
    }
    instrument_config = {
        "channels": {
            "center_frequencies_ghz": [],
        }
    }

    with pytest.raises(ValueError):
        simulate_brightness_temperature(profile, instrument_config)
