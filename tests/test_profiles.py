from __future__ import annotations

import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from mwsat.profiles import (
    load_era5_profile,
    load_igra_profile,
    validate_profile_data,
)


def test_validate_profile_data_success() -> None:
    profile = {
        "pressure": [1000.0, 900.0, 800.0],
        "temperature": [290.0, 285.0, 280.0],
    }

    result = validate_profile_data(profile)

    assert result is profile


def test_validate_profile_data_missing_field() -> None:
    profile = {
        "pressure": [1000.0, 900.0, 800.0],
    }

    with pytest.raises(ValueError):
        validate_profile_data(profile)


def test_validate_profile_data_wrong_type() -> None:
    with pytest.raises(TypeError):
        validate_profile_data(["not", "a", "dict"])


def test_load_era5_profile_not_implemented() -> None:
    with pytest.raises(NotImplementedError):
        load_era5_profile("dummy/path.nc")


def test_load_igra_profile_not_implemented() -> None:
    with pytest.raises(NotImplementedError):
        load_igra_profile("dummy/path.txt")
