from __future__ import annotations

import sys
from pathlib import Path

import pytest
import xarray as xr


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from mwsat.profiles import (
    load_igra_profile,
    load_profile,
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


def test_load_igra_profile_not_implemented() -> None:
    with pytest.raises(NotImplementedError):
        load_igra_profile("dummy/path.txt")


def test_load_profile_era5_success(tmp_path: Path) -> None:
    dataset = xr.Dataset(
        data_vars={
            "t": (("time", "level"), [[280.0, 275.0, 270.0]]),
        },
        coords={
            "time": [0],
            "level": [1000.0, 850.0, 700.0],
        },
    )
    path = tmp_path / "profile_era5.nc"
    dataset.to_netcdf(path)

    profile = load_profile("era5", str(path))

    assert set(profile) == {"pressure", "temperature"}


def test_load_profile_igra_not_implemented() -> None:
    with pytest.raises(NotImplementedError):
        load_profile("igra", "dummy/path.txt")


def test_load_profile_unknown_source() -> None:
    with pytest.raises(ValueError):
        load_profile("unknown", "dummy/path")
