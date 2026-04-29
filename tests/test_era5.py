from __future__ import annotations

import sys
from pathlib import Path

import pytest
import xarray as xr


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from mwsat.profiles.era5 import load_era5_profile


def test_load_era5_profile_success_with_level_and_t(tmp_path: Path) -> None:
    dataset = xr.Dataset(
        data_vars={
            "t": (("time", "level"), [[280.0, 275.0, 270.0]]),
        },
        coords={
            "time": [0],
            "level": [1000.0, 850.0, 700.0],
        },
    )
    path = tmp_path / "era5_level_t.nc"
    dataset.to_netcdf(path)

    profile = load_era5_profile(str(path))

    assert set(profile) == {"pressure", "temperature"}
    assert len(profile["pressure"]) == 3
    assert len(profile["temperature"]) == 3


def test_load_era5_profile_success_with_pressure_and_temperature(tmp_path: Path) -> None:
    dataset = xr.Dataset(
        data_vars={
            "temperature": (("profile", "pressure"), [[290.0, 285.0, 280.0]]),
        },
        coords={
            "profile": [0],
            "pressure": [1000.0, 900.0, 800.0],
        },
    )
    path = tmp_path / "era5_pressure_temperature.nc"
    dataset.to_netcdf(path)

    profile = load_era5_profile(str(path))

    assert set(profile) == {"pressure", "temperature"}
    assert len(profile["pressure"]) == 3
    assert len(profile["temperature"]) == 3


def test_load_era5_profile_missing_pressure(tmp_path: Path) -> None:
    dataset = xr.Dataset(
        data_vars={
            "t": (("time", "z"), [[280.0, 275.0, 270.0]]),
        },
        coords={
            "time": [0],
            "z": [0, 1, 2],
        },
    )
    path = tmp_path / "missing_pressure.nc"
    dataset.to_netcdf(path)

    with pytest.raises(
        ValueError,
        match="vertical coordinate named 'pressure', 'level', or 'pressure_level'",
    ):
        load_era5_profile(str(path))


def test_load_era5_profile_missing_temperature(tmp_path: Path) -> None:
    dataset = xr.Dataset(
        data_vars={
            "q": (("time", "level"), [[0.1, 0.2, 0.3]]),
        },
        coords={
            "time": [0],
            "level": [1000.0, 850.0, 700.0],
        },
    )
    path = tmp_path / "missing_temperature.nc"
    dataset.to_netcdf(path)

    with pytest.raises(ValueError, match="temperature variable named 'temperature' or 't'"):
        load_era5_profile(str(path))


def test_load_era5_profile_invalid_pressure_structure(tmp_path: Path) -> None:
    dataset = xr.Dataset(
        data_vars={
            "pressure": (("x", "y"), [[1000.0, 900.0], [800.0, 700.0]]),
            "temperature": (("time", "level"), [[290.0, 285.0, 280.0]]),
        },
        coords={
            "time": [0],
            "x": [0, 1],
            "y": [0, 1],
            "level": [1000.0, 850.0, 700.0],
        },
    )
    path = tmp_path / "invalid_pressure_structure.nc"
    dataset.to_netcdf(path)

    with pytest.raises(ValueError, match="one-dimensional after squeezing"):
        load_era5_profile(str(path))
