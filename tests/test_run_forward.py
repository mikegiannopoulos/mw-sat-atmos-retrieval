from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest
import xarray as xr


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from mwsat.pipeline.run_forward import run_forward_simulation
from mwsat.utils.config import load_all_configs


def test_run_forward_simulation_success(tmp_path: Path) -> None:
    dataset = xr.Dataset(
        data_vars={
            "t": (("time", "level"), [[290.0, 285.0, 275.0]]),
        },
        coords={
            "time": [0],
            "level": [1000.0, 850.0, 700.0],
        },
    )
    path = tmp_path / "forward_era5.nc"
    dataset.to_netcdf(path)

    configs = load_all_configs()
    result = run_forward_simulation(configs, str(path))

    n_channels = len(configs["instrument"]["instrument"]["channels"]["center_frequencies_ghz"])

    assert "tb" in result
    assert len(result["tb"]) == n_channels


def test_run_forward_simulation_missing_instrument_config(tmp_path: Path) -> None:
    dataset = xr.Dataset(
        data_vars={
            "t": (("time", "level"), [[290.0, 285.0, 275.0]]),
        },
        coords={
            "time": [0],
            "level": [1000.0, 850.0, 700.0],
        },
    )
    path = tmp_path / "forward_missing_instrument.nc"
    dataset.to_netcdf(path)

    configs = copy.deepcopy(load_all_configs())
    del configs["instrument"]["instrument"]

    with pytest.raises(ValueError):
        run_forward_simulation(configs, str(path))


def test_run_forward_simulation_uses_mock_by_default(tmp_path: Path) -> None:
    dataset = xr.Dataset(
        data_vars={
            "t": (("time", "level"), [[290.0, 285.0, 275.0]]),
        },
        coords={
            "time": [0],
            "level": [1000.0, 850.0, 700.0],
        },
    )
    path = tmp_path / "forward_mock_default.nc"
    dataset.to_netcdf(path)

    configs = load_all_configs()
    configs["project"]["environment"]["use_pyarts"] = False

    result = run_forward_simulation(configs, str(path))

    assert "tb" in result


def test_run_forward_simulation_arts_backend_not_implemented(tmp_path: Path) -> None:
    dataset = xr.Dataset(
        data_vars={
            "t": (("time", "level"), [[290.0, 285.0, 275.0]]),
        },
        coords={
            "time": [0],
            "level": [1000.0, 850.0, 700.0],
        },
    )
    path = tmp_path / "forward_arts_backend.nc"
    dataset.to_netcdf(path)

    configs = load_all_configs()
    configs["project"]["environment"]["use_pyarts"] = True

    with pytest.raises(RuntimeError, match="ARTS simulation failed"):
        run_forward_simulation(configs, str(path))
