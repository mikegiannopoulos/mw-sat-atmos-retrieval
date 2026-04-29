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

from mwsat.pipeline.run_experiment import (
    run_experiment,
    run_experiment_batch,
    run_experiment_summary,
)
from mwsat.utils.config import load_all_configs


def test_run_experiment_success(tmp_path: Path) -> None:
    raw_dir = tmp_path / "era5"
    raw_dir.mkdir()

    dataset = xr.Dataset(
        data_vars={
            "t": (("time", "level"), [[290.0, 285.0, 275.0]]),
        },
        coords={
            "time": [0],
            "level": [1000.0, 850.0, 700.0],
        },
    )
    dataset.to_netcdf(raw_dir / "dummy_era5.nc")

    configs = load_all_configs()
    configs["paths"]["paths"]["raw"]["era5"] = str(raw_dir)

    result = run_experiment(configs)

    assert "tb" in result
    assert "n_channels" in result
    assert "profile_source" in result
    assert "metrics" in result
    assert "retrieval" in result


def test_run_experiment_no_files(tmp_path: Path) -> None:
    raw_dir = tmp_path / "empty_era5"
    raw_dir.mkdir()

    configs = load_all_configs()
    configs["paths"]["paths"]["raw"]["era5"] = str(raw_dir)

    with pytest.raises(ValueError):
        run_experiment(configs)


def test_run_experiment_missing_raw_path() -> None:
    configs = copy.deepcopy(load_all_configs())
    del configs["paths"]["paths"]["raw"]["era5"]

    with pytest.raises(ValueError):
        run_experiment(configs)


def test_run_experiment_batch_success(tmp_path: Path) -> None:
    raw_dir = tmp_path / "batch_era5"
    raw_dir.mkdir()

    dataset_one = xr.Dataset(
        data_vars={
            "t": (("time", "level"), [[290.0, 285.0, 275.0]]),
        },
        coords={
            "time": [0],
            "level": [1000.0, 850.0, 700.0],
        },
    )
    dataset_two = xr.Dataset(
        data_vars={
            "t": (("time", "level"), [[289.0, 284.0, 274.0]]),
        },
        coords={
            "time": [0],
            "level": [1000.0, 850.0, 700.0],
        },
    )
    dataset_one.to_netcdf(raw_dir / "dummy_era5_1.nc")
    dataset_two.to_netcdf(raw_dir / "dummy_era5_2.nc")

    configs = load_all_configs()
    configs["paths"]["paths"]["raw"]["era5"] = str(raw_dir)
    configs["experiments"]["experiments"][0]["inputs"]["n_profiles"] = 2

    result = run_experiment_batch(configs)

    assert isinstance(result, list)
    assert len(result) == 2
    assert "tb" in result[0]
    assert "retrieval" in result[0]
    assert "tb" in result[1]
    assert "retrieval" in result[1]


def test_run_experiment_batch_invalid_n_profiles() -> None:
    configs = copy.deepcopy(load_all_configs())
    configs["experiments"]["experiments"][0]["inputs"]["n_profiles"] = 0

    with pytest.raises(ValueError):
        run_experiment_batch(configs)


def test_run_experiment_summary_success(tmp_path: Path) -> None:
    raw_dir = tmp_path / "summary_era5"
    raw_dir.mkdir()

    dataset_one = xr.Dataset(
        data_vars={
            "t": (("time", "level"), [[290.0, 285.0, 275.0]]),
        },
        coords={
            "time": [0],
            "level": [1000.0, 850.0, 700.0],
        },
    )
    dataset_two = xr.Dataset(
        data_vars={
            "t": (("time", "level"), [[289.0, 284.0, 274.0]]),
        },
        coords={
            "time": [0],
            "level": [1000.0, 850.0, 700.0],
        },
    )
    dataset_one.to_netcdf(raw_dir / "dummy_era5_1.nc")
    dataset_two.to_netcdf(raw_dir / "dummy_era5_2.nc")

    configs = load_all_configs()
    configs["paths"]["paths"]["raw"]["era5"] = str(raw_dir)
    configs["experiments"]["experiments"][0]["inputs"]["n_profiles"] = 2

    output = run_experiment_summary(configs)

    assert "results" in output
    assert "summary" in output
    assert len(output["results"]) == 2
    assert output["summary"]["n_profiles"] == 2
    assert "mean_bias" in output["summary"]
    assert "mean_rmse" in output["summary"]
