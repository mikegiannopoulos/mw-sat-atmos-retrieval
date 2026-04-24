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

from mwsat.pipeline.profile_loader import load_profile_from_config
from mwsat.utils.config import load_all_configs


def test_pipeline_load_era5_success(tmp_path: Path) -> None:
    dataset = xr.Dataset(
        data_vars={
            "t": (("time", "level"), [[290.0, 285.0, 275.0]]),
        },
        coords={
            "time": [0],
            "level": [1000.0, 850.0, 700.0],
        },
    )
    path = tmp_path / "pipeline_era5.nc"
    dataset.to_netcdf(path)

    configs = load_all_configs()
    profile = load_profile_from_config(configs, str(path))

    assert "pressure" in profile
    assert "temperature" in profile


def test_pipeline_missing_profile_source() -> None:
    configs = copy.deepcopy(load_all_configs())
    del configs["experiments"]["experiments"][0]["inputs"]["profile_source"]

    with pytest.raises(ValueError):
        load_profile_from_config(configs, "dummy/path.nc")


def test_pipeline_invalid_source() -> None:
    configs = copy.deepcopy(load_all_configs())
    configs["experiments"]["experiments"][0]["inputs"]["profile_source"] = "invalid_source"

    with pytest.raises(ValueError):
        load_profile_from_config(configs, "dummy/path.nc")
