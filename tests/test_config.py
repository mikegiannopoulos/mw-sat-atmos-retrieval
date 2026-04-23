from __future__ import annotations

import copy
import shutil
import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from mwsat.utils.config import get_active_experiment, load_all_configs


def test_load_all_configs_success() -> None:
    configs = load_all_configs()

    assert "project" in configs
    assert "paths" in configs
    assert "instrument" in configs
    assert "retrieval" in configs
    assert "experiments" in configs


def test_get_active_experiment_success() -> None:
    configs = load_all_configs()

    experiment = get_active_experiment(configs)

    assert experiment["name"] == "baseline_clear_sky_temperature"


def test_missing_config_file(tmp_path: Path) -> None:
    source_dir = PROJECT_ROOT / "configs"
    temp_config_dir = tmp_path / "configs"
    shutil.copytree(source_dir, temp_config_dir)

    missing_file = temp_config_dir / "instrument.yaml"
    missing_file.rename(temp_config_dir / "instrument.yaml.bak")

    with pytest.raises(FileNotFoundError):
        load_all_configs(str(temp_config_dir))


def test_no_active_experiment() -> None:
    configs = copy.deepcopy(load_all_configs())

    for experiment in configs["experiments"]["experiments"]:
        experiment["active"] = False

    with pytest.raises(ValueError, match="found none"):
        get_active_experiment(configs)


def test_multiple_active_experiments() -> None:
    configs = copy.deepcopy(load_all_configs())
    experiments = configs["experiments"]["experiments"]

    duplicate_experiment = copy.deepcopy(experiments[0])
    duplicate_experiment["name"] = "second_active_experiment"
    duplicate_experiment["active"] = True
    experiments.append(duplicate_experiment)

    with pytest.raises(ValueError, match="found multiple"):
        get_active_experiment(configs)
