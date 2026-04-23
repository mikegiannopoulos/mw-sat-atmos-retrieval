from __future__ import annotations

from pathlib import Path

import yaml


REQUIRED_CONFIG_FILES = (
    "project.yaml",
    "paths.yaml",
    "instrument.yaml",
    "retrieval.yaml",
    "experiments.yaml",
)


def load_yaml(path: str) -> dict:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Config file not found: {file_path}")
    if not file_path.is_file():
        raise ValueError(f"Config path is not a file: {file_path}")

    with file_path.open("r", encoding="utf-8") as stream:
        data = yaml.safe_load(stream)

    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"Top-level YAML structure must be a mapping: {file_path}")
    return data


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _resolve_path_values(value: object, project_root: Path) -> object:
    if isinstance(value, dict):
        return {
            key: _resolve_path_values(nested_value, project_root)
            for key, nested_value in value.items()
        }
    if isinstance(value, list):
        return [_resolve_path_values(item, project_root) for item in value]
    if isinstance(value, str):
        path = Path(value)
        if path.is_absolute():
            return str(path)
        return str((project_root / path).resolve())
    return value


def _resolve_paths_config(paths_config: dict, project_root: Path) -> dict:
    return _resolve_path_values(paths_config, project_root)


def load_all_configs(config_dir: str = "configs") -> dict:
    project_root = _project_root()
    config_path = Path(config_dir)
    if not config_path.is_absolute():
        config_path = project_root / config_path

    if not config_path.exists():
        raise FileNotFoundError(f"Config directory not found: {config_path}")
    if not config_path.is_dir():
        raise ValueError(f"Config path is not a directory: {config_path}")

    missing_files = [
        filename for filename in REQUIRED_CONFIG_FILES if not (config_path / filename).exists()
    ]
    if missing_files:
        missing = ", ".join(missing_files)
        raise FileNotFoundError(f"Missing required config file(s) in {config_path}: {missing}")

    configs = {
        "project": load_yaml(str(config_path / "project.yaml")),
        "paths": load_yaml(str(config_path / "paths.yaml")),
        "instrument": load_yaml(str(config_path / "instrument.yaml")),
        "retrieval": load_yaml(str(config_path / "retrieval.yaml")),
        "experiments": load_yaml(str(config_path / "experiments.yaml")),
    }

    configs["paths"] = _resolve_paths_config(configs["paths"], project_root)
    get_active_experiment(configs)
    return configs


def get_active_experiment(configs: dict) -> dict:
    experiments_config = configs.get("experiments")
    if not isinstance(experiments_config, dict):
        raise ValueError("Expected 'experiments' config to be a dictionary")

    experiments = experiments_config.get("experiments")
    if not isinstance(experiments, list):
        raise ValueError("Expected 'experiments.experiments' to be a list")

    active_experiments = [
        experiment
        for experiment in experiments
        if isinstance(experiment, dict) and experiment.get("active") is True
    ]

    if not active_experiments:
        raise ValueError("Expected exactly one active experiment, found none")
    if len(active_experiments) > 1:
        names = [
            experiment.get("name", "<unnamed>")
            for experiment in active_experiments
        ]
        raise ValueError(
            "Expected exactly one active experiment, found multiple: "
            + ", ".join(names)
        )

    return active_experiments[0]


if __name__ == "__main__":
    configs = load_all_configs()
    exp = get_active_experiment(configs)
    print(exp["name"])
