from __future__ import annotations

from pathlib import Path

from mwsat.pipeline.run_forward import run_forward_simulation
from mwsat.utils.config import get_active_experiment


def _get_experiment_input_files(configs: dict) -> list[Path]:
    experiment = get_active_experiment(configs)

    inputs = experiment.get("inputs")
    if not isinstance(inputs, dict):
        raise ValueError("Active experiment is missing an 'inputs' configuration block")

    profile_source = inputs.get("profile_source")
    if not profile_source:
        raise ValueError("Active experiment is missing 'inputs.profile_source'")

    n_profiles = inputs.get("n_profiles")
    if n_profiles is None:
        raise ValueError("Active experiment is missing 'inputs.n_profiles'")
    if n_profiles < 1:
        raise ValueError("Active experiment 'inputs.n_profiles' must be at least 1")

    paths_config = configs.get("paths")
    if not isinstance(paths_config, dict):
        raise ValueError("Missing paths configuration")

    paths_section = paths_config.get("paths")
    if not isinstance(paths_section, dict):
        raise ValueError("Missing nested paths configuration")

    raw_paths = paths_section.get("raw")
    if not isinstance(raw_paths, dict):
        raise ValueError("Missing raw data paths configuration")

    data_dir = raw_paths.get(profile_source)
    if not data_dir:
        raise ValueError(f"Missing raw data path for profile source: {profile_source}")

    data_path = Path(data_dir)
    if not data_path.is_dir():
        raise ValueError(f"Profile data directory does not exist: {data_path}")

    files = sorted(path for path in data_path.iterdir() if path.is_file())
    if not files:
        raise ValueError(f"No input files found in profile data directory: {data_path}")

    return files[:n_profiles]


def run_experiment_batch(configs: dict) -> list[dict]:
    files = _get_experiment_input_files(configs)
    return [run_forward_simulation(configs, str(path)) for path in files]


def run_experiment(configs: dict) -> dict:
    """Run a single config-driven experiment as a convenience wrapper."""
    return run_experiment_batch(configs)[0]
