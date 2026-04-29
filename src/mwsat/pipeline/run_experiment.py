from __future__ import annotations

from pathlib import Path

from mwsat.pipeline.run_forward import run_forward_simulation
from mwsat.utils.config import get_active_experiment


def run_experiment(configs: dict) -> dict:
    """Run a single config-driven experiment.

    This helper resolves the active experiment, selects one input profile file,
    and runs the current forward-plus-retrieval pipeline. Multi-profile support
    will be added later.
    """
    experiment = get_active_experiment(configs)

    inputs = experiment.get("inputs")
    if not isinstance(inputs, dict):
        raise ValueError("Active experiment is missing an 'inputs' configuration block")

    profile_source = inputs.get("profile_source")
    if not profile_source:
        raise ValueError("Active experiment is missing 'inputs.profile_source'")

    inputs.get("n_profiles")

    paths_config = configs.get("paths")
    if not isinstance(paths_config, dict):
        raise ValueError("Missing paths configuration")

    raw_paths = paths_config.get("raw")
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

    return run_forward_simulation(configs, str(files[0]))
