from __future__ import annotations

from mwsat.profiles import load_profile
from mwsat.utils.config import get_active_experiment


def load_profile_from_config(configs: dict, path: str) -> dict:
    """Load a profile by connecting the active experiment config to ingestion."""
    experiment = get_active_experiment(configs)
    inputs = experiment.get("inputs")
    if not isinstance(inputs, dict):
        raise ValueError("Active experiment is missing an 'inputs' configuration block")

    profile_source = inputs.get("profile_source")
    if not profile_source:
        raise ValueError("Active experiment is missing 'inputs.profile_source'")

    return load_profile(profile_source, path)
