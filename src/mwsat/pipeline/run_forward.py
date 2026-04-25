from __future__ import annotations

from mwsat.forward.simulator import simulate_brightness_temperature
from mwsat.pipeline.profile_loader import load_profile_from_config
from mwsat.utils.config import get_active_experiment


def run_forward_simulation(configs: dict, path: str) -> dict:
    """Run a single forward simulation experiment from config and profile input."""
    get_active_experiment(configs)
    profile = load_profile_from_config(configs, path)

    instrument_section = configs.get("instrument")
    if not isinstance(instrument_section, dict):
        raise ValueError("Missing instrument configuration")

    instrument_config = instrument_section.get("instrument")
    if not isinstance(instrument_config, dict):
        raise ValueError("Missing instrument configuration")

    return simulate_brightness_temperature(profile, instrument_config)
