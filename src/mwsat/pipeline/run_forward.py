from __future__ import annotations

from mwsat.evaluation.metrics import compute_bias, compute_rmse
from mwsat.forward.arts_adapter import simulate_with_arts
from mwsat.forward.simulator import simulate_brightness_temperature
from mwsat.pipeline.profile_loader import load_profile_from_config
from mwsat.retrieval.baseline import retrieve_temperature_profile
from mwsat.utils.config import get_active_experiment


def run_forward_simulation(configs: dict, path: str) -> dict:
    """Run a single forward simulation experiment from config and profile input."""
    experiment = get_active_experiment(configs)
    profile = load_profile_from_config(configs, path)

    instrument_section = configs.get("instrument")
    if not isinstance(instrument_section, dict):
        raise ValueError("Missing instrument configuration")

    instrument_config = instrument_section.get("instrument")
    if not isinstance(instrument_config, dict):
        raise ValueError("Missing instrument configuration")

    inputs = experiment.get("inputs")
    if not isinstance(inputs, dict):
        raise ValueError("Active experiment is missing an 'inputs' configuration block")

    profile_source = inputs.get("profile_source")
    if not profile_source:
        raise ValueError("Active experiment is missing 'inputs.profile_source'")

    environment_config = configs.get("environment")
    if not isinstance(environment_config, dict):
        project_config = configs.get("project")
        if isinstance(project_config, dict):
            environment_config = project_config.get("environment")

    use_pyarts = False
    if isinstance(environment_config, dict):
        use_pyarts = bool(environment_config.get("use_pyarts", False))

    if use_pyarts:
        try:
            result = simulate_with_arts(profile, instrument_config)
        except Exception as exc:
            raise RuntimeError("ARTS simulation failed") from exc
    else:
        result = simulate_brightness_temperature(profile, instrument_config)

    retrieval_config = configs.get("retrieval")
    if not isinstance(retrieval_config, dict):
        raise ValueError("Missing retrieval configuration")

    retrieval_result = retrieve_temperature_profile(result, retrieval_config)
    result["profile_source"] = profile_source
    reference = profile["temperature"]
    estimate = result["tb"]
    n_values = min(len(reference), len(estimate))
    reference = reference[:n_values]
    estimate = estimate[:n_values]
    result["metrics"] = {
        "bias": compute_bias(reference, estimate),
        "rmse": compute_rmse(reference, estimate),
    }
    result["retrieval"] = retrieval_result
    return result
