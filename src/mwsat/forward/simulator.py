from __future__ import annotations

from mwsat.profiles.base import validate_profile_data


def simulate_brightness_temperature(profile: dict, instrument_config: dict) -> dict:
    """Simulate brightness temperatures using a simple placeholder model.

    This function is a minimal stand-in for future ARTS/PyARTS-based forward
    simulations. It validates the required profile and instrument inputs, then
    returns a deterministic mock brightness temperature for each configured
    channel.
    """
    validate_profile_data(profile)

    if not isinstance(instrument_config, dict):
        raise ValueError("Instrument configuration must be a dictionary")

    channels = instrument_config.get("channels")
    if not isinstance(channels, dict):
        raise ValueError("Instrument configuration is missing 'channels'")

    center_frequencies = channels.get("center_frequencies_ghz")
    if not isinstance(center_frequencies, list) or not center_frequencies:
        raise ValueError(
            "Instrument configuration is missing 'channels.center_frequencies_ghz'"
        )

    temperatures = profile.get("temperature")
    if not isinstance(temperatures, (list, tuple)) or not temperatures:
        raise ValueError("Profile temperature data must be a non-empty sequence")

    mean_temperature = sum(temperatures) / len(temperatures)
    tb = [
        mean_temperature + 0.1 * channel_index
        for channel_index in range(len(center_frequencies))
    ]

    return {
        "tb": tb,
        "n_channels": len(center_frequencies),
    }
