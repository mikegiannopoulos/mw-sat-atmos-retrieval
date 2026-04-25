from __future__ import annotations


def simulate_with_arts(profile: dict, instrument_config: dict) -> dict:
    """Scaffold for future PyARTS-based radiative transfer simulation.

    This module will later host the ARTS/PyARTS forward-model implementation.
    For now, the project continues to use the mock simulation in
    ``simulator.py`` for placeholder brightness temperature calculations.
    """
    try:
        import pyarts  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "PyARTS is not installed. Install 'pyarts' to enable ARTS-based simulation."
        ) from exc

    raise NotImplementedError(
        "ARTS integration is scaffolded but not implemented yet. "
        "Use the mock simulator in 'simulator.py' for now."
    )
