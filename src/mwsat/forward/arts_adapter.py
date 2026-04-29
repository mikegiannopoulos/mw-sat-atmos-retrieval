from __future__ import annotations


def simulate_with_arts(profile: dict, instrument_config: dict) -> dict:
    """Future PyARTS backend for clear-sky forward simulation.

    PyARTS is available in the target environment, but this adapter is
    intentionally deferred until a verified ARTS 2.6 clear-sky workspace
    setup is added. The current mock simulator remains the default backend
    for forward calculations in this project.
    """
    try:
        import pyarts  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "PyARTS is not installed. Install 'pyarts' to enable ARTS-based simulation."
        ) from exc

    raise NotImplementedError(
        "PyARTS is available, but verified ARTS 2.6 clear-sky workspace setup "
        "is still pending. The mock simulator remains the default backend."
    )
