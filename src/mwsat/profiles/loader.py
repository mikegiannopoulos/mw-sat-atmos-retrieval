from __future__ import annotations

from .era5 import load_era5_profile
from .igra import load_igra_profile


def load_profile(source: str, path: str) -> dict:
    """Load a single atmospheric profile from a supported source."""
    if source == "era5":
        return load_era5_profile(path)
    if source == "igra":
        return load_igra_profile(path)
    raise ValueError(f"Unsupported profile source: {source}")
