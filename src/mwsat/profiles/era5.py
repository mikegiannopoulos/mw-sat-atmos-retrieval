from __future__ import annotations

from pathlib import Path

import xarray as xr

from .base import validate_profile_data


def _select_first_profile(data_array: xr.DataArray) -> xr.DataArray:
    indexers = {
        dim: 0
        for dim in data_array.dims
        if dim not in {"level", "pressure", "pressure_level"}
    }
    if indexers:
        data_array = data_array.isel(**indexers)
    return data_array


def load_era5_profile(path: str) -> dict:
    """Load a single vertical temperature profile from an ERA5 NetCDF file.

    This function expects an ERA5-style NetCDF dataset containing a vertical
    coordinate stored as either ``pressure``, ``level``, or
    ``pressure_level`` and a temperature variable stored as either ``t`` or
    ``temperature``.

    The implementation is intentionally minimal. It extracts only one profile,
    selects the first element along any non-vertical dimensions, performs no
    interpolation, and does not yet support explicit time handling or
    multi-profile workflows.
    """
    file_path = Path(path)

    with xr.open_dataset(file_path) as dataset:
        pressure_name = None
        for candidate in ("pressure", "level", "pressure_level"):
            if candidate in dataset.variables or candidate in dataset.coords:
                pressure_name = candidate
                break
        if pressure_name is None:
            raise ValueError(
                "ERA5 file must contain a vertical coordinate named "
                "'pressure', 'level', or 'pressure_level'"
            )

        temperature_name = None
        for candidate in ("temperature", "t"):
            if candidate in dataset.data_vars:
                temperature_name = candidate
                break
        if temperature_name is None:
            raise ValueError(
                "ERA5 file must contain a temperature variable named "
                "'temperature' or 't'"
            )

        pressure = dataset[pressure_name]
        temperature = dataset[temperature_name]

        pressure = _select_first_profile(pressure).squeeze().values
        temperature = _select_first_profile(temperature).squeeze().values

        if pressure.ndim != 1:
            raise ValueError(
                "ERA5 pressure coordinate must be one-dimensional after squeezing"
            )
        if temperature.ndim != 1:
            raise ValueError(
                "ERA5 temperature variable must be one-dimensional after squeezing"
            )
        if len(pressure) != len(temperature):
            raise ValueError(
                "ERA5 pressure and temperature profiles must have the same length"
            )

        profile = {
            "pressure": pressure.tolist(),
            "temperature": temperature.tolist(),
        }

    return validate_profile_data(profile)
