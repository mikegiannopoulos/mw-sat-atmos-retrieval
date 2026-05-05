from __future__ import annotations

import numpy as np

from mwsat.forward.simulate import simulate_brightness_temperatures
from mwsat.instrument.config import InstrumentConfig


def temperature_perturbation_sensitivity(
    pressure: np.ndarray,
    temperature: np.ndarray,
    altitude: np.ndarray,
    instrument: InstrumentConfig,
    vmr_h2o: np.ndarray | None = None,
    delta_temperature_k: float = 1.0,
) -> dict[str, np.ndarray]:
    y_base = simulate_brightness_temperatures(
        pressure,
        temperature,
        altitude,
        instrument,
        vmr_h2o=vmr_h2o,
    )
    y_warm = simulate_brightness_temperatures(
        pressure,
        temperature + delta_temperature_k,
        altitude,
        instrument,
        vmr_h2o=vmr_h2o,
        surface_skin_t=float(temperature[0]),
    )
    dTb_dT = (y_warm - y_base) / delta_temperature_k
    nedt = np.array([channel.nedt_k for channel in instrument.channels]).reshape(
        y_base.shape
    )
    ratio = np.abs(dTb_dT) / nedt
    return {
        "y_base": y_base,
        "y_warm": y_warm,
        "dTb_dT": dTb_dT,
        "nedt": nedt,
        "ratio": ratio,
    }


def layer_temperature_sensitivity(
    pressure: np.ndarray,
    temperature: np.ndarray,
    altitude: np.ndarray,
    instrument: InstrumentConfig,
    vmr_h2o: np.ndarray | None = None,
    delta_temperature_k: float = 1.0,
    lower_pressure_threshold_pa: float = 70000.0,
    upper_pressure_threshold_pa: float = 30000.0,
) -> dict[str, np.ndarray]:
    lower_atmosphere_mask = pressure >= lower_pressure_threshold_pa
    upper_atmosphere_mask = pressure <= upper_pressure_threshold_pa

    temperature_lower_warm = temperature.copy()
    temperature_lower_warm[lower_atmosphere_mask] += delta_temperature_k

    temperature_upper_warm = temperature.copy()
    temperature_upper_warm[upper_atmosphere_mask] += delta_temperature_k

    y_base = simulate_brightness_temperatures(
        pressure,
        temperature,
        altitude,
        instrument,
        vmr_h2o=vmr_h2o,
    )
    y_lower_warm = simulate_brightness_temperatures(
        pressure,
        temperature_lower_warm,
        altitude,
        instrument,
        vmr_h2o=vmr_h2o,
        surface_skin_t=float(temperature[0]),
    )
    y_upper_warm = simulate_brightness_temperatures(
        pressure,
        temperature_upper_warm,
        altitude,
        instrument,
        vmr_h2o=vmr_h2o,
        surface_skin_t=float(temperature[0]),
    )

    dTb_lower = y_lower_warm - y_base
    dTb_upper = y_upper_warm - y_base
    nedt = np.array([channel.nedt_k for channel in instrument.channels]).reshape(
        y_base.shape
    )
    lower_ratio = np.abs(dTb_lower) / nedt
    upper_ratio = np.abs(dTb_upper) / nedt

    return {
        "y_base": y_base,
        "y_lower_warm": y_lower_warm,
        "y_upper_warm": y_upper_warm,
        "dTb_lower": dTb_lower,
        "dTb_upper": dTb_upper,
        "nedt": nedt,
        "lower_ratio": lower_ratio,
        "upper_ratio": upper_ratio,
        "lower_atmosphere_mask": lower_atmosphere_mask,
        "upper_atmosphere_mask": upper_atmosphere_mask,
    }
