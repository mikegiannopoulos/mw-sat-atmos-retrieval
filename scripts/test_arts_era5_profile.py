from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from mwsat.forward.simulate import simulate_brightness_temperatures  # noqa: E402
from mwsat.experiments.sensitivity import (  # noqa: E402
    layer_temperature_sensitivity as run_layer_temperature_sensitivity,
)
from mwsat.experiments.sensitivity import (  # noqa: E402
    temperature_perturbation_sensitivity as run_temperature_perturbation_sensitivity,
)
from mwsat.instrument.config import Channel, InstrumentConfig  # noqa: E402
from mwsat.instrument.noise import apply_noise  # noqa: E402


GRAVITY = 9.80665
M_DRY_AIR_KG_PER_MOL = 28.9647e-3
M_H2O_KG_PER_MOL = 18.01528e-3
INSTRUMENT_NOISE_SEED = 4
DEGRADED_INSTRUMENT_NOISE_SEED = 4
MULTI_PROFILE_NOISE_BASE_SEED = 100
TEMPERATURE_PERTURBATION_K = 1.0
LOWER_ATMOSPHERE_PRESSURE_THRESHOLD_PA = 70000.0
UPPER_ATMOSPHERE_PRESSURE_THRESHOLD_PA = 30000.0
N_MULTI_PROFILES = 6
OVERWRITE_EXPERIMENT_OUTPUTS = True
EXPERIMENT_OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "experiments"


baseline_instrument = InstrumentConfig(
    name="minimal-mw-sounder",
    channels=[
        Channel(frequency_hz=50.3e9, nedt_k=0.3),
        Channel(frequency_hz=52.8e9, nedt_k=0.5),
        Channel(frequency_hz=54.4e9, nedt_k=0.7),
    ],
)


degraded_instrument = InstrumentConfig(
    name="minimal-mw-sounder-degraded",
    channels=[
        Channel(
            frequency_hz=channel.frequency_hz,
            nedt_k=2.0 * channel.nedt_k,
        )
        for channel in baseline_instrument.channels
    ],
    sensor_za_deg=baseline_instrument.sensor_za_deg,
)


def default_era5_path() -> Path:
    return PROJECT_ROOT / "data" / "raw" / "era5" / "arts_era5_sample.nc"


def first_profile_metadata(path: Path) -> dict[str, object]:
    with xr.open_dataset(path) as dataset:
        metadata: dict[str, object] = {
            "path": str(path),
            "dims": dict(dataset.sizes),
            "variables": list(dataset.data_vars),
        }

        for name in ("valid_time", "time", "latitude", "longitude"):
            if name in dataset.coords:
                values = np.asarray(dataset[name].values).reshape(-1)
                if len(values):
                    metadata[name] = values[0]

    return metadata


def select_first_profile(data_array: xr.DataArray) -> xr.DataArray:
    vertical_dims = {"level", "pressure", "pressure_level"}
    indexers = {dim: 0 for dim in data_array.dims if dim not in vertical_dims}
    if indexers:
        data_array = data_array.isel(**indexers)
    return data_array.squeeze()


def variable_name(dataset: xr.Dataset, candidates: tuple[str, ...], label: str) -> str:
    for candidate in candidates:
        if candidate in dataset.variables or candidate in dataset.coords:
            return candidate
    raise ValueError(
        f"ERA5 file must contain {label} named one of: {', '.join(candidates)}"
    )


def load_era5_arts_profile(path: Path) -> dict[str, list[float]]:
    with xr.open_dataset(path) as dataset:
        pressure_name = variable_name(
            dataset, ("pressure", "level", "pressure_level"), "a pressure coordinate"
        )
        temperature_name = variable_name(
            dataset, ("temperature", "t"), "a temperature variable"
        )
        humidity_name = variable_name(
            dataset, ("specific_humidity", "q"), "a specific humidity variable"
        )
        geopotential_name = variable_name(
            dataset, ("geopotential", "z"), "a geopotential variable"
        )

        pressure = select_first_profile(dataset[pressure_name])
        temperature = select_first_profile(dataset[temperature_name])
        humidity = select_first_profile(dataset[humidity_name])
        geopotential = select_first_profile(dataset[geopotential_name])

        profile = {
            "pressure": np.asarray(pressure.values, dtype=float).tolist(),
            "temperature": np.asarray(temperature.values, dtype=float).tolist(),
            "specific_humidity": np.asarray(humidity.values, dtype=float).tolist(),
            "geopotential": np.asarray(geopotential.values, dtype=float).tolist(),
        }

    return profile


def select_profile_indices(n_available: int, n_select: int) -> np.ndarray:
    if n_select >= n_available:
        return np.arange(n_available, dtype=int)
    return np.unique(np.round(np.linspace(0, n_available - 1, n_select)).astype(int))


def load_era5_arts_profiles(path: Path, limit: int) -> list[dict[str, object]]:
    profiles: list[dict[str, object]] = []
    with xr.open_dataset(path) as dataset:
        pressure_name = variable_name(
            dataset, ("pressure", "level", "pressure_level"), "a pressure coordinate"
        )
        temperature_name = variable_name(
            dataset, ("temperature", "t"), "a temperature variable"
        )
        humidity_name = variable_name(
            dataset, ("specific_humidity", "q"), "a specific humidity variable"
        )
        geopotential_name = variable_name(
            dataset, ("geopotential", "z"), "a geopotential variable"
        )

        time_name = next(
            (name for name in ("valid_time", "time") if name in dataset.coords),
            None,
        )
        time_size = int(dataset.sizes.get(time_name, 1)) if time_name else 1
        profile_indices = select_profile_indices(time_size, limit)

        for profile_index in profile_indices:
            indexers = {}
            if time_name:
                indexers[time_name] = int(profile_index)
            for dim in dataset[temperature_name].dims:
                if dim not in {"level", "pressure", "pressure_level", time_name}:
                    indexers[dim] = 0

            pressure = dataset[pressure_name]
            if time_name and time_name in pressure.dims:
                pressure = pressure.isel({time_name: int(profile_index)})
            for dim in pressure.dims:
                if dim not in {"level", "pressure", "pressure_level"}:
                    pressure = pressure.isel({dim: 0})

            temperature = dataset[temperature_name].isel(**indexers).squeeze()
            humidity = dataset[humidity_name].isel(**indexers).squeeze()
            geopotential = dataset[geopotential_name].isel(**indexers).squeeze()

            valid_time = None
            if time_name:
                valid_time_values = np.asarray(dataset[time_name].values).reshape(-1)
                if int(profile_index) < len(valid_time_values):
                    valid_time = valid_time_values[int(profile_index)]

            profiles.append(
                {
                    "profile_index": int(profile_index),
                    "valid_time": None if valid_time is None else str(valid_time),
                    "pressure": np.asarray(pressure.values, dtype=float).tolist(),
                    "temperature": np.asarray(temperature.values, dtype=float).tolist(),
                    "specific_humidity": np.asarray(humidity.values, dtype=float).tolist(),
                    "geopotential": np.asarray(geopotential.values, dtype=float).tolist(),
                }
            )

    return profiles


def specific_humidity_to_h2o_vmr(q: np.ndarray) -> np.ndarray:
    mass_mixing_ratio = q / (1.0 - q)
    return mass_mixing_ratio * (M_DRY_AIR_KG_PER_MOL / M_H2O_KG_PER_MOL)


def prepare_arts_profile(
    profile: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
    pressure = np.asarray(profile["pressure"], dtype=float)
    temperature = np.asarray(profile["temperature"], dtype=float)
    humidity = np.asarray(profile["specific_humidity"], dtype=float)
    geopotential = np.asarray(profile["geopotential"], dtype=float)

    if np.nanmax(pressure) < 2000.0:
        pressure = pressure * 100.0

    order = np.argsort(pressure)[::-1]
    pressure = pressure[order]
    temperature = temperature[order]
    humidity = humidity[order]
    geopotential = geopotential[order]
    altitude = geopotential / GRAVITY
    vmr_h2o = specific_humidity_to_h2o_vmr(humidity)
    altitude_source = "ERA5 geopotential / 9.80665"

    validate_arts_profile_inputs(pressure, temperature, altitude, humidity, vmr_h2o)

    return pressure, temperature, altitude, humidity, vmr_h2o, altitude_source


def validate_arts_profile_inputs(
    pressure: np.ndarray,
    temperature: np.ndarray,
    altitude: np.ndarray,
    humidity: np.ndarray,
    vmr_h2o: np.ndarray,
) -> None:
    assert np.all(np.isfinite(pressure))
    assert np.all(pressure > 0.0)
    assert np.all(np.diff(pressure) < 0.0)
    assert np.all(np.isfinite(temperature))
    assert np.all(np.isfinite(altitude))
    assert np.all(np.diff(altitude) > 0.0)
    assert np.all(np.isfinite(humidity))
    assert np.all(humidity >= 0.0)
    assert np.all(np.isfinite(vmr_h2o))
    assert np.all(vmr_h2o >= 0.0)
    assert np.all(vmr_h2o < 0.1)


def validate_brightness_temperatures(y: np.ndarray, n_channels: int) -> None:
    assert y.shape == (n_channels, 1)
    assert np.all(np.isfinite(y))
    assert np.all((y > 100.0) & (y < 350.0))


def validate_instrument_noise(
    noise: np.ndarray, instrument: InstrumentConfig
) -> tuple[float, float, float]:
    noise_mean = float(noise.mean())
    noise_std = float(noise.std())
    expected_average_nedt = float(
        np.mean([channel.nedt_k for channel in instrument.channels])
    )
    assert np.all(np.isfinite(noise))
    assert abs(noise_mean) < expected_average_nedt
    assert abs(noise_std - expected_average_nedt) < 0.3
    return noise_mean, noise_std, expected_average_nedt


def channel_signal_to_noise_proxy(
    y_true: np.ndarray, instrument: InstrumentConfig
) -> list[tuple[float, float, float, float]]:
    return [
        (
            channel.frequency_hz,
            float(tb_true),
            channel.nedt_k,
            float(tb_true / channel.nedt_k),
        )
        for channel, tb_true in zip(instrument.channels, y_true[:, 0])
    ]


def profile_metadata_row(
    temperature: np.ndarray, vmr_h2o: np.ndarray
) -> dict[str, float]:
    return {
        "surface_temperature_k": float(temperature[0]),
        "min_temperature_k": float(temperature.min()),
        "max_temperature_k": float(temperature.max()),
        "mean_temperature_k": float(temperature.mean()),
        "min_h2o_vmr": float(vmr_h2o.min()),
        "max_h2o_vmr": float(vmr_h2o.max()),
    }


def simulate_multi_profile_observations(
    path: Path, instrument: InstrumentConfig, n_profiles: int, base_seed: int
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for profile in load_era5_arts_profiles(path, n_profiles):
        pressure, temperature, altitude, _, vmr_h2o, _ = prepare_arts_profile(profile)
        metadata = profile_metadata_row(temperature, vmr_h2o)
        tb_true = simulate_brightness_temperatures(
            pressure, temperature, altitude, instrument, vmr_h2o=vmr_h2o
        )
        profile_seed = base_seed + int(profile["profile_index"])
        tb_obs, noise = apply_noise(tb_true, instrument, profile_seed)
        for channel_index, channel in enumerate(instrument.channels):
            rows.append(
                {
                    "profile_index": int(profile["profile_index"]),
                    "valid_time": profile["valid_time"],
                    "frequency_ghz": channel.frequency_hz / 1e9,
                    "tb_true_k": float(tb_true[channel_index, 0]),
                    "tb_obs_k": float(tb_obs[channel_index, 0]),
                    "noise_k": float(noise[channel_index, 0]),
                    "nedt_k": channel.nedt_k,
                    **metadata,
                }
            )
    return pd.DataFrame(rows)


def simulate_multi_profile_temperature_sensitivity(
    path: Path,
    instrument: InstrumentConfig,
    n_profiles: int,
    delta_temperature_k: float,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for profile in load_era5_arts_profiles(path, n_profiles):
        pressure, temperature, altitude, _, vmr_h2o, _ = prepare_arts_profile(profile)
        metadata = profile_metadata_row(temperature, vmr_h2o)
        results = run_temperature_perturbation_sensitivity(
            pressure,
            temperature,
            altitude,
            instrument,
            vmr_h2o=vmr_h2o,
            delta_temperature_k=delta_temperature_k,
        )
        for channel_index, channel in enumerate(instrument.channels):
            rows.append(
                {
                    "profile_index": int(profile["profile_index"]),
                    "valid_time": profile["valid_time"],
                    "frequency_ghz": channel.frequency_hz / 1e9,
                    "tb_base_k": float(results["y_base"][channel_index, 0]),
                    "tb_warm_k": float(results["y_warm"][channel_index, 0]),
                    "dTb_dT": float(results["dTb_dT"][channel_index, 0]),
                    "nedt_k": channel.nedt_k,
                    "sensitivity_to_noise_ratio": float(
                        results["ratio"][channel_index, 0]
                    ),
                    **metadata,
                }
            )
    return pd.DataFrame(rows)


def simulate_multi_profile_layer_temperature_sensitivity(
    path: Path,
    instrument: InstrumentConfig,
    n_profiles: int,
    delta_temperature_k: float,
    lower_pressure_threshold_pa: float,
    upper_pressure_threshold_pa: float,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for profile in load_era5_arts_profiles(path, n_profiles):
        pressure, temperature, altitude, _, vmr_h2o, _ = prepare_arts_profile(profile)
        metadata = profile_metadata_row(temperature, vmr_h2o)
        results = run_layer_temperature_sensitivity(
            pressure,
            temperature,
            altitude,
            instrument,
            vmr_h2o=vmr_h2o,
            delta_temperature_k=delta_temperature_k,
            lower_pressure_threshold_pa=lower_pressure_threshold_pa,
            upper_pressure_threshold_pa=upper_pressure_threshold_pa,
        )
        for channel_index, channel in enumerate(instrument.channels):
            rows.append(
                {
                    "profile_index": int(profile["profile_index"]),
                    "valid_time": profile["valid_time"],
                    "frequency_ghz": channel.frequency_hz / 1e9,
                    "tb_base_k": float(results["y_base"][channel_index, 0]),
                    "dTb_lower_k": float(results["dTb_lower"][channel_index, 0]),
                    "dTb_upper_k": float(results["dTb_upper"][channel_index, 0]),
                    "nedt_k": channel.nedt_k,
                    "lower_ratio": float(results["lower_ratio"][channel_index, 0]),
                    "upper_ratio": float(results["upper_ratio"][channel_index, 0]),
                    **metadata,
                }
            )
    return pd.DataFrame(rows)


def multi_profile_noise_summary(results: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        results.groupby("frequency_ghz", as_index=False)
        .agg(
            count=("noise_k", "count"),
            nedt_k=("nedt_k", "first"),
            mean_noise_k=("noise_k", "mean"),
            std_noise_k=("noise_k", lambda values: float(values.std(ddof=0))),
            rms_noise_k=("noise_k", lambda values: float(np.sqrt(np.mean(values**2)))),
        )
        .sort_values("frequency_ghz")
    )
    return grouped


def multi_profile_temperature_sensitivity_summary(
    results: pd.DataFrame,
) -> pd.DataFrame:
    grouped = (
        results.groupby("frequency_ghz", as_index=False)
        .agg(
            mean_dtb_dt=("dTb_dT", "mean"),
            std_dtb_dt=("dTb_dT", lambda values: float(values.std(ddof=0))),
            mean_sensitivity_to_noise_ratio=("sensitivity_to_noise_ratio", "mean"),
            min_sensitivity_to_noise_ratio=("sensitivity_to_noise_ratio", "min"),
            max_sensitivity_to_noise_ratio=("sensitivity_to_noise_ratio", "max"),
        )
        .sort_values("frequency_ghz")
    )
    return grouped


def multi_profile_layer_temperature_sensitivity_summary(
    results: pd.DataFrame,
) -> pd.DataFrame:
    grouped = (
        results.groupby("frequency_ghz", as_index=False)
        .agg(
            mean_lower_ratio=("lower_ratio", "mean"),
            mean_upper_ratio=("upper_ratio", "mean"),
            mean_dtb_lower_k=("dTb_lower_k", "mean"),
            mean_dtb_upper_k=("dTb_upper_k", "mean"),
        )
        .sort_values("frequency_ghz")
    )
    return grouped


def multi_profile_sensitivity_stability_summary(
    temperature_results: pd.DataFrame, layer_results: pd.DataFrame
) -> pd.DataFrame:
    temperature_grouped = (
        temperature_results.groupby("frequency_ghz", as_index=False)
        .agg(
            mean_dtb_dt=("dTb_dT", "mean"),
            std_dtb_dt=("dTb_dT", lambda values: float(values.std(ddof=0))),
        )
        .sort_values("frequency_ghz")
    )
    temperature_grouped["cv_dtb_dt"] = (
        temperature_grouped["std_dtb_dt"]
        / temperature_grouped["mean_dtb_dt"].abs()
    )

    layer_grouped = (
        layer_results.groupby("frequency_ghz", as_index=False)
        .agg(
            mean_lower_ratio=("lower_ratio", "mean"),
            std_lower_ratio=("lower_ratio", lambda values: float(values.std(ddof=0))),
            mean_upper_ratio=("upper_ratio", "mean"),
            std_upper_ratio=("upper_ratio", lambda values: float(values.std(ddof=0))),
        )
        .sort_values("frequency_ghz")
    )

    return temperature_grouped.merge(layer_grouped, on="frequency_ghz", how="inner")


def multi_profile_diversity_summary(results: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        results.groupby(["profile_index", "valid_time"], as_index=False)
        .agg(
            surface_temperature_k=("surface_temperature_k", "first"),
            min_temperature_k=("min_temperature_k", "first"),
            max_temperature_k=("max_temperature_k", "first"),
            max_h2o_vmr=("max_h2o_vmr", "first"),
            tb_min_k=("tb_true_k", "min"),
            tb_max_k=("tb_true_k", "max"),
        )
        .sort_values("profile_index")
    )
    return grouped


def save_experiment_dataframe(
    dataframe: pd.DataFrame, output_path: Path, overwrite: bool
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Refusing to overwrite existing file without overwrite flag: {output_path}"
        )
    dataframe.to_csv(output_path, index=False)
    return output_path


def validate_multi_profile_reproducibility(
    results: pd.DataFrame, repeated_results: pd.DataFrame
) -> None:
    pd.testing.assert_frame_equal(results, repeated_results, check_like=False)

    noise_vectors = {
        int(profile_index): tuple(group["noise_k"].to_numpy())
        for profile_index, group in results.groupby("profile_index", sort=True)
    }
    assert len(set(noise_vectors.values())) == len(noise_vectors)


def print_metadata(metadata: dict[str, object]) -> None:
    print(f"ERA5 file: {metadata['path']}")
    print(f"Dataset dimensions: {metadata['dims']}")
    print(f"Data variables: {metadata['variables']}")

    for name in ("valid_time", "time", "latitude", "longitude"):
        if name in metadata:
            print(f"Selected {name}: {metadata[name]}")


def print_profile_summary(
    pressure: np.ndarray,
    temperature: np.ndarray,
    altitude: np.ndarray,
    humidity: np.ndarray,
    vmr_h2o: np.ndarray,
    altitude_source: str,
) -> None:
    print(f"Pressure range: {pressure.min():.0f}-{pressure.max():.0f} Pa")
    print(f"Temperature range: {temperature.min():.1f}-{temperature.max():.1f} K")
    print(
        "Altitude range: "
        f"{altitude.min():.0f}-{altitude.max():.0f} m ({altitude_source})"
    )
    print(f"Specific humidity range: {humidity.min():.3e}-{humidity.max():.3e} kg/kg")
    print(f"True H2O VMR range: {vmr_h2o.min():.3e}-{vmr_h2o.max():.3e}")


def print_brightness_temperatures(y: np.ndarray, instrument: InstrumentConfig) -> None:
    print("Brightness temperatures:")
    for channel, brightness_temperature in zip(instrument.channels, y[:, 0]):
        print(f"{channel.frequency_hz / 1e9:.1f} GHz -> {brightness_temperature:.1f} K")


def print_brightness_temperature_comparison(
    y_o2_only: np.ndarray, y_o2_h2o: np.ndarray, instrument: InstrumentConfig
) -> None:
    print("Brightness temperatures comparison:")
    for channel, bt_o2_only, bt_o2_h2o in zip(
        instrument.channels, y_o2_only[:, 0], y_o2_h2o[:, 0]
    ):
        print(
            f"{channel.frequency_hz / 1e9:.1f} GHz -> "
            f"O2-only {bt_o2_only:.1f} K, O2+H2O {bt_o2_h2o:.1f} K"
        )


def print_instrument_comparison(
    y_true: np.ndarray,
    y_obs_baseline: np.ndarray,
    y_obs_degraded: np.ndarray,
    delta_tb_obs: np.ndarray,
    baseline: InstrumentConfig,
    degraded: InstrumentConfig,
) -> None:
    print("Instrument comparison:")
    print("Frequency | Tb_true | Tb_baseline | Tb_degraded | ΔTb | NEΔT_base | NEΔT_deg")
    for channel_base, channel_deg, tb_true, tb_base, tb_deg, delta_tb in zip(
        baseline.channels,
        degraded.channels,
        y_true[:, 0],
        y_obs_baseline[:, 0],
        y_obs_degraded[:, 0],
        delta_tb_obs[:, 0],
    ):
        print(
            f"{channel_base.frequency_hz / 1e9:.1f} GHz | "
            f"{tb_true:.1f} K | {tb_base:.1f} K | {tb_deg:.1f} K | "
            f"{delta_tb:+.2f} K | {channel_base.nedt_k:.2f} K | "
            f"{channel_deg.nedt_k:.2f} K"
        )


def print_instrument_comparison_summary(
    baseline_noise_mean: float,
    baseline_noise_std: float,
    baseline_expected_average_nedt: float,
    degraded_noise_mean: float,
    degraded_noise_std: float,
    degraded_expected_average_nedt: float,
    delta_tb_obs: np.ndarray,
    baseline: InstrumentConfig,
) -> None:
    mean_absolute_delta_tb = float(np.mean(np.abs(delta_tb_obs)))
    mean_ratio = float(
        np.mean(
            np.abs(delta_tb_obs[:, 0])
            / np.array([channel.nedt_k for channel in baseline.channels])
        )
    )
    print(
        f"Baseline noise summary: mean={baseline_noise_mean:+.3f} K, "
        f"std={baseline_noise_std:.3f} K, "
        f"expected average NEΔT={baseline_expected_average_nedt:.3f} K"
    )
    print(
        f"Degraded noise summary: mean={degraded_noise_mean:+.3f} K, "
        f"std={degraded_noise_std:.3f} K, "
        f"expected average NEΔT={degraded_expected_average_nedt:.3f} K"
    )
    print(
        f"Instrument comparison summary: mean |ΔTb|={mean_absolute_delta_tb:.3f} K, "
        f"mean |ΔTb| / NEΔT_base={mean_ratio:.3f}"
    )


def print_channel_information_summary(
    signal_to_noise_proxy: list[tuple[float, float, float, float]]
) -> None:
    print("Observation-level channel diagnostic:")
    print("Frequency | Tb_true | NEΔT | Tb_true / NEΔT")
    for frequency_hz, tb_true, nedt_k, proxy in signal_to_noise_proxy:
        print(
            f"{frequency_hz / 1e9:.1f} GHz | "
            f"{tb_true:.1f} K | {nedt_k:.2f} K | {proxy:.1f}"
        )


def print_channel_information_analysis(
    signal_to_noise_proxy: list[tuple[float, float, float, float]]
) -> None:
    largest_proxy = max(signal_to_noise_proxy, key=lambda channel: channel[3])
    print(
        "Channel diagnostic analysis: "
        f"the largest Tb_true / NEΔT occurs at {largest_proxy[0] / 1e9:.1f} GHz."
    )
    print(
        "This is only a first-order observation-level diagnostic. "
        "It does not measure vertical sensitivity, Jacobian structure, or retrieval value."
    )


def print_temperature_sensitivity_summary(results: dict[str, np.ndarray]) -> None:
    print("Temperature perturbation diagnostic:")
    print("Frequency | Tb_base | Tb_warm | dTb/dT | NEΔT | |dTb/dT| / NEΔT")
    for channel, tb_base, tb_warm, dtb_dt, nedt_k, ratio in zip(
        baseline_instrument.channels,
        results["y_base"][:, 0],
        results["y_warm"][:, 0],
        results["dTb_dT"][:, 0],
        results["nedt"][:, 0],
        results["ratio"][:, 0],
    ):
        print(
            f"{channel.frequency_hz / 1e9:.1f} GHz | "
            f"{tb_base:.1f} K | {tb_warm:.1f} K | {dtb_dt:+.3f} K/K | "
            f"{nedt_k:.2f} K | {ratio:.3f}"
        )


def print_temperature_sensitivity_analysis(results: dict[str, np.ndarray]) -> None:
    strongest_channel_index = int(np.argmax(results["ratio"][:, 0]))
    strongest_channel = baseline_instrument.channels[strongest_channel_index]
    print(
        "Temperature perturbation analysis: "
        f"the largest |dTb/dT| / NEΔT occurs at {strongest_channel.frequency_hz / 1e9:.1f} GHz."
    )
    print(
        "This is a finite-difference response to a uniform +1 K atmospheric warming only. "
        "It is a useful first-order sensitivity check, but it is not a Jacobian-based "
        "retrieval information metric."
    )


def print_layer_temperature_sensitivity_summary(results: dict[str, np.ndarray]) -> None:
    print("Layer temperature perturbation diagnostic:")
    print(
        "Frequency | Tb_base | dTb_lower | dTb_upper | NEΔT | "
        "|dTb_lower|/NEΔT | |dTb_upper|/NEΔT"
    )
    for channel, tb_base, dTb_lower, dTb_upper, nedt_k, lower_ratio, upper_ratio in zip(
        baseline_instrument.channels,
        results["y_base"][:, 0],
        results["dTb_lower"][:, 0],
        results["dTb_upper"][:, 0],
        results["nedt"][:, 0],
        results["lower_ratio"][:, 0],
        results["upper_ratio"][:, 0],
    ):
        print(
            f"{channel.frequency_hz / 1e9:.1f} GHz | {tb_base:.1f} K | "
            f"{dTb_lower:+.3f} K | {dTb_upper:+.3f} K | {nedt_k:.2f} K | "
            f"{lower_ratio:.3f} | {upper_ratio:.3f}"
        )


def print_layer_temperature_sensitivity_analysis(results: dict[str, np.ndarray]) -> None:
    strongest_lower = baseline_instrument.channels[int(np.argmax(results["lower_ratio"][:, 0]))]
    strongest_upper = baseline_instrument.channels[int(np.argmax(results["upper_ratio"][:, 0]))]
    print(
        "Layer temperature analysis: "
        f"the strongest lower-atmosphere response relative to noise occurs at "
        f"{strongest_lower.frequency_hz / 1e9:.1f} GHz, while the strongest upper-atmosphere "
        f"response occurs at {strongest_upper.frequency_hz / 1e9:.1f} GHz."
    )
    print(
        "This remains a profile-specific finite-difference diagnostic from one ERA5 case. "
        "It suggests relative sensitivity to lower versus upper temperature structure, "
        "but it should not be overgeneralized beyond this scene."
    )


def print_summary() -> None:
    print("Summary:")
    print("- ERA5 profile loading: PASS")
    print("- ARTS input preparation: PASS")
    print("- ERA5 forward simulation: PASS")
    print("- Brightness-temperature sanity: PASS")


def print_multi_profile_summary(results: pd.DataFrame) -> None:
    print("Multi-profile simulation summary:")
    selected_valid_times = [
        value for value in results["valid_time"].drop_duplicates().tolist() if value is not None
    ]
    if selected_valid_times:
        print(f"Selected valid_time values: {selected_valid_times}")
    print(f"Profiles simulated: {results['profile_index'].nunique()}")
    print(f"Rows in DataFrame: {len(results)}")
    print(
        f"Tb range: {results['tb_true_k'].min():.1f}-{results['tb_true_k'].max():.1f} K"
    )
    print(
        "Temperature-profile range across selected profiles: "
        f"{results['min_temperature_k'].min():.1f}-{results['max_temperature_k'].max():.1f} K"
    )
    print(
        f"Noise mean/std: {results['noise_k'].mean():+.3f}/{results['noise_k'].std(ddof=0):.3f} K"
    )


def print_multi_profile_diversity_summary(summary: pd.DataFrame) -> None:
    print("Profile diversity summary:")
    print("Profile | valid_time | surface T | min T | max T | max H2O VMR | Tb min | Tb max")
    for row in summary.itertuples(index=False):
        print(
            f"{row.profile_index} | {row.valid_time} | "
            f"{row.surface_temperature_k:.1f} K | {row.min_temperature_k:.1f} K | "
            f"{row.max_temperature_k:.1f} K | {row.max_h2o_vmr:.3e} | "
            f"{row.tb_min_k:.1f} K | {row.tb_max_k:.1f} K"
        )


def print_multi_profile_noise_summary(
    results: pd.DataFrame, noise_summary: pd.DataFrame
) -> None:
    overall_rms_noise = float(np.sqrt(np.mean(results["noise_k"] ** 2)))
    expected_rms_nedt = float(np.sqrt(np.mean(results["nedt_k"] ** 2)))

    print("Multi-profile noise diagnostic:")
    print("Frequency | Count | NEΔT | Mean noise | Std noise | RMS noise")
    for row in noise_summary.itertuples(index=False):
        print(
            f"{row.frequency_ghz:.1f} GHz | {row.count} | {row.nedt_k:.2f} K | "
            f"{row.mean_noise_k:+.3f} K | {row.std_noise_k:.3f} K | "
            f"{row.rms_noise_k:.3f} K"
        )
    print(f"Overall RMS noise: {overall_rms_noise:.3f} K")
    print(f"Expected RMS NEΔT: {expected_rms_nedt:.3f} K")
    print(
        f"Multi-profile reproducibility: PASS with base seed {MULTI_PROFILE_NOISE_BASE_SEED}"
    )
    print("Profile-specific noise vectors: PASS")
    print(
        "Noise diagnostic interpretation: with only a few samples per frequency, "
        "the grouped mean, standard deviation, and RMS estimates are expected to be noisy. "
        "This sample is useful for a sanity check, but it is too small to expect stable "
        "per-channel noise statistics."
    )


def print_multi_profile_temperature_sensitivity_summary(
    results: pd.DataFrame, summary: pd.DataFrame
) -> None:
    print("Multi-profile temperature sensitivity summary:")
    print(
        "Frequency | Mean dTb/dT | Std dTb/dT | "
        "Mean |dTb/dT|/NEΔT | Min |dTb/dT|/NEΔT | Max |dTb/dT|/NEΔT"
    )
    for row in summary.itertuples(index=False):
        print(
            f"{row.frequency_ghz:.1f} GHz | {row.mean_dtb_dt:+.3f} K/K | "
            f"{row.std_dtb_dt:.3f} K/K | "
            f"{row.mean_sensitivity_to_noise_ratio:.3f} | "
            f"{row.min_sensitivity_to_noise_ratio:.3f} | "
            f"{row.max_sensitivity_to_noise_ratio:.3f}"
        )
    print(
        f"Profiles in sensitivity batch: {results['profile_index'].nunique()}, "
        f"rows: {len(results)}"
    )


def print_multi_profile_layer_temperature_sensitivity_summary(
    results: pd.DataFrame, summary: pd.DataFrame
) -> None:
    print("Multi-profile layer temperature sensitivity summary:")
    print(
        "Frequency | Mean lower_ratio | Mean upper_ratio | "
        "Mean dTb_lower | Mean dTb_upper"
    )
    for row in summary.itertuples(index=False):
        print(
            f"{row.frequency_ghz:.1f} GHz | {row.mean_lower_ratio:.3f} | "
            f"{row.mean_upper_ratio:.3f} | "
            f"{row.mean_dtb_lower_k:+.3f} K | {row.mean_dtb_upper_k:+.3f} K"
        )
    print(
        "Layer sensitivity interpretation: across these selected profiles, the lower/upper "
        "response pattern is consistent if the same channels keep the larger mean "
        "lower_ratio or upper_ratio. With only a few profiles, that pattern is still a "
        "small-sample indication rather than a stable climatological result."
    )
    print(
        f"Profiles in layer-sensitivity batch: {results['profile_index'].nunique()}, "
        f"rows: {len(results)}"
    )


def print_multi_profile_sensitivity_stability_summary(
    summary: pd.DataFrame, n_profiles: int
) -> None:
    print("Multi-profile sensitivity stability summary:")
    print(
        "Frequency | Mean dTb/dT | Std dTb/dT | CV dTb/dT | "
        "Mean lower_ratio | Std lower_ratio | Mean upper_ratio | Std upper_ratio"
    )
    for row in summary.itertuples(index=False):
        print(
            f"{row.frequency_ghz:.1f} GHz | {row.mean_dtb_dt:+.3f} K/K | "
            f"{row.std_dtb_dt:.3f} K/K | {row.cv_dtb_dt:.3f} | "
            f"{row.mean_lower_ratio:.3f} | {row.std_lower_ratio:.3f} | "
            f"{row.mean_upper_ratio:.3f} | {row.std_upper_ratio:.3f}"
        )

    strongest_uniform = summary.loc[summary["mean_dtb_dt"].abs().idxmax()]
    strongest_lower = summary.loc[summary["mean_lower_ratio"].idxmax()]
    strongest_upper = summary.loc[summary["mean_upper_ratio"].idxmax()]
    print(
        "Sensitivity stability interpretation: "
        f"{strongest_uniform['frequency_ghz']:.1f} GHz remains strongest for uniform "
        "temperature sensitivity in this sample."
    )
    print(
        "Lower/upper pattern interpretation: "
        f"{strongest_lower['frequency_ghz']:.1f} GHz remains strongest for lower-level "
        f"sensitivity, while {strongest_upper['frequency_ghz']:.1f} GHz remains "
        "strongest for upper-level sensitivity."
    )
    print(
        f"Variability interpretation: the selected sample contains {n_profiles} profiles, "
        "so the grouped variability is still too limited for general conclusions even if "
        "the ranking pattern looks stable."
    )


def print_saved_output_paths(paths: list[Path]) -> None:
    print("Saved experiment outputs:")
    for path in paths:
        print(path)


def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else default_era5_path()

    metadata = first_profile_metadata(path)
    profile = load_era5_arts_profile(path)
    pressure, temperature, altitude, humidity, vmr_h2o, altitude_source = (
        prepare_arts_profile(profile)
    )

    y_o2_only = simulate_brightness_temperatures(
        pressure, temperature, altitude, baseline_instrument
    )
    validate_brightness_temperatures(y_o2_only, len(baseline_instrument.channels))

    y_o2_h2o = simulate_brightness_temperatures(
        pressure, temperature, altitude, baseline_instrument, vmr_h2o=vmr_h2o
    )
    validate_brightness_temperatures(y_o2_h2o, len(baseline_instrument.channels))
    y_o2_h2o_warm = simulate_brightness_temperatures(
        pressure,
        temperature + TEMPERATURE_PERTURBATION_K,
        altitude,
        baseline_instrument,
        vmr_h2o=vmr_h2o,
        surface_skin_t=float(temperature[0]),
    )
    validate_brightness_temperatures(y_o2_h2o_warm, len(baseline_instrument.channels))
    y_obs_baseline, noise_baseline = apply_noise(
        y_o2_h2o, baseline_instrument, INSTRUMENT_NOISE_SEED
    )
    (
        baseline_noise_mean,
        baseline_noise_std,
        baseline_expected_average_nedt,
    ) = validate_instrument_noise(noise_baseline, baseline_instrument)
    y_obs_degraded, noise_degraded = apply_noise(
        y_o2_h2o, degraded_instrument, DEGRADED_INSTRUMENT_NOISE_SEED
    )
    (
        degraded_noise_mean,
        degraded_noise_std,
        degraded_expected_average_nedt,
    ) = validate_instrument_noise(noise_degraded, degraded_instrument)
    delta_tb_obs = y_obs_baseline - y_obs_degraded
    signal_to_noise_proxy = channel_signal_to_noise_proxy(
        y_o2_h2o, baseline_instrument
    )
    temperature_sensitivity_results = run_temperature_perturbation_sensitivity(
        pressure,
        temperature,
        altitude,
        baseline_instrument,
        vmr_h2o=vmr_h2o,
        delta_temperature_k=TEMPERATURE_PERTURBATION_K,
    )
    assert np.all(np.isfinite(temperature_sensitivity_results["y_warm"]))
    assert np.any(
        np.abs(
            temperature_sensitivity_results["y_warm"]
            - temperature_sensitivity_results["y_base"]
        )
        > 0.0
    )
    assert np.all(np.isfinite(temperature_sensitivity_results["dTb_dT"]))
    assert np.all(np.isfinite(temperature_sensitivity_results["ratio"]))
    layer_temperature_sensitivity_results = run_layer_temperature_sensitivity(
        pressure,
        temperature,
        altitude,
        baseline_instrument,
        vmr_h2o=vmr_h2o,
        delta_temperature_k=TEMPERATURE_PERTURBATION_K,
        lower_pressure_threshold_pa=LOWER_ATMOSPHERE_PRESSURE_THRESHOLD_PA,
        upper_pressure_threshold_pa=UPPER_ATMOSPHERE_PRESSURE_THRESHOLD_PA,
    )
    assert np.all(np.isfinite(layer_temperature_sensitivity_results["y_lower_warm"]))
    assert np.all(np.isfinite(layer_temperature_sensitivity_results["y_upper_warm"]))
    assert np.any(
        np.abs(
            layer_temperature_sensitivity_results["y_lower_warm"]
            - layer_temperature_sensitivity_results["y_base"]
        )
        > 0.0
    )
    assert np.any(
        np.abs(
            layer_temperature_sensitivity_results["y_upper_warm"]
            - layer_temperature_sensitivity_results["y_base"]
        )
        > 0.0
    )
    assert np.all(np.isfinite(layer_temperature_sensitivity_results["dTb_lower"]))
    assert np.all(np.isfinite(layer_temperature_sensitivity_results["dTb_upper"]))
    assert np.all(np.isfinite(layer_temperature_sensitivity_results["lower_ratio"]))
    assert np.all(np.isfinite(layer_temperature_sensitivity_results["upper_ratio"]))
    multi_profile_results = simulate_multi_profile_observations(
        path, baseline_instrument, N_MULTI_PROFILES, MULTI_PROFILE_NOISE_BASE_SEED
    )
    repeated_multi_profile_results = simulate_multi_profile_observations(
        path, baseline_instrument, N_MULTI_PROFILES, MULTI_PROFILE_NOISE_BASE_SEED
    )
    multi_profile_noise_results = multi_profile_noise_summary(multi_profile_results)
    multi_profile_diversity_results = multi_profile_diversity_summary(
        multi_profile_results
    )
    multi_profile_temperature_sensitivity_results = (
        simulate_multi_profile_temperature_sensitivity(
            path,
            baseline_instrument,
            N_MULTI_PROFILES,
            TEMPERATURE_PERTURBATION_K,
        )
    )
    multi_profile_temperature_sensitivity_grouped = (
        multi_profile_temperature_sensitivity_summary(
            multi_profile_temperature_sensitivity_results
        )
    )
    multi_profile_layer_temperature_sensitivity_results = (
        simulate_multi_profile_layer_temperature_sensitivity(
            path,
            baseline_instrument,
            N_MULTI_PROFILES,
            TEMPERATURE_PERTURBATION_K,
            LOWER_ATMOSPHERE_PRESSURE_THRESHOLD_PA,
            UPPER_ATMOSPHERE_PRESSURE_THRESHOLD_PA,
        )
    )
    multi_profile_layer_temperature_sensitivity_grouped = (
        multi_profile_layer_temperature_sensitivity_summary(
            multi_profile_layer_temperature_sensitivity_results
        )
    )
    multi_profile_sensitivity_stability_results = (
        multi_profile_sensitivity_stability_summary(
            multi_profile_temperature_sensitivity_results,
            multi_profile_layer_temperature_sensitivity_results,
        )
    )
    selected_profile_count = multi_profile_results["profile_index"].nunique()
    saved_output_paths = [
        save_experiment_dataframe(
            multi_profile_results,
            EXPERIMENT_OUTPUT_DIR / "multi_profile_observations.csv",
            OVERWRITE_EXPERIMENT_OUTPUTS,
        ),
        save_experiment_dataframe(
            multi_profile_temperature_sensitivity_results,
            EXPERIMENT_OUTPUT_DIR / "multi_profile_temperature_sensitivity.csv",
            OVERWRITE_EXPERIMENT_OUTPUTS,
        ),
        save_experiment_dataframe(
            multi_profile_layer_temperature_sensitivity_results,
            EXPERIMENT_OUTPUT_DIR / "multi_profile_layer_sensitivity.csv",
            OVERWRITE_EXPERIMENT_OUTPUTS,
        ),
    ]
    assert selected_profile_count >= 1
    assert len(multi_profile_results) == selected_profile_count * len(
        baseline_instrument.channels
    )
    assert np.all(np.isfinite(multi_profile_results["tb_true_k"]))
    assert np.all(np.isfinite(multi_profile_results["tb_obs_k"]))
    assert np.all(np.isfinite(multi_profile_results["noise_k"]))
    assert len(multi_profile_noise_results) == len(baseline_instrument.channels)
    assert len(multi_profile_diversity_results) == selected_profile_count
    assert (
        multi_profile_temperature_sensitivity_results["profile_index"].nunique()
        == selected_profile_count
    )
    assert len(multi_profile_temperature_sensitivity_results) == (
        selected_profile_count * len(baseline_instrument.channels)
    )
    assert np.all(np.isfinite(multi_profile_temperature_sensitivity_results["tb_base_k"]))
    assert np.all(np.isfinite(multi_profile_temperature_sensitivity_results["tb_warm_k"]))
    assert np.all(np.isfinite(multi_profile_temperature_sensitivity_results["dTb_dT"]))
    assert np.all(
        np.isfinite(
            multi_profile_temperature_sensitivity_results[
                "sensitivity_to_noise_ratio"
            ]
        )
    )
    assert len(multi_profile_temperature_sensitivity_grouped) == len(
        baseline_instrument.channels
    )
    assert (
        multi_profile_layer_temperature_sensitivity_results[
            "profile_index"
        ].nunique()
        == selected_profile_count
    )
    assert len(multi_profile_layer_temperature_sensitivity_results) == (
        selected_profile_count * len(baseline_instrument.channels)
    )
    assert np.all(
        np.isfinite(multi_profile_layer_temperature_sensitivity_results["tb_base_k"])
    )
    assert np.all(
        np.isfinite(multi_profile_layer_temperature_sensitivity_results["dTb_lower_k"])
    )
    assert np.all(
        np.isfinite(multi_profile_layer_temperature_sensitivity_results["dTb_upper_k"])
    )
    assert np.all(
        np.isfinite(multi_profile_layer_temperature_sensitivity_results["lower_ratio"])
    )
    assert np.all(
        np.isfinite(multi_profile_layer_temperature_sensitivity_results["upper_ratio"])
    )
    assert len(multi_profile_layer_temperature_sensitivity_grouped) == len(
        baseline_instrument.channels
    )
    assert len(multi_profile_sensitivity_stability_results) == len(
        baseline_instrument.channels
    )
    validate_multi_profile_reproducibility(
        multi_profile_results, repeated_multi_profile_results
    )

    print_metadata(metadata)
    print_profile_summary(
        pressure, temperature, altitude, humidity, vmr_h2o, altitude_source
    )
    print_brightness_temperatures(y_o2_h2o, baseline_instrument)
    print_brightness_temperature_comparison(
        y_o2_only, y_o2_h2o, baseline_instrument
    )
    print_instrument_comparison(
        y_o2_h2o,
        y_obs_baseline,
        y_obs_degraded,
        delta_tb_obs,
        baseline_instrument,
        degraded_instrument,
    )
    print_instrument_comparison_summary(
        baseline_noise_mean,
        baseline_noise_std,
        baseline_expected_average_nedt,
        degraded_noise_mean,
        degraded_noise_std,
        degraded_expected_average_nedt,
        delta_tb_obs,
        baseline_instrument,
    )
    print_channel_information_summary(signal_to_noise_proxy)
    print_channel_information_analysis(signal_to_noise_proxy)
    print_temperature_sensitivity_summary(temperature_sensitivity_results)
    print_temperature_sensitivity_analysis(temperature_sensitivity_results)
    print_layer_temperature_sensitivity_summary(layer_temperature_sensitivity_results)
    print_layer_temperature_sensitivity_analysis(layer_temperature_sensitivity_results)
    print_multi_profile_summary(multi_profile_results)
    print_multi_profile_diversity_summary(multi_profile_diversity_results)
    print_multi_profile_noise_summary(
        multi_profile_results, multi_profile_noise_results
    )
    print_multi_profile_temperature_sensitivity_summary(
        multi_profile_temperature_sensitivity_results,
        multi_profile_temperature_sensitivity_grouped,
    )
    print_multi_profile_layer_temperature_sensitivity_summary(
        multi_profile_layer_temperature_sensitivity_results,
        multi_profile_layer_temperature_sensitivity_grouped,
    )
    print_multi_profile_sensitivity_stability_summary(
        multi_profile_sensitivity_stability_results, selected_profile_count
    )
    print_saved_output_paths(saved_output_paths)
    print_summary()


if __name__ == "__main__":
    main()
