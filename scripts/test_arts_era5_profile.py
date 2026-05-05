from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import xarray as xr
from pyarts.workspace import Workspace, arts_agenda


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FREQUENCIES_HZ = np.array([50.3e9, 52.8e9, 54.4e9])
GRAVITY = 9.80665


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


def prepare_arts_profile(
    profile: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
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
    altitude_source = "ERA5 geopotential / 9.80665"

    validate_arts_profile_inputs(pressure, temperature, altitude, humidity)

    return pressure, temperature, altitude, humidity, altitude_source


def validate_arts_profile_inputs(
    pressure: np.ndarray,
    temperature: np.ndarray,
    altitude: np.ndarray,
    humidity: np.ndarray,
) -> None:
    assert np.all(np.isfinite(pressure))
    assert np.all(pressure > 0.0)
    assert np.all(np.diff(pressure) < 0.0)
    assert np.all(np.isfinite(temperature))
    assert np.all(np.isfinite(altitude))
    assert np.all(np.diff(altitude) > 0.0)
    assert np.all(np.isfinite(humidity))
    assert np.all(humidity >= 0.0)


def build_workspace(
    pressure: np.ndarray,
    temperature: np.ndarray,
    altitude: np.ndarray,
) -> Workspace:
    ws = Workspace()

    ws.AtmosphereSet1D()

    ws.p_grid = pressure
    ws.t_field = temperature.reshape((-1, 1, 1))
    ws.z_field = altitude.reshape((-1, 1, 1))
    ws.f_grid = FREQUENCIES_HZ

    ws.stokes_dim = 1
    ws.iy_unit = "PlanckBT"
    ws.sensorOff()
    ws.sensor_pos = np.array([[850000.0]])
    ws.sensor_los = np.array([[180.0]])
    ws.sensor_checkedCalc()

    ws.abs_speciesSet(species=["O2-PWR98"])
    ws.vmr_field = np.full((1, len(ws.p_grid.value), 1, 1), 0.21)
    ws.cloudbox_on = 0
    ws.jacobianOff()
    ws.cloudboxOff()
    ws.surface_skin_t = float(temperature[0])
    ws.iy_main_agendaSet(option="Emission")
    ws.iy_space_agendaSet(option="CosmicBackground")
    ws.iy_surface_agendaSet(option="UseSurfaceRtprop")
    ws.ppath_agendaSet(option="FollowSensorLosPath")
    ws.ppath_step_agendaSet(option="GeometricPath")
    ws.water_p_eq_agendaSet()

    @arts_agenda(ws=ws, set_agenda=True)
    def surface_rtprop_agenda(ws):
        ws.Touch(ws.surface_skin_t)
        ws.surfaceBlackbody()

    @arts_agenda(ws=ws, set_agenda=True)
    def propmat_clearsky_agenda(ws):
        ws.Ignore(ws.rtp_mag)
        ws.Ignore(ws.rtp_los)
        ws.Ignore(ws.rtp_nlte)
        ws.propmat_clearskyInit()
        ws.propmat_clearskyAddPredefined()

    ws.propmat_clearsky_agenda_checkedCalc()
    ws.atmfields_checkedCalc()
    ws.refellipsoidSet(re=6371000.0, e=0.0)
    ws.z_surfaceConstantAltitude(altitude=float(altitude[0]))
    ws.atmgeom_checkedCalc()
    ws.cloudbox_checkedCalc()

    return ws


def run_ycalc(ws: Workspace) -> np.ndarray:
    ws.yCalc()
    y = np.asarray(ws.y.value).copy()
    if y.ndim == 1:
        y = y.reshape((-1, 1))
    return y


def validate_brightness_temperatures(y: np.ndarray, n_channels: int) -> None:
    assert y.shape == (n_channels, 1)
    assert np.all(np.isfinite(y))
    assert np.all((y > 100.0) & (y < 350.0))


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
    altitude_source: str,
) -> None:
    print(f"Pressure range: {pressure.min():.0f}-{pressure.max():.0f} Pa")
    print(f"Temperature range: {temperature.min():.1f}-{temperature.max():.1f} K")
    print(
        "Altitude range: "
        f"{altitude.min():.0f}-{altitude.max():.0f} m ({altitude_source})"
    )
    print(f"Specific humidity range: {humidity.min():.3e}-{humidity.max():.3e} kg/kg")


def print_brightness_temperatures(y: np.ndarray) -> None:
    print("Brightness temperatures:")
    for frequency_hz, brightness_temperature in zip(FREQUENCIES_HZ, y[:, 0]):
        print(f"{frequency_hz / 1e9:.1f} GHz -> {brightness_temperature:.1f} K")


def print_summary() -> None:
    print("Summary:")
    print("- ERA5 profile loading: PASS")
    print("- ARTS input preparation: PASS")
    print("- ERA5 forward simulation: PASS")
    print("- Brightness-temperature sanity: PASS")


def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else default_era5_path()

    metadata = first_profile_metadata(path)
    profile = load_era5_arts_profile(path)
    pressure, temperature, altitude, humidity, altitude_source = prepare_arts_profile(
        profile
    )

    ws = build_workspace(pressure, temperature, altitude)
    y = run_ycalc(ws)
    validate_brightness_temperatures(y, len(FREQUENCIES_HZ))

    print_metadata(metadata)
    print_profile_summary(pressure, temperature, altitude, humidity, altitude_source)
    print_brightness_temperatures(y)
    print_summary()


if __name__ == "__main__":
    main()
