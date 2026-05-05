from __future__ import annotations

import numpy as np
from pyarts.workspace import Workspace, arts_agenda


def build_workspace() -> tuple[Workspace, np.ndarray]:
    ws = Workspace()

    ws.AtmosphereSet1D()

    ws.p_grid = np.array([100000.0, 85000.0, 70000.0, 50000.0, 30000.0])
    t_field = np.array([290.0, 285.0, 275.0, 260.0, 240.0]).reshape((-1, 1, 1))
    ws.t_field = t_field
    ws.z_field = np.array([0.0, 1500.0, 3000.0, 5500.0, 9000.0]).reshape((-1, 1, 1))
    ws.f_grid = np.array([50.3e9, 52.8e9, 54.4e9])

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
    ws.surface_skin_t = 290.0
    ws.iy_main_agendaSet(option="Emission")
    ws.iy_space_agendaSet(option="CosmicBackground")
    ws.iy_surface_agendaSet(option="UseSurfaceRtprop")
    ws.ppath_agendaSet(option="FollowSensorLosPath")
    ws.ppath_step_agendaSet(option="GeometricPath")
    ws.water_p_eq_agendaSet()

    @arts_agenda(ws=ws, set_agenda=True)
    def surface_rtprop_agenda(ws):
        # Keep surface temperature controlled by surface_skin_t for perturbation tests.
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
    ws.z_surfaceConstantAltitude(altitude=0.0)
    ws.atmgeom_checkedCalc()
    ws.cloudbox_checkedCalc()

    return ws, t_field


def run_ycalc(ws: Workspace) -> np.ndarray:
    ws.yCalc()
    y = ws.y.value
    y = np.asarray(y).copy()
    if y.ndim == 1:
        y = y.reshape((-1, 1))
    return y


def validate_brightness_temperatures(
    y: np.ndarray,
    n_channels: int,
    expected_shape: tuple[int, ...] | None = None,
) -> None:
    if expected_shape is None:
        expected_shape = (n_channels, 1)

    assert y.shape == expected_shape
    assert np.all(np.isfinite(y))
    assert np.all((y > 100.0) & (y < 350.0))


def print_frequency_diagnostics(
    frequencies_ghz: np.ndarray,
    brightness_temperatures: np.ndarray,
) -> None:
    for frequency, brightness_temperature in zip(
        frequencies_ghz, brightness_temperatures
    ):
        print(f"{frequency:.1f} GHz -> {brightness_temperature:.1f} K")

    channel_differences = np.diff(brightness_temperatures)
    print(f"Channel differences (K): {channel_differences}")

    if np.allclose(channel_differences, 0.0):
        print("Warning: brightness temperatures are flat across channels.")
    elif np.all(channel_differences > 0.0):
        print("Warning: brightness temperatures increase across all channels.")


def print_comparison_table(
    frequencies_ghz: np.ndarray,
    baseline_tb: np.ndarray,
    comparison_tb: np.ndarray,
    label: str,
) -> None:
    deltas = comparison_tb - baseline_tb

    print(f"Frequency GHz | Baseline TB K | {label} | Delta K")
    for frequency, baseline_value, comparison_value, delta in zip(
        frequencies_ghz,
        baseline_tb,
        comparison_tb,
        deltas,
    ):
        print(
            f"{frequency:13.1f} | {baseline_value:13.1f} | "
            f"{comparison_value:11.1f} | {delta:7.2f}"
        )


def run_temperature_perturbation(
    ws: Workspace,
    t_field: np.ndarray,
    y_baseline: np.ndarray,
) -> np.ndarray:
    ws.t_field = t_field
    ws.atmfields_checkedCalc()
    y_perturbed = run_ycalc(ws)
    validate_brightness_temperatures(
        y_perturbed, len(ws.f_grid.value), expected_shape=y_baseline.shape
    )
    return y_perturbed


def main() -> None:
    ws, t_field = build_workspace()

    y = run_ycalc(ws)
    validate_brightness_temperatures(y, len(ws.f_grid.value))

    frequencies_ghz = ws.f_grid.value / 1e9
    brightness_temperatures = y[:, 0]
    print_frequency_diagnostics(frequencies_ghz, brightness_temperatures)

    ws.t_field = t_field + 5.0
    ws.atmfields_checkedCalc()
    y_warmed = run_ycalc(ws)
    validate_brightness_temperatures(
        y_warmed, len(ws.f_grid.value), expected_shape=y.shape
    )

    warmed_brightness_temperatures = y_warmed[:, 0]
    deltas = warmed_brightness_temperatures - brightness_temperatures
    assert np.any(np.abs(deltas) > 0.1)

    print_comparison_table(
        frequencies_ghz,
        brightness_temperatures,
        warmed_brightness_temperatures,
        "Warmed TB K",
    )

    ws.t_field = t_field
    ws.surface_skin_t = 295.0
    ws.atmfields_checkedCalc()
    y_surface_warmed = run_ycalc(ws)
    validate_brightness_temperatures(
        y_surface_warmed, len(ws.f_grid.value), expected_shape=y.shape
    )

    surface_warmed_brightness_temperatures = y_surface_warmed[:, 0]
    surface_deltas = (
        surface_warmed_brightness_temperatures - brightness_temperatures
    )
    assert np.any(np.abs(surface_deltas) > 0.01)

    # Stronger O2 absorption channels should be less sensitive to the surface
    # because their weighting shifts upward away from the lower boundary.
    print_comparison_table(
        frequencies_ghz,
        brightness_temperatures,
        surface_warmed_brightness_temperatures,
        "Surface+5K TB K",
    )

    lower_layer_t_field = t_field.copy()
    lower_layer_t_field[0, 0, 0] += 5.0
    y_lower_layer_warmed = run_temperature_perturbation(
        ws, lower_layer_t_field, y
    )
    lower_layer_brightness_temperatures = y_lower_layer_warmed[:, 0]

    # Lower/surface-sensitive channels should respond more to perturbations
    # near the lower boundary.
    print_comparison_table(
        frequencies_ghz,
        brightness_temperatures,
        lower_layer_brightness_temperatures,
        "Lower-layer +5K TB K",
    )

    upper_layer_t_field = t_field.copy()
    upper_layer_t_field[-1, 0, 0] += 5.0
    y_upper_layer_warmed = run_temperature_perturbation(
        ws, upper_layer_t_field, y
    )
    upper_layer_brightness_temperatures = y_upper_layer_warmed[:, 0]

    # Stronger absorption channels should become relatively more sensitive to
    # higher atmospheric layers as their weighting shifts upward.
    print_comparison_table(
        frequencies_ghz,
        brightness_temperatures,
        upper_layer_brightness_temperatures,
        "Upper-layer +5K TB K",
    )

    print("Summary:")
    print("- Baseline run: PASS")
    print("- Unit sanity: PASS")
    print("- Uniform atmospheric perturbation: PASS")
    print("- Surface perturbation: PASS")
    print("- Lower-layer perturbation: PASS")
    print("- Upper-layer perturbation: PASS")


if __name__ == "__main__":
    main()
