from __future__ import annotations

import numpy as np
from pyarts.workspace import Workspace, arts_agenda

from mwsat.instrument.config import InstrumentConfig


def _build_workspace(
    pressure: np.ndarray,
    temperature: np.ndarray,
    altitude: np.ndarray,
    instrument: InstrumentConfig,
    vmr_h2o: np.ndarray | None = None,
    surface_skin_t: float | None = None,
) -> Workspace:
    ws = Workspace()

    ws.AtmosphereSet1D()

    ws.p_grid = pressure
    ws.t_field = temperature.reshape((-1, 1, 1))
    ws.z_field = altitude.reshape((-1, 1, 1))
    ws.f_grid = np.array([channel.frequency_hz for channel in instrument.channels])

    ws.stokes_dim = 1
    ws.iy_unit = "PlanckBT"
    ws.sensorOff()
    ws.sensor_pos = np.array([[850000.0]])
    ws.sensor_los = np.array([[180.0 - instrument.sensor_za_deg]])
    ws.sensor_checkedCalc()

    if vmr_h2o is None:
        ws.abs_speciesSet(species=["O2-PWR98"])
        ws.vmr_field = np.full((1, len(ws.p_grid.value), 1, 1), 0.21)
    else:
        ws.abs_speciesSet(species=["O2-PWR98", "H2O-PWR98"])
        ws.vmr_field = np.stack(
            [
                np.full(len(ws.p_grid.value), 0.21),
                np.asarray(vmr_h2o, dtype=float),
            ],
            axis=0,
        ).reshape((2, len(ws.p_grid.value), 1, 1))
    ws.cloudbox_on = 0
    ws.jacobianOff()
    ws.cloudboxOff()
    ws.surface_skin_t = (
        float(temperature[0]) if surface_skin_t is None else float(surface_skin_t)
    )
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


def simulate_brightness_temperatures(
    pressure: np.ndarray,
    temperature: np.ndarray,
    altitude: np.ndarray,
    instrument: InstrumentConfig,
    vmr_h2o: np.ndarray | None = None,
    surface_skin_t: float | None = None,
) -> np.ndarray:
    ws = _build_workspace(
        pressure,
        temperature,
        altitude,
        instrument,
        vmr_h2o=vmr_h2o,
        surface_skin_t=surface_skin_t,
    )
    ws.yCalc()
    y = np.asarray(ws.y.value).copy()
    if y.ndim == 1:
        y = y.reshape((-1, 1))
    return y
