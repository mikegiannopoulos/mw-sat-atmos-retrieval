from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Channel:
    frequency_hz: float
    nedt_k: float


@dataclass(frozen=True)
class InstrumentConfig:
    name: str
    channels: list[Channel]
    sensor_za_deg: float = 0.0
