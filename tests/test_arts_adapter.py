from __future__ import annotations

import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from mwsat.forward.arts_adapter import simulate_with_arts


def test_simulate_with_arts_not_implemented() -> None:
    profile = {
        "pressure": [1000.0, 850.0, 700.0],
        "temperature": [290.0, 280.0, 270.0],
    }
    instrument_config = {
        "channels": {
            "center_frequencies_ghz": [50.3, 52.8],
        }
    }

    with pytest.raises(NotImplementedError, match="implemented"):
        simulate_with_arts(profile, instrument_config)
