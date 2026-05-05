from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from mwsat.profiles.era5 import load_era5_profile  # noqa: E402


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: python scripts/check_era5_profile.py <path-to-era5-netcdf>")
        return 1

    path = sys.argv[1]

    try:
        profile = load_era5_profile(path)
        pressure = profile["pressure"]
        temperature = profile["temperature"]

        print(f"Loaded file: {path}")
        print(f"Number of levels: {len(pressure)}")
        print(f"First 5 pressure values: {pressure[:5]}")
        print(f"First 5 temperature values: {temperature[:5]}")

        try:
            import matplotlib.pyplot as plt
        except ImportError:
            return 0

        plt.figure()
        plt.plot(temperature, pressure, marker="o")
        plt.xlabel("Temperature")
        plt.ylabel("Pressure")
        plt.gca().invert_yaxis()
        plt.title("ERA5 Temperature Profile")
        plt.tight_layout()
        plt.show()
        return 0
    except Exception as exc:
        print(f"Error: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
