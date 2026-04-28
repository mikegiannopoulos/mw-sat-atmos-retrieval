from __future__ import annotations

from pathlib import Path

import xarray as xr


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    output_dir = project_root / "data" / "raw" / "era5"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "dummy_era5.nc"

    dataset = xr.Dataset(
        data_vars={
            "t": (("time", "level"), [[290.0, 285.0, 275.0, 260.0, 240.0]]),
        },
        coords={
            "time": [0],
            "level": [1000, 850, 700, 500, 300],
        },
    )

    dataset.to_netcdf(output_path)
    print(f"Created file: {output_path}")


if __name__ == "__main__":
    main()
