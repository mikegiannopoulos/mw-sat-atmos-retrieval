from __future__ import annotations

from pathlib import Path

import cdsapi


def main() -> int:
    output_dir = Path("data/raw/era5")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "real_era5.nc"

    request = {
        "product_type": "reanalysis",
        "variable": "temperature",
        "year": "2020",
        "month": "01",
        "day": "01",
        "time": "00:00",
        "pressure_level": ["300", "500", "700", "850", "1000"],
        "data_format": "netcdf",
        "download_format": "unarchived",
        "area": [58, 10, 57, 11],
    }

    try:
        client = cdsapi.Client()
        client.retrieve(
            "reanalysis-era5-pressure-levels",
            request,
            str(output_path),
        )
        print(output_path)
        return 0
    except Exception as exc:
        print(f"Download failed: {exc}. Check CDS API credentials and configuration.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
