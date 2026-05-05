from __future__ import annotations

import argparse
from pathlib import Path


DATASET = "reanalysis-era5-pressure-levels"
DEFAULT_OUTPUT = Path("data/raw/era5/arts_era5_sample.nc")

PRESSURE_LEVELS = [
    "1",
    "2",
    "3",
    "5",
    "7",
    "10",
    "20",
    "30",
    "50",
    "70",
    "100",
    "125",
    "150",
    "175",
    "200",
    "225",
    "250",
    "300",
    "350",
    "400",
    "450",
    "500",
    "550",
    "600",
    "650",
    "700",
    "750",
    "775",
    "800",
    "825",
    "850",
    "875",
    "900",
    "925",
    "950",
    "975",
    "1000",
]


def build_request() -> dict:
    return {
        "product_type": ["reanalysis"],
        "variable": [
            "geopotential",
            "specific_humidity",
            "temperature",
        ],
        "year": ["2020"],
        "month": ["01"],
        "day": ["01"],
        "time": [
            "00:00",
            "06:00",
            "12:00",
            "18:00",
        ],
        "pressure_level": PRESSURE_LEVELS,
        "data_format": "netcdf",
        "download_format": "unarchived",
        "area": [58, 10, 57, 11],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download a compact ERA5 pressure-level sample for ARTS tests."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output NetCDF path. Defaults to {DEFAULT_OUTPUT}",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the output file if it already exists.",
    )
    return parser.parse_args()


def print_request_summary(output_path: Path, request: dict) -> None:
    print(f"Output path: {output_path.resolve()}")
    print(f"Dataset: {DATASET}")
    print(f"Variables: {', '.join(request['variable'])}")
    print(f"Date: {request['year'][0]}-{request['month'][0]}-{request['day'][0]}")
    print(f"Times: {', '.join(request['time'])}")
    print(
        "Pressure levels: "
        f"{len(request['pressure_level'])} levels "
        f"({request['pressure_level'][0]}-{request['pressure_level'][-1]} hPa)"
    )
    print(f"Area [N, W, S, E]: {request['area']}")
    print(f"Data format: {request['data_format']}")
    print(f"Download format: {request['download_format']}")


def main() -> int:
    args = parse_args()
    output_path = args.output
    request = build_request()

    if output_path.exists() and not args.overwrite:
        print(f"Output file already exists: {output_path.resolve()}")
        print("Use --overwrite to replace it.")
        return 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    print_request_summary(output_path, request)

    try:
        import cdsapi

        client = cdsapi.Client()
        client.retrieve(DATASET, request).download(str(output_path))
    except Exception as exc:
        print(f"Download failed: {exc}. Check CDS API credentials and configuration.")
        return 1

    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"Downloaded file: {output_path.resolve()}")
    print(f"File size: {file_size_mb:.2f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
