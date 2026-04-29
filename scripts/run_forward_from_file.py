from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from mwsat.pipeline.run_forward import run_forward_simulation
from mwsat.utils.config import load_all_configs


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: python scripts/run_forward_from_file.py <path-to-era5-netcdf>")
        return 1

    path = sys.argv[1]

    try:
        configs = load_all_configs()
        result = run_forward_simulation(configs, path)
        print(f"Channels: {result['n_channels']}")
        print(f"First 5 TB values: {result['tb'][:5]}")
        return 0
    except Exception as exc:
        print(f"Error: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
