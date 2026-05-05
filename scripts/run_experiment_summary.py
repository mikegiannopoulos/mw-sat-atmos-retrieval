from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from mwsat.pipeline.run_experiment import run_experiment_summary  # noqa: E402
from mwsat.utils.config import load_all_configs  # noqa: E402


def main() -> int:
    try:
        configs = load_all_configs()
        output = run_experiment_summary(configs)
        summary = output["summary"]

        print(f"Number of profiles: {summary['n_profiles']}")
        print(f"Mean bias: {summary['mean_bias']}")
        print(f"Mean RMSE: {summary['mean_rmse']}")
        return 0
    except Exception as exc:
        print(f"Error: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
