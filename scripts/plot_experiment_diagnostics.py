from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = PROJECT_ROOT / "data" / "processed" / "experiments"
OUTPUT_DIR = PROJECT_ROOT / "reports" / "figures" / "experiments"

TEMPERATURE_SENSITIVITY_CSV = INPUT_DIR / "multi_profile_temperature_sensitivity.csv"
LAYER_SENSITIVITY_CSV = INPUT_DIR / "multi_profile_layer_sensitivity.csv"


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    temperature_sensitivity = pd.read_csv(TEMPERATURE_SENSITIVITY_CSV)
    layer_sensitivity = pd.read_csv(LAYER_SENSITIVITY_CSV)
    return temperature_sensitivity, layer_sensitivity


def save_temperature_sensitivity_plot(
    temperature_sensitivity: pd.DataFrame,
) -> Path:
    output_path = OUTPUT_DIR / "temperature_sensitivity_vs_frequency.png"
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for profile_index, group in temperature_sensitivity.groupby("profile_index"):
        group = group.sort_values("frequency_ghz")
        ax.plot(
            group["frequency_ghz"],
            group["dTb_dT"],
            marker="o",
            label=f"Profile {profile_index}",
        )
    ax.set_title("Temperature Sensitivity vs Frequency")
    ax.set_xlabel("Frequency (GHz)")
    ax.set_ylabel("dTb/dT (K/K)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def save_sensitivity_to_noise_plot(
    temperature_sensitivity: pd.DataFrame,
) -> Path:
    output_path = OUTPUT_DIR / "sensitivity_to_noise_vs_frequency.png"
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for profile_index, group in temperature_sensitivity.groupby("profile_index"):
        group = group.sort_values("frequency_ghz")
        ax.plot(
            group["frequency_ghz"],
            group["sensitivity_to_noise_ratio"],
            marker="o",
            label=f"Profile {profile_index}",
        )
    ax.set_title("Sensitivity-to-Noise Ratio vs Frequency")
    ax.set_xlabel("Frequency (GHz)")
    ax.set_ylabel("|dTb/dT| / NEΔT")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def save_layer_sensitivity_ratio_plot(layer_sensitivity: pd.DataFrame) -> Path:
    output_path = OUTPUT_DIR / "layer_sensitivity_ratio_vs_frequency.png"
    summary = (
        layer_sensitivity.groupby("frequency_ghz", as_index=False)
        .agg(
            mean_lower_ratio=("lower_ratio", "mean"),
            mean_upper_ratio=("upper_ratio", "mean"),
        )
        .sort_values("frequency_ghz")
    )

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(
        summary["frequency_ghz"],
        summary["mean_lower_ratio"],
        marker="o",
        label="Mean lower_ratio",
    )
    ax.plot(
        summary["frequency_ghz"],
        summary["mean_upper_ratio"],
        marker="o",
        label="Mean upper_ratio",
    )
    ax.set_title("Layer Sensitivity Ratio vs Frequency")
    ax.set_xlabel("Frequency (GHz)")
    ax.set_ylabel("Ratio")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    temperature_sensitivity, layer_sensitivity = load_inputs()

    output_paths = [
        save_temperature_sensitivity_plot(temperature_sensitivity),
        save_sensitivity_to_noise_plot(temperature_sensitivity),
        save_layer_sensitivity_ratio_plot(layer_sensitivity),
    ]

    for output_path in output_paths:
        print(output_path)


if __name__ == "__main__":
    main()
