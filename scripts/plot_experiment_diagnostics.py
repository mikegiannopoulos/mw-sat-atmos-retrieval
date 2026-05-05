from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = PROJECT_ROOT / "data" / "processed" / "experiments"
OUTPUT_DIR = PROJECT_ROOT / "reports" / "figures" / "experiments"

TEMPERATURE_SENSITIVITY_CSV = INPUT_DIR / "multi_profile_temperature_sensitivity.csv"
LAYER_SENSITIVITY_CSV = INPUT_DIR / "multi_profile_layer_sensitivity.csv"
HUMIDITY_SENSITIVITY_CSV = INPUT_DIR / "multi_profile_humidity_sensitivity.csv"
LINEAR_RETRIEVAL_RESULTS_CSV = INPUT_DIR / "linear_retrieval_results.csv"


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    temperature_sensitivity = pd.read_csv(TEMPERATURE_SENSITIVITY_CSV)
    layer_sensitivity = pd.read_csv(LAYER_SENSITIVITY_CSV)
    humidity_sensitivity = pd.read_csv(HUMIDITY_SENSITIVITY_CSV)
    linear_retrieval_results = pd.read_csv(LINEAR_RETRIEVAL_RESULTS_CSV)
    return (
        temperature_sensitivity,
        layer_sensitivity,
        humidity_sensitivity,
        linear_retrieval_results,
    )


def is_synthetic_profile(region_name: str) -> bool:
    return str(region_name).startswith("synthetic_")


def synthetic_profile_label(region_name: str) -> str:
    return str(region_name).replace("synthetic_", "Synthetic: ").replace("_", " ")


def save_temperature_sensitivity_plot(
    temperature_sensitivity: pd.DataFrame,
) -> Path:
    output_path = OUTPUT_DIR / "temperature_sensitivity_vs_frequency.png"
    fig, ax = plt.subplots(figsize=(7, 4.5))
    era5_label_used = False
    synthetic_styles = {
        "synthetic_warm_moist": {"color": "tab:red", "marker": "s"},
        "synthetic_cold_dry": {"color": "tab:blue", "marker": "^"},
        "synthetic_strong_lapse": {"color": "tab:green", "marker": "D"},
    }
    for _, group in temperature_sensitivity.groupby("profile_index"):
        group = group.sort_values("frequency_ghz")
        region_name = str(group["region_name"].iloc[0])
        if is_synthetic_profile(region_name):
            style = synthetic_styles.get(region_name, {"color": "tab:orange", "marker": "o"})
            ax.plot(
                group["frequency_ghz"],
                group["dTb_dT"],
                color=style["color"],
                marker=style["marker"],
                linewidth=2.2,
                markersize=6,
                label=synthetic_profile_label(region_name),
            )
        else:
            ax.plot(
                group["frequency_ghz"],
                group["dTb_dT"],
                color="0.7",
                marker="o",
                linewidth=0.9,
                markersize=3,
                alpha=0.6,
                label="ERA5 regional profiles" if not era5_label_used else None,
            )
            era5_label_used = True
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
    era5_label_used = False
    synthetic_styles = {
        "synthetic_warm_moist": {"color": "tab:red", "marker": "s"},
        "synthetic_cold_dry": {"color": "tab:blue", "marker": "^"},
        "synthetic_strong_lapse": {"color": "tab:green", "marker": "D"},
    }
    for _, group in temperature_sensitivity.groupby("profile_index"):
        group = group.sort_values("frequency_ghz")
        region_name = str(group["region_name"].iloc[0])
        if is_synthetic_profile(region_name):
            style = synthetic_styles.get(region_name, {"color": "tab:orange", "marker": "o"})
            ax.plot(
                group["frequency_ghz"],
                group["sensitivity_to_noise_ratio"],
                color=style["color"],
                marker=style["marker"],
                linewidth=2.2,
                markersize=6,
                label=synthetic_profile_label(region_name),
            )
        else:
            ax.plot(
                group["frequency_ghz"],
                group["sensitivity_to_noise_ratio"],
                color="0.7",
                marker="o",
                linewidth=0.9,
                markersize=3,
                alpha=0.6,
                label="ERA5 regional profiles" if not era5_label_used else None,
            )
            era5_label_used = True
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


def save_humidity_sensitivity_ratio_plot(humidity_sensitivity: pd.DataFrame) -> Path:
    output_path = OUTPUT_DIR / "humidity_sensitivity_ratio_vs_frequency.png"
    summary = (
        humidity_sensitivity.groupby("frequency_ghz", as_index=False)
        .agg(
            mean_humid_ratio=("humid_ratio", "mean"),
            mean_dry_ratio=("dry_ratio", "mean"),
        )
        .sort_values("frequency_ghz")
    )

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(
        summary["frequency_ghz"],
        summary["mean_humid_ratio"],
        marker="o",
        label="Mean humid_ratio",
    )
    ax.plot(
        summary["frequency_ghz"],
        summary["mean_dry_ratio"],
        marker="o",
        label="Mean dry_ratio",
    )
    ax.set_title("Humidity Sensitivity Ratio vs Frequency")
    ax.set_xlabel("Frequency (GHz)")
    ax.set_ylabel("Ratio")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def save_linear_retrieval_rmse_plot(linear_retrieval_results: pd.DataFrame) -> Path:
    output_path = OUTPUT_DIR / "linear_retrieval_rmse_by_channel_set.png"
    results = linear_retrieval_results.loc[
        linear_retrieval_results["evaluation_split"] == "era5_leave_one_profile_out"
    ].copy()
    results["instrument_case"] = (
        results["noise_case"].str.replace("_", " ", regex=False)
        + "\n"
        + results["feature_case"].str.replace("_", " ", regex=False)
    )
    ordered_cases = [
        "baseline nedt\nfull 3channel",
        "baseline nedt\ndrop 50 3",
        "baseline nedt\ndrop 52 8",
        "baseline nedt\ndrop 54 4",
        "degraded nedt\nfull 3channel",
        "degraded nedt\ndrop 50 3",
        "degraded nedt\ndrop 52 8",
        "degraded nedt\ndrop 54 4",
    ]
    results["instrument_case"] = pd.Categorical(
        results["instrument_case"], categories=ordered_cases, ordered=True
    )
    results = results.sort_values(["instrument_case", "model_name"])

    fig, ax = plt.subplots(figsize=(9, 4.8))
    model_styles = {
        "mean_baseline": {"marker": "o", "label": "Mean baseline"},
        "linear_regression": {"marker": "s", "label": "Linear regression"},
        "ridge_regression": {"marker": "^", "label": "Ridge regression"},
    }
    x_positions = np.arange(len(ordered_cases))
    for model_name, style in model_styles.items():
        model_results = results.loc[results["model_name"] == model_name]
        ax.plot(
            x_positions,
            model_results["rmse_k"],
            marker=style["marker"],
            linewidth=1.8,
            label=style["label"],
        )

    ax.set_title("Linear Retrieval RMSE by Channel Set")
    ax.set_xlabel("Instrument / Channel Case")
    ax.set_ylabel("RMSE (K)")
    ax.set_xticks(x_positions)
    ax.set_xticklabels(ordered_cases, rotation=30, ha="right")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (
        temperature_sensitivity,
        layer_sensitivity,
        humidity_sensitivity,
        linear_retrieval_results,
    ) = load_inputs()

    output_paths = [
        save_temperature_sensitivity_plot(temperature_sensitivity),
        save_sensitivity_to_noise_plot(temperature_sensitivity),
        save_layer_sensitivity_ratio_plot(layer_sensitivity),
        save_humidity_sensitivity_ratio_plot(humidity_sensitivity),
        save_linear_retrieval_rmse_plot(linear_retrieval_results),
    ]

    for output_path in output_paths:
        print(output_path)


if __name__ == "__main__":
    main()
