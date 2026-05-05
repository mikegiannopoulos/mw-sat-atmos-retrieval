from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from run_linear_retrieval_experiment import (
    INPUT_PATH,
    MULTI_PROFILE_NOISE_BASE_SEED,
    compute_metrics,
    frequency_feature_name,
    is_synthetic_region,
    pivot_observations,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "experiments"
    / "channel_noise_sensitivity_metrics.csv"
)
FIGURE_DIR = PROJECT_ROOT / "reports" / "figures" / "experiments"

CHANNELS_GHZ = [50.3, 52.8, 54.4]
NOISE_FACTORS = [1.0, 2.0, 3.0, 5.0]
TARGETS = [
    "lower_layer_mean_temperature_k",
    "upper_layer_mean_temperature_k",
]
TARGET_LABELS = {
    "lower_layer_mean_temperature_k": "Lower-layer mean temperature",
    "upper_layer_mean_temperature_k": "Upper-layer mean temperature",
}
MODEL_ORDER = ["mean_baseline", "linear_regression", "ridge_regression"]
MODEL_LABELS = {
    "mean_baseline": "Mean baseline",
    "linear_regression": "Linear regression",
    "ridge_regression": "Ridge regression",
}
MODEL_COLORS = {
    "mean_baseline": "0.55",
    "linear_regression": "tab:blue",
    "ridge_regression": "tab:orange",
}
CHANNEL_COLORS = {
    50.3: "tab:green",
    52.8: "tab:blue",
    54.4: "tab:red",
}
FEATURE_COLUMNS = [frequency_feature_name(freq, "tb_obs") for freq in CHANNELS_GHZ]


def fit_model_predict(
    model_name: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
) -> np.ndarray:
    if model_name == "mean_baseline":
        return np.full(len(x_test), y_train.mean(), dtype=float)
    if model_name == "linear_regression":
        model = LinearRegression()
    elif model_name == "ridge_regression":
        model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    else:
        raise ValueError(f"Unsupported model name: {model_name}")
    model.fit(x_train, y_train)
    return model.predict(x_test)


def target_std(dataset: pd.DataFrame, target_name: str) -> float:
    return float(np.std(dataset[target_name].to_numpy(dtype=float), ddof=0))


def channel_factor_column(channel_ghz: float, factor: float) -> str:
    return f"tb_obs_channel_{str(channel_ghz).replace('.', '_')}_factor_{str(factor).replace('.', '_')}"


def add_channel_noise_scenarios(observations: pd.DataFrame) -> pd.DataFrame:
    observations = observations.copy()
    for channel_ghz in CHANNELS_GHZ:
        for factor in NOISE_FACTORS:
            observations[channel_factor_column(channel_ghz, factor)] = np.nan

    for profile_index, group in observations.groupby("profile_index", sort=False):
        group = group.sort_values("frequency_ghz")
        rng = np.random.default_rng(MULTI_PROFILE_NOISE_BASE_SEED + int(profile_index))
        standard_noise = rng.normal(loc=0.0, scale=1.0, size=len(group))
        tb_true = group["tb_true_k"].to_numpy(dtype=float)
        nedt = group["nedt_k"].to_numpy(dtype=float)
        frequencies = group["frequency_ghz"].to_numpy(dtype=float)

        for target_channel_ghz in CHANNELS_GHZ:
            for factor in NOISE_FACTORS:
                scaled_nedt = nedt.copy()
                scaled_nedt[np.isclose(frequencies, target_channel_ghz)] *= factor
                scenario_tb_obs = tb_true + standard_noise * scaled_nedt
                observations.loc[group.index, channel_factor_column(target_channel_ghz, factor)] = (
                    scenario_tb_obs
                )

    return observations


def build_pivot_for_scenario(
    observations: pd.DataFrame,
    channel_ghz: float,
    factor: float,
) -> pd.DataFrame:
    scenario_column = channel_factor_column(channel_ghz, factor)
    scenario_observations = observations.copy()
    scenario_observations["tb_obs_k"] = scenario_observations[scenario_column]
    scenario_observations["tb_obs_degraded_k"] = scenario_observations["tb_obs_k"]
    return pivot_observations(scenario_observations)


def build_metric_row(
    target_name: str,
    degraded_channel_ghz: float,
    noise_factor: float,
    setup_type: str,
    group_name: str,
    model_name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_std_k: float,
    baseline_rmse_k: float,
    n_samples: int,
) -> dict[str, object]:
    metrics = compute_metrics(y_true, y_pred)
    return {
        "target_name": target_name,
        "degraded_channel_ghz": degraded_channel_ghz,
        "noise_factor": noise_factor,
        "setup_type": setup_type,
        "group_name": group_name,
        "model_name": model_name,
        "n_samples": n_samples,
        **metrics,
        "target_std_k": target_std_k,
        "normalized_rmse": metrics["rmse_k"] / target_std_k,
        "rmse_reduction_vs_baseline_pct": 100.0
        * (1.0 - metrics["rmse_k"] / baseline_rmse_k),
    }


def evaluate_global_target(
    era5_dataset: pd.DataFrame,
    target_name: str,
    degraded_channel_ghz: float,
    noise_factor: float,
) -> pd.DataFrame:
    x = era5_dataset[FEATURE_COLUMNS].to_numpy(dtype=float)
    y = era5_dataset[target_name].to_numpy(dtype=float)
    predictions = {
        model_name: np.zeros(len(era5_dataset), dtype=float) for model_name in MODEL_ORDER
    }
    for test_index in range(len(era5_dataset)):
        train_mask = np.ones(len(era5_dataset), dtype=bool)
        train_mask[test_index] = False
        x_train = x[train_mask]
        y_train = y[train_mask]
        x_test = x[[test_index]]
        for model_name in MODEL_ORDER:
            predictions[model_name][test_index] = fit_model_predict(
                model_name, x_train, y_train, x_test
            )[0]

    target_std_k = target_std(era5_dataset, target_name)
    baseline_rmse_k = compute_metrics(y, predictions["mean_baseline"])["rmse_k"]
    rows = [
        build_metric_row(
            target_name=target_name,
            degraded_channel_ghz=degraded_channel_ghz,
            noise_factor=noise_factor,
            setup_type="global",
            group_name="all_era5",
            model_name=model_name,
            y_true=y,
            y_pred=predictions[model_name],
            target_std_k=target_std_k,
            baseline_rmse_k=baseline_rmse_k,
            n_samples=len(era5_dataset),
        )
        for model_name in MODEL_ORDER
    ]
    return pd.DataFrame(rows)


def evaluate_source_specific_target(
    era5_dataset: pd.DataFrame,
    target_name: str,
    degraded_channel_ghz: float,
    noise_factor: float,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    pooled_truth: list[float] = []
    pooled_predictions = {model_name: [] for model_name in MODEL_ORDER}

    for source_name, source_group in era5_dataset.groupby("source_name", sort=True):
        x = source_group[FEATURE_COLUMNS].to_numpy(dtype=float)
        y = source_group[target_name].to_numpy(dtype=float)
        predictions = {
            model_name: np.zeros(len(source_group), dtype=float) for model_name in MODEL_ORDER
        }
        for test_index in range(len(source_group)):
            train_mask = np.ones(len(source_group), dtype=bool)
            train_mask[test_index] = False
            x_train = x[train_mask]
            y_train = y[train_mask]
            x_test = x[[test_index]]
            for model_name in MODEL_ORDER:
                predictions[model_name][test_index] = fit_model_predict(
                    model_name, x_train, y_train, x_test
                )[0]

        source_target_std_k = target_std(source_group, target_name)
        source_baseline_rmse_k = compute_metrics(y, predictions["mean_baseline"])["rmse_k"]
        for model_name in MODEL_ORDER:
            rows.append(
                build_metric_row(
                    target_name=target_name,
                    degraded_channel_ghz=degraded_channel_ghz,
                    noise_factor=noise_factor,
                    setup_type="source_name",
                    group_name=str(source_name),
                    model_name=model_name,
                    y_true=y,
                    y_pred=predictions[model_name],
                    target_std_k=source_target_std_k,
                    baseline_rmse_k=source_baseline_rmse_k,
                    n_samples=len(source_group),
                )
            )
            pooled_predictions[model_name].extend(predictions[model_name].tolist())
        pooled_truth.extend(y.tolist())

    pooled_truth_array = np.asarray(pooled_truth, dtype=float)
    pooled_target_std_k = target_std(era5_dataset, target_name)
    pooled_baseline_rmse_k = compute_metrics(
        pooled_truth_array,
        np.asarray(pooled_predictions["mean_baseline"], dtype=float),
    )["rmse_k"]
    for model_name in MODEL_ORDER:
        rows.append(
            build_metric_row(
                target_name=target_name,
                degraded_channel_ghz=degraded_channel_ghz,
                noise_factor=noise_factor,
                setup_type="source_name_pooled",
                group_name="all_sources_pooled",
                model_name=model_name,
                y_true=pooled_truth_array,
                y_pred=np.asarray(pooled_predictions[model_name], dtype=float),
                target_std_k=pooled_target_std_k,
                baseline_rmse_k=pooled_baseline_rmse_k,
                n_samples=len(era5_dataset),
            )
        )

    return pd.DataFrame(rows)


def plot_metric(
    metrics: pd.DataFrame,
    metric_column: str,
    output_name: str,
    ylabel: str,
    title: str,
) -> Path:
    output_path = FIGURE_DIR / output_name
    ridge_global = metrics.loc[
        (metrics["setup_type"] == "global") & (metrics["model_name"] == "ridge_regression")
    ].copy()
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.8), sharex=True)
    for axis, target_name in zip(axes, TARGETS, strict=True):
        target_data = ridge_global.loc[ridge_global["target_name"] == target_name].copy()
        for channel_ghz in CHANNELS_GHZ:
            channel_data = target_data.loc[
                np.isclose(target_data["degraded_channel_ghz"], channel_ghz)
            ].sort_values("noise_factor")
            axis.plot(
                channel_data["noise_factor"],
                channel_data[metric_column],
                marker="o",
                linewidth=2.0,
                color=CHANNEL_COLORS[channel_ghz],
                label=f"{channel_ghz:.1f} GHz",
            )
        axis.set_title(TARGET_LABELS[target_name])
        axis.set_xlabel("Scaled NEΔT factor for degraded channel")
        axis.grid(True, alpha=0.3)
    axes[0].set_ylabel(ylabel)
    axes[1].legend()
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def print_summary(metrics: pd.DataFrame) -> None:
    global_ridge = metrics.loc[
        (metrics["setup_type"] == "global") & (metrics["model_name"] == "ridge_regression")
    ].copy()
    source_pooled_ridge = metrics.loc[
        (metrics["setup_type"] == "source_name_pooled")
        & (metrics["model_name"] == "ridge_regression")
    ].copy()
    print("Channel-specific noise sensitivity summary:")
    print("Setup | Target | Channel | Factor | RMSE | NRMSE | Skill")
    for frame, label in ((global_ridge, "global"), (source_pooled_ridge, "source_specific")):
        for row in frame.sort_values(
            ["target_name", "degraded_channel_ghz", "noise_factor"]
        ).itertuples(index=False):
            print(
                f"{label} | {row.target_name} | {row.degraded_channel_ghz:.1f} GHz | "
                f"{row.noise_factor:.1f} | {row.rmse_k:.3f} K | "
                f"{row.normalized_rmse:.3f} | {row.rmse_reduction_vs_baseline_pct:.1f}%"
            )


def main() -> None:
    observations = pd.read_csv(INPUT_PATH)
    observations = add_channel_noise_scenarios(observations)

    metric_frames: list[pd.DataFrame] = []
    for degraded_channel_ghz in CHANNELS_GHZ:
        for noise_factor in NOISE_FACTORS:
            retrieval_dataset = build_pivot_for_scenario(
                observations,
                degraded_channel_ghz,
                noise_factor,
            )
            era5_dataset = retrieval_dataset.loc[
                ~retrieval_dataset["region_name"].map(is_synthetic_region)
            ].copy()
            for target_name in TARGETS:
                metric_frames.append(
                    evaluate_global_target(
                        era5_dataset,
                        target_name,
                        degraded_channel_ghz,
                        noise_factor,
                    )
                )
                metric_frames.append(
                    evaluate_source_specific_target(
                        era5_dataset,
                        target_name,
                        degraded_channel_ghz,
                        noise_factor,
                    )
                )

    metrics = pd.concat(metric_frames, ignore_index=True)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(OUTPUT_PATH, index=False)

    output_paths = [
        plot_metric(
            metrics,
            "rmse_k",
            "rmse_vs_noise_by_channel.png",
            "RMSE (K)",
            "Global Ridge RMSE vs Channel-Specific Noise",
        ),
        plot_metric(
            metrics,
            "normalized_rmse",
            "normalized_rmse_vs_noise_by_channel.png",
            "Normalized RMSE",
            "Global Ridge Normalized RMSE vs Channel-Specific Noise",
        ),
        plot_metric(
            metrics,
            "rmse_reduction_vs_baseline_pct",
            "skill_vs_noise_by_channel.png",
            "RMSE reduction vs baseline (%)",
            "Global Ridge Skill vs Channel-Specific Noise",
        ),
        OUTPUT_PATH,
    ]

    print_summary(metrics)
    for output_path in output_paths:
        print(output_path)


if __name__ == "__main__":
    main()
