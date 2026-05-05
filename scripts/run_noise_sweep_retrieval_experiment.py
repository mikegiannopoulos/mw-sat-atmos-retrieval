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
    / "noise_sweep_retrieval_metrics.csv"
)
FIGURE_DIR = PROJECT_ROOT / "reports" / "figures" / "experiments"

NOISE_FACTORS = [0.5, 1.0, 2.0, 3.0, 5.0]
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
SOURCE_COLORS = {
    "winter_midlatitude_maritime_sample": "tab:blue",
    "lower_latitude_maritime_2020": "tab:green",
    "high_latitude_continental_2020": "tab:purple",
}
FEATURE_COLUMNS = [frequency_feature_name(freq, "tb_obs") for freq in (50.3, 52.8, 54.4)]


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


def add_noise_sweep_observations(observations: pd.DataFrame) -> pd.DataFrame:
    observations = observations.copy()
    for factor in NOISE_FACTORS:
        column_name = frequency_feature_name(factor, "tb_obs_factor")
        observations[column_name] = np.nan

    for profile_index, group in observations.groupby("profile_index", sort=False):
        group = group.sort_values("frequency_ghz")
        rng = np.random.default_rng(MULTI_PROFILE_NOISE_BASE_SEED + int(profile_index))
        standard_noise = rng.normal(loc=0.0, scale=1.0, size=len(group))
        tb_true = group["tb_true_k"].to_numpy(dtype=float)
        nedt = group["nedt_k"].to_numpy(dtype=float)
        for factor in NOISE_FACTORS:
            column_name = frequency_feature_name(factor, "tb_obs_factor")
            observations.loc[group.index, column_name] = tb_true + standard_noise * (
                factor * nedt
            )

    return observations


def build_pivot_for_factor(
    observations: pd.DataFrame,
    factor: float,
) -> pd.DataFrame:
    factor_column = frequency_feature_name(factor, "tb_obs_factor")
    factor_observations = observations.copy()
    factor_observations["tb_obs_k"] = factor_observations[factor_column]
    # Keep the shared pivot helper happy without changing the existing retrieval scripts.
    factor_observations["tb_obs_degraded_k"] = factor_observations["tb_obs_k"]
    return pivot_observations(factor_observations)


def build_metric_row(
    target_name: str,
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
                model_name,
                x_train,
                y_train,
                x_test,
            )[0]

    target_std_k = target_std(era5_dataset, target_name)
    baseline_rmse_k = compute_metrics(y, predictions["mean_baseline"])["rmse_k"]
    rows = [
        build_metric_row(
            target_name=target_name,
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
                    model_name,
                    x_train,
                    y_train,
                    x_test,
                )[0]

        source_target_std_k = target_std(source_group, target_name)
        source_baseline_rmse_k = compute_metrics(y, predictions["mean_baseline"])["rmse_k"]
        for model_name in MODEL_ORDER:
            rows.append(
                build_metric_row(
                    target_name=target_name,
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


def plot_global_metric(
    metrics: pd.DataFrame,
    metric_column: str,
    output_name: str,
    ylabel: str,
    title: str,
) -> Path:
    output_path = FIGURE_DIR / output_name
    global_data = metrics.loc[metrics["setup_type"] == "global"].copy()
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.6), sharex=True)
    for axis, target_name in zip(axes, TARGETS, strict=True):
        target_data = global_data.loc[global_data["target_name"] == target_name].copy()
        for model_name in MODEL_ORDER:
            model_data = target_data.loc[target_data["model_name"] == model_name].sort_values(
                "noise_factor"
            )
            axis.plot(
                model_data["noise_factor"],
                model_data[metric_column],
                marker="o",
                linewidth=2.0,
                color=MODEL_COLORS[model_name],
                label=MODEL_LABELS[model_name],
            )
        axis.set_title(TARGET_LABELS[target_name])
        axis.set_xlabel("NEΔT factor")
        axis.grid(True, alpha=0.3)
    axes[0].set_ylabel(ylabel)
    axes[1].legend()
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def plot_regime_aware_rmse(metrics: pd.DataFrame) -> Path:
    output_path = FIGURE_DIR / "noise_sweep_regime_aware_rmse.png"
    regime_data = metrics.loc[
        (metrics["setup_type"] == "source_name") & (metrics["model_name"] == "ridge_regression")
    ].copy()
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.8), sharex=True)
    for axis, target_name in zip(axes, TARGETS, strict=True):
        target_data = regime_data.loc[regime_data["target_name"] == target_name].copy()
        for source_name, source_group in target_data.groupby("group_name", sort=True):
            source_group = source_group.sort_values("noise_factor")
            axis.plot(
                source_group["noise_factor"],
                source_group["rmse_k"],
                marker="o",
                linewidth=2.0,
                color=SOURCE_COLORS.get(str(source_name), "0.35"),
                label=str(source_name),
            )
        axis.set_title(TARGET_LABELS[target_name])
        axis.set_xlabel("NEΔT factor")
        axis.grid(True, alpha=0.3)
    axes[0].set_ylabel("RMSE (K)")
    axes[1].legend(fontsize=8)
    fig.suptitle("Source-Specific Ridge RMSE vs Instrument Noise")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def plot_target_comparison(metrics: pd.DataFrame) -> Path:
    output_path = FIGURE_DIR / "noise_sweep_target_comparison.png"
    pooled_regime_data = metrics.loc[
        (metrics["setup_type"] == "source_name_pooled")
        & (metrics["model_name"] == "ridge_regression")
    ].copy()
    global_data = metrics.loc[
        (metrics["setup_type"] == "global") & (metrics["model_name"] == "ridge_regression")
    ].copy()
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    for target_name in TARGETS:
        target_global = global_data.loc[global_data["target_name"] == target_name].sort_values(
            "noise_factor"
        )
        target_regime = pooled_regime_data.loc[
            pooled_regime_data["target_name"] == target_name
        ].sort_values("noise_factor")
        ax.plot(
            target_global["noise_factor"],
            target_global["rmse_k"],
            marker="o",
            linewidth=2.0,
            color="tab:blue" if target_name == TARGETS[0] else "tab:orange",
            label=f"Global: {TARGET_LABELS[target_name]}",
        )
        ax.plot(
            target_regime["noise_factor"],
            target_regime["rmse_k"],
            marker="s",
            linestyle="--",
            linewidth=2.0,
            color="tab:blue" if target_name == TARGETS[0] else "tab:orange",
            label=f"Source-specific: {TARGET_LABELS[target_name]}",
        )
    ax.set_xlabel("NEΔT factor")
    ax.set_ylabel("RMSE (K)")
    ax.set_title("Lower vs Upper Retrieval Degradation (Ridge)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def print_summary(metrics: pd.DataFrame) -> None:
    global_ridge = metrics.loc[
        (metrics["setup_type"] == "global") & (metrics["model_name"] == "ridge_regression")
    ].copy()
    regime_ridge = metrics.loc[
        (metrics["setup_type"] == "source_name_pooled")
        & (metrics["model_name"] == "ridge_regression")
    ].copy()
    print("Noise sweep retrieval summary:")
    print("Setup | Target | Noise factor | RMSE | MAE | Bias | NRMSE | Skill")
    for row in global_ridge.sort_values(["target_name", "noise_factor"]).itertuples(index=False):
        print(
            f"global | {row.target_name} | {row.noise_factor:.1f} | "
            f"{row.rmse_k:.3f} K | {row.mae_k:.3f} K | {row.bias_k:+.3f} K | "
            f"{row.normalized_rmse:.3f} | {row.rmse_reduction_vs_baseline_pct:.1f}%"
        )
    for row in regime_ridge.sort_values(["target_name", "noise_factor"]).itertuples(index=False):
        print(
            f"source_specific | {row.target_name} | {row.noise_factor:.1f} | "
            f"{row.rmse_k:.3f} K | {row.mae_k:.3f} K | {row.bias_k:+.3f} K | "
            f"{row.normalized_rmse:.3f} | {row.rmse_reduction_vs_baseline_pct:.1f}%"
        )


def main() -> None:
    observations = pd.read_csv(INPUT_PATH)
    observations = add_noise_sweep_observations(observations)

    metric_frames: list[pd.DataFrame] = []
    for noise_factor in NOISE_FACTORS:
        retrieval_dataset = build_pivot_for_factor(observations, noise_factor)
        era5_dataset = retrieval_dataset.loc[
            ~retrieval_dataset["region_name"].map(is_synthetic_region)
        ].copy()
        for target_name in TARGETS:
            metric_frames.append(
                evaluate_global_target(era5_dataset, target_name, noise_factor)
            )
            metric_frames.append(
                evaluate_source_specific_target(era5_dataset, target_name, noise_factor)
            )

    metrics = pd.concat(metric_frames, ignore_index=True)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(OUTPUT_PATH, index=False)

    output_paths = [
        plot_global_metric(
            metrics,
            "rmse_k",
            "noise_sweep_rmse.png",
            "RMSE (K)",
            "Global Retrieval RMSE vs Instrument Noise",
        ),
        plot_global_metric(
            metrics,
            "normalized_rmse",
            "noise_sweep_normalized_rmse.png",
            "Normalized RMSE",
            "Global Normalized RMSE vs Instrument Noise",
        ),
        plot_global_metric(
            metrics,
            "rmse_reduction_vs_baseline_pct",
            "noise_sweep_skill_vs_baseline.png",
            "RMSE reduction vs baseline (%)",
            "Global Retrieval Skill vs Instrument Noise",
        ),
        plot_regime_aware_rmse(metrics),
        plot_target_comparison(metrics),
        OUTPUT_PATH,
    ]

    print_summary(metrics)
    for output_path in output_paths:
        print(output_path)


if __name__ == "__main__":
    main()
