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
    add_degraded_observations,
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
    / "channel_ablation_metrics.csv"
)
FIGURE_DIR = PROJECT_ROOT / "reports" / "figures" / "experiments"

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
CHANNEL_COMBINATIONS = [
    [50.3, 52.8, 54.4],
    [50.3, 52.8],
    [50.3, 54.4],
    [52.8, 54.4],
    [50.3],
    [52.8],
    [54.4],
]
COMBINATION_ORDER = [
    "50.3+52.8+54.4",
    "50.3+52.8",
    "50.3+54.4",
    "52.8+54.4",
    "50.3",
    "52.8",
    "54.4",
]
TARGET_COLORS = {
    "lower_layer_mean_temperature_k": "tab:blue",
    "upper_layer_mean_temperature_k": "tab:orange",
}


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


def combination_name(channels_ghz: list[float]) -> str:
    return "+".join(f"{channel:.1f}" for channel in channels_ghz)


def feature_columns_for_combination(channels_ghz: list[float]) -> list[str]:
    return [frequency_feature_name(channel_ghz, "tb_obs") for channel_ghz in channels_ghz]


def build_metric_row(
    target_name: str,
    setup_type: str,
    group_name: str,
    model_name: str,
    channel_combination: str,
    n_channels: int,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_std_k: float,
    baseline_rmse_k: float,
    n_samples: int,
) -> dict[str, object]:
    metrics = compute_metrics(y_true, y_pred)
    return {
        "target_name": target_name,
        "setup_type": setup_type,
        "group_name": group_name,
        "model_name": model_name,
        "channel_combination": channel_combination,
        "n_channels": n_channels,
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
    channels_ghz: list[float],
) -> pd.DataFrame:
    feature_columns = feature_columns_for_combination(channels_ghz)
    x = era5_dataset[feature_columns].to_numpy(dtype=float)
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
    channel_label = combination_name(channels_ghz)
    rows = [
        build_metric_row(
            target_name=target_name,
            setup_type="global",
            group_name="all_era5",
            model_name=model_name,
            channel_combination=channel_label,
            n_channels=len(channels_ghz),
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
    channels_ghz: list[float],
) -> pd.DataFrame:
    feature_columns = feature_columns_for_combination(channels_ghz)
    channel_label = combination_name(channels_ghz)
    rows: list[dict[str, object]] = []
    pooled_truth: list[float] = []
    pooled_predictions = {model_name: [] for model_name in MODEL_ORDER}

    for source_name, source_group in era5_dataset.groupby("source_name", sort=True):
        x = source_group[feature_columns].to_numpy(dtype=float)
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
                    setup_type="source_name",
                    group_name=str(source_name),
                    model_name=model_name,
                    channel_combination=channel_label,
                    n_channels=len(channels_ghz),
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
                setup_type="source_name_pooled",
                group_name="all_sources_pooled",
                model_name=model_name,
                channel_combination=channel_label,
                n_channels=len(channels_ghz),
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
    ridge_global = metrics.loc[
        (metrics["setup_type"] == "global") & (metrics["model_name"] == "ridge_regression")
    ].copy()
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.8), sharey=False)
    for axis, target_name in zip(axes, TARGETS, strict=True):
        target_data = ridge_global.loc[ridge_global["target_name"] == target_name].copy()
        target_data["channel_combination"] = pd.Categorical(
            target_data["channel_combination"],
            categories=COMBINATION_ORDER,
            ordered=True,
        )
        target_data = target_data.sort_values("channel_combination")
        axis.bar(
            np.arange(len(target_data)),
            target_data[metric_column],
            color=TARGET_COLORS[target_name],
        )
        axis.set_title(TARGET_LABELS[target_name])
        axis.set_xticks(np.arange(len(target_data)))
        axis.set_xticklabels(target_data["channel_combination"], rotation=30, ha="right")
        axis.grid(True, axis="y", alpha=0.3)
    axes[0].set_ylabel(ylabel)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def print_summary(metrics: pd.DataFrame) -> None:
    global_ridge = metrics.loc[
        (metrics["setup_type"] == "global") & (metrics["model_name"] == "ridge_regression")
    ].copy()
    source_specific_ridge = metrics.loc[
        (metrics["setup_type"] == "source_name_pooled")
        & (metrics["model_name"] == "ridge_regression")
    ].copy()
    print("Channel ablation summary:")
    print("Setup | Target | Channels | RMSE | NRMSE | Skill")
    for frame, label in ((global_ridge, "global"), (source_specific_ridge, "source_specific")):
        frame["channel_combination"] = pd.Categorical(
            frame["channel_combination"],
            categories=COMBINATION_ORDER,
            ordered=True,
        )
        for row in frame.sort_values(["target_name", "channel_combination"]).itertuples(index=False):
            print(
                f"{label} | {row.target_name} | {row.channel_combination} | "
                f"{row.rmse_k:.3f} K | {row.normalized_rmse:.3f} | "
                f"{row.rmse_reduction_vs_baseline_pct:.1f}%"
            )


def main() -> None:
    observations = pd.read_csv(INPUT_PATH)
    observations = add_degraded_observations(observations)
    retrieval_dataset = pivot_observations(observations)
    era5_dataset = retrieval_dataset.loc[
        ~retrieval_dataset["region_name"].map(is_synthetic_region)
    ].copy()

    metric_frames: list[pd.DataFrame] = []
    for channels_ghz in CHANNEL_COMBINATIONS:
        for target_name in TARGETS:
            metric_frames.append(evaluate_global_target(era5_dataset, target_name, channels_ghz))
            metric_frames.append(
                evaluate_source_specific_target(era5_dataset, target_name, channels_ghz)
            )

    metrics = pd.concat(metric_frames, ignore_index=True)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(OUTPUT_PATH, index=False)

    output_paths = [
        plot_global_metric(
            metrics,
            "rmse_k",
            "rmse_by_channel_combination.png",
            "RMSE (K)",
            "Global Ridge RMSE by Channel Combination",
        ),
        plot_global_metric(
            metrics,
            "normalized_rmse",
            "normalized_rmse_by_channel_combination.png",
            "Normalized RMSE",
            "Global Ridge Normalized RMSE by Channel Combination",
        ),
        plot_global_metric(
            metrics,
            "rmse_reduction_vs_baseline_pct",
            "skill_by_channel_combination.png",
            "RMSE reduction vs baseline (%)",
            "Global Ridge Skill by Channel Combination",
        ),
        OUTPUT_PATH,
    ]

    print_summary(metrics)
    for output_path in output_paths:
        print(output_path)


if __name__ == "__main__":
    main()
