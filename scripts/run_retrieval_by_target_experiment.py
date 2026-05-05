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
    feature_sets,
    is_synthetic_region,
    pivot_observations,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RETRIEVAL_OUTPUT_PATH = (
    PROJECT_ROOT / "data" / "processed" / "experiments" / "retrieval_metrics_by_target.csv"
)
REGIME_AWARE_OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "experiments"
    / "regime_aware_retrieval_metrics_by_target.csv"
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
FEATURE_ORDER = ["full_3channel", "drop_50_3", "drop_52_8", "drop_54_4"]
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


def build_metric_row(
    target_name: str,
    model_name: str,
    feature_case: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_std_k: float,
    baseline_rmse_k: float,
    n_samples: int,
    grouping: str,
    source_name: str | None = None,
) -> dict[str, object]:
    metrics = compute_metrics(y_true, y_pred)
    return {
        "target_name": target_name,
        "model_name": model_name,
        "feature_case": feature_case,
        "grouping": grouping,
        "source_name": source_name,
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
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for feature_case, feature_columns in feature_sets("tb_obs").items():
        x = era5_dataset[feature_columns].to_numpy(dtype=float)
        y = era5_dataset[target_name].to_numpy(dtype=float)
        predictions = {model_name: np.zeros(len(era5_dataset), dtype=float) for model_name in MODEL_ORDER}
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
        for model_name in MODEL_ORDER:
            rows.append(
                build_metric_row(
                    target_name=target_name,
                    model_name=model_name,
                    feature_case=feature_case,
                    y_true=y,
                    y_pred=predictions[model_name],
                    target_std_k=target_std_k,
                    baseline_rmse_k=baseline_rmse_k,
                    n_samples=len(era5_dataset),
                    grouping="global",
                )
            )
    return pd.DataFrame(rows)


def evaluate_source_specific_target(
    era5_dataset: pd.DataFrame,
    target_name: str,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for feature_case, feature_columns in feature_sets("tb_obs").items():
        pooled_truth: list[float] = []
        pooled_predictions = {model_name: [] for model_name in MODEL_ORDER}
        for source_name, source_group in era5_dataset.groupby("source_name", sort=True):
            x = source_group[feature_columns].to_numpy(dtype=float)
            y = source_group[target_name].to_numpy(dtype=float)
            predictions = {model_name: np.zeros(len(source_group), dtype=float) for model_name in MODEL_ORDER}
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
                        model_name=model_name,
                        feature_case=feature_case,
                        y_true=y,
                        y_pred=predictions[model_name],
                        target_std_k=source_target_std_k,
                        baseline_rmse_k=source_baseline_rmse_k,
                        n_samples=len(source_group),
                        grouping="source_name",
                        source_name=str(source_name),
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
                    model_name=model_name,
                    feature_case=feature_case,
                    y_true=pooled_truth_array,
                    y_pred=np.asarray(pooled_predictions[model_name], dtype=float),
                    target_std_k=pooled_target_std_k,
                    baseline_rmse_k=pooled_baseline_rmse_k,
                    n_samples=len(era5_dataset),
                    grouping="source_name_pooled",
                )
            )
    return pd.DataFrame(rows)


def plot_grouped_metric(
    data: pd.DataFrame,
    metric_column: str,
    output_name: str,
    ylabel: str,
    title: str,
    model_subset: list[str] | None = None,
) -> Path:
    output_path = FIGURE_DIR / output_name
    if model_subset is None:
        model_order = MODEL_ORDER
    else:
        model_order = model_subset
        data = data.loc[data["model_name"].isin(model_subset)].copy()

    target_order = TARGETS
    width = 0.22 if len(model_order) == 3 else 0.28
    x_positions = np.arange(len(target_order))
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    center_offset = (len(model_order) - 1) / 2
    for offset_index, model_name in enumerate(model_order):
        model_data = (
            data.loc[data["model_name"] == model_name]
            .set_index("target_name")
            .reindex(target_order)
        )
        ax.bar(
            x_positions + (offset_index - center_offset) * width,
            model_data[metric_column],
            width=width,
            label=MODEL_LABELS[model_name],
            color=MODEL_COLORS[model_name],
        )
    ax.set_xticks(x_positions)
    ax.set_xticklabels([TARGET_LABELS[target_name] for target_name in target_order], rotation=15, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def plot_channel_importance_by_target(
    data: pd.DataFrame,
    output_name: str,
    title: str,
) -> Path:
    output_path = FIGURE_DIR / output_name
    ridge_data = data.loc[data["model_name"] == "ridge_regression"].copy()
    width = 0.35
    x_positions = np.arange(len(FEATURE_ORDER))
    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    for offset_index, target_name in enumerate(TARGETS):
        target_data = (
            ridge_data.loc[ridge_data["target_name"] == target_name]
            .set_index("feature_case")
            .reindex(FEATURE_ORDER)
        )
        ax.bar(
            x_positions + (offset_index - 0.5) * width,
            target_data["rmse_k"],
            width=width,
            label=TARGET_LABELS[target_name],
            color=TARGET_COLORS[target_name],
        )
    ax.set_xticks(x_positions)
    ax.set_xticklabels(FEATURE_ORDER, rotation=20, ha="right")
    ax.set_ylabel("RMSE (K)")
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def print_summary_table(global_metrics: pd.DataFrame, regime_metrics: pd.DataFrame) -> None:
    print("Retrieval-by-target summary:")
    print("Type | Target | Feature | Model | RMSE | MAE | Bias | Target std | NRMSE | Skill")
    for row in global_metrics.itertuples(index=False):
        print(
            f"global | {row.target_name} | {row.feature_case} | {row.model_name} | "
            f"{row.rmse_k:.3f} K | {row.mae_k:.3f} K | {row.bias_k:+.3f} K | "
            f"{row.target_std_k:.3f} K | {row.normalized_rmse:.3f} | "
            f"{row.rmse_reduction_vs_baseline_pct:.1f}%"
        )
    for row in regime_metrics.loc[regime_metrics["grouping"] == "source_name_pooled"].itertuples(index=False):
        print(
            f"source_specific | {row.target_name} | {row.feature_case} | {row.model_name} | "
            f"{row.rmse_k:.3f} K | {row.mae_k:.3f} K | {row.bias_k:+.3f} K | "
            f"{row.target_std_k:.3f} K | {row.normalized_rmse:.3f} | "
            f"{row.rmse_reduction_vs_baseline_pct:.1f}%"
        )


def main() -> None:
    observations = pd.read_csv(INPUT_PATH)
    observations = add_degraded_observations(observations)
    retrieval_dataset = pivot_observations(observations)
    era5_dataset = retrieval_dataset.loc[
        ~retrieval_dataset["region_name"].map(is_synthetic_region)
    ].copy()

    global_frames = []
    regime_frames = []
    for target_name in TARGETS:
        global_frames.append(evaluate_global_target(era5_dataset, target_name))
        regime_frames.append(evaluate_source_specific_target(era5_dataset, target_name))

    global_metrics = pd.concat(global_frames, ignore_index=True)
    regime_metrics = pd.concat(regime_frames, ignore_index=True)

    RETRIEVAL_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    global_metrics.to_csv(RETRIEVAL_OUTPUT_PATH, index=False)
    regime_metrics.to_csv(REGIME_AWARE_OUTPUT_PATH, index=False)

    global_full = global_metrics.loc[global_metrics["feature_case"] == "full_3channel"].copy()
    regime_full = regime_metrics.loc[
        (regime_metrics["grouping"] == "source_name_pooled")
        & (regime_metrics["feature_case"] == "full_3channel")
    ].copy()

    output_paths = [
        plot_grouped_metric(
            global_full,
            "rmse_k",
            "retrieval_rmse_by_target.png",
            "RMSE (K)",
            "Global Retrieval RMSE by Target",
        ),
        plot_grouped_metric(
            global_full,
            "normalized_rmse",
            "retrieval_normalized_rmse_by_target.png",
            "Normalized RMSE",
            "Global Normalized RMSE by Target",
        ),
        plot_channel_importance_by_target(
            global_metrics,
            "retrieval_channel_importance_by_target.png",
            "Global Channel Importance by Target (Ridge)",
        ),
        plot_grouped_metric(
            regime_full,
            "rmse_k",
            "regime_aware_rmse_by_target.png",
            "RMSE (K)",
            "Source-Specific Retrieval RMSE by Target",
        ),
        plot_grouped_metric(
            regime_full,
            "normalized_rmse",
            "regime_aware_normalized_rmse_by_target.png",
            "Normalized RMSE",
            "Source-Specific Normalized RMSE by Target",
        ),
        plot_channel_importance_by_target(
            regime_metrics.loc[regime_metrics["grouping"] == "source_name_pooled"].copy(),
            "regime_aware_channel_importance_by_target.png",
            "Source-Specific Channel Importance by Target (Ridge)",
        ),
        RETRIEVAL_OUTPUT_PATH,
        REGIME_AWARE_OUTPUT_PATH,
    ]

    print_summary_table(global_metrics, regime_metrics)
    for output_path in output_paths:
        print(output_path)


if __name__ == "__main__":
    main()
