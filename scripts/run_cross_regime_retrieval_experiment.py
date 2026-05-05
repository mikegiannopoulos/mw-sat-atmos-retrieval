from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import LeaveOneOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from run_linear_retrieval_experiment import (
    INPUT_PATH,
    PRIMARY_TARGET,
    add_degraded_observations,
    compute_metrics,
    pivot_observations,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = (
    PROJECT_ROOT / "data" / "processed" / "experiments" / "cross_regime_metrics.csv"
)
FIGURE_DIR = PROJECT_ROOT / "reports" / "figures" / "experiments"

FEATURE_COLUMNS = ["tb_obs_50_3", "tb_obs_52_8", "tb_obs_54_4"]
SOURCE_ORDER = [
    "winter_midlatitude_maritime_sample",
    "lower_latitude_maritime_2020",
    "high_latitude_continental_2020",
]
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


def rmse(values: pd.Series) -> float:
    return float(np.sqrt(np.mean(np.square(values.to_numpy(dtype=float)))))


def source_std(dataset: pd.DataFrame) -> float:
    return float(np.std(dataset[PRIMARY_TARGET].to_numpy(dtype=float), ddof=0))


def ordered_sources(data: pd.DataFrame) -> list[str]:
    present_sources = list(pd.unique(data["source_name"]))
    ordered = [source_name for source_name in SOURCE_ORDER if source_name in present_sources]
    ordered.extend(source_name for source_name in present_sources if source_name not in ordered)
    return ordered


def build_row(
    evaluation_mode: str,
    test_source_name: str,
    train_source_names: str,
    model_name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_std_k: float,
    baseline_rmse_k: float,
    n_train: int,
    n_test: int,
) -> dict[str, object]:
    metrics = compute_metrics(y_true, y_pred)
    return {
        "evaluation_mode": evaluation_mode,
        "test_source_name": test_source_name,
        "train_source_names": train_source_names,
        "model_name": model_name,
        "target_name": PRIMARY_TARGET,
        "n_train": n_train,
        "n_test": n_test,
        **metrics,
        "target_std_k": target_std_k,
        "normalized_rmse": metrics["rmse_k"] / target_std_k,
        "rmse_reduction_vs_baseline_pct": 100.0
        * (1.0 - metrics["rmse_k"] / baseline_rmse_k),
    }


def evaluate_in_distribution_by_source(
    era5_dataset: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []
    for source_name in ordered_sources(era5_dataset):
        source_data = era5_dataset.loc[era5_dataset["source_name"] == source_name].copy()
        x = source_data[FEATURE_COLUMNS].to_numpy(dtype=float)
        y = source_data[PRIMARY_TARGET].to_numpy(dtype=float)
        target_std_k = source_std(source_data)
        loo = LeaveOneOut()
        predictions = {
            model_name: np.zeros(len(source_data), dtype=float) for model_name in MODEL_ORDER
        }

        for train_index, test_index in loo.split(x):
            x_train = x[train_index]
            y_train = y[train_index]
            x_test = x[test_index]
            for model_name in MODEL_ORDER:
                predictions[model_name][test_index] = fit_model_predict(
                    model_name, x_train, y_train, x_test
                )

        baseline_rmse_k = compute_metrics(y, predictions["mean_baseline"])["rmse_k"]
        for model_name in MODEL_ORDER:
            metric_rows.append(
                build_row(
                    evaluation_mode="in_distribution_leave_one_profile_out",
                    test_source_name=source_name,
                    train_source_names=source_name,
                    model_name=model_name,
                    y_true=y,
                    y_pred=predictions[model_name],
                    target_std_k=target_std_k,
                    baseline_rmse_k=baseline_rmse_k,
                    n_train=len(source_data) - 1,
                    n_test=len(source_data),
                )
            )
            for row_index, prediction in enumerate(predictions[model_name]):
                row = source_data.iloc[row_index]
                prediction_rows.append(
                    {
                        "evaluation_mode": "in_distribution_leave_one_profile_out",
                        "test_source_name": source_name,
                        "train_source_names": source_name,
                        "model_name": model_name,
                        "profile_id": row["profile_id"],
                        "valid_time": row["valid_time"],
                        "latitude": row["latitude"],
                        "longitude": row["longitude"],
                        "y_true_k": float(y[row_index]),
                        "y_pred_k": float(prediction),
                        "residual_k": float(prediction - y[row_index]),
                    }
                )

    return pd.DataFrame(metric_rows), pd.DataFrame(prediction_rows)


def evaluate_cross_regime(
    era5_dataset: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []
    for test_source_name in ordered_sources(era5_dataset):
        test_data = era5_dataset.loc[era5_dataset["source_name"] == test_source_name].copy()
        train_data = era5_dataset.loc[era5_dataset["source_name"] != test_source_name].copy()

        x_train = train_data[FEATURE_COLUMNS].to_numpy(dtype=float)
        y_train = train_data[PRIMARY_TARGET].to_numpy(dtype=float)
        x_test = test_data[FEATURE_COLUMNS].to_numpy(dtype=float)
        y_test = test_data[PRIMARY_TARGET].to_numpy(dtype=float)
        target_std_k = source_std(test_data)
        predictions: dict[str, np.ndarray] = {}
        for model_name in MODEL_ORDER:
            predictions[model_name] = fit_model_predict(model_name, x_train, y_train, x_test)

        baseline_rmse_k = compute_metrics(y_test, predictions["mean_baseline"])["rmse_k"]
        train_sources = ",".join(ordered_sources(train_data))
        for model_name in MODEL_ORDER:
            metric_rows.append(
                build_row(
                    evaluation_mode="cross_regime_holdout",
                    test_source_name=test_source_name,
                    train_source_names=train_sources,
                    model_name=model_name,
                    y_true=y_test,
                    y_pred=predictions[model_name],
                    target_std_k=target_std_k,
                    baseline_rmse_k=baseline_rmse_k,
                    n_train=len(train_data),
                    n_test=len(test_data),
                )
            )
            for row_index, prediction in enumerate(predictions[model_name]):
                row = test_data.iloc[row_index]
                prediction_rows.append(
                    {
                        "evaluation_mode": "cross_regime_holdout",
                        "test_source_name": test_source_name,
                        "train_source_names": train_sources,
                        "model_name": model_name,
                        "profile_id": row["profile_id"],
                        "valid_time": row["valid_time"],
                        "latitude": row["latitude"],
                        "longitude": row["longitude"],
                        "y_true_k": float(y_test[row_index]),
                        "y_pred_k": float(prediction),
                        "residual_k": float(prediction - y_test[row_index]),
                    }
                )

    return pd.DataFrame(metric_rows), pd.DataFrame(prediction_rows)


def save_cross_regime_rmse_plot(metrics: pd.DataFrame) -> Path:
    output_path = FIGURE_DIR / "cross_regime_rmse.png"
    data = metrics.loc[metrics["evaluation_mode"] == "cross_regime_holdout"].copy()
    source_order = ordered_sources(data.rename(columns={"test_source_name": "source_name"}))
    width = 0.22
    x_positions = np.arange(len(source_order))
    fig, ax = plt.subplots(figsize=(9, 4.8))
    for offset_index, model_name in enumerate(MODEL_ORDER):
        model_data = (
            data.loc[data["model_name"] == model_name]
            .set_index("test_source_name")
            .reindex(source_order)
        )
        ax.bar(
            x_positions + (offset_index - 1) * width,
            model_data["rmse_k"],
            width=width,
            label=MODEL_LABELS[model_name],
            color=MODEL_COLORS[model_name],
        )
    ax.set_xticks(x_positions)
    ax.set_xticklabels(source_order, rotation=20, ha="right")
    ax.set_ylabel("RMSE (K)")
    ax.set_title("Cross-Regime Retrieval RMSE")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def save_cross_regime_normalized_rmse_plot(metrics: pd.DataFrame) -> Path:
    output_path = FIGURE_DIR / "cross_regime_normalized_rmse.png"
    data = metrics.loc[metrics["evaluation_mode"] == "cross_regime_holdout"].copy()
    source_order = ordered_sources(data.rename(columns={"test_source_name": "source_name"}))
    width = 0.22
    x_positions = np.arange(len(source_order))
    fig, ax = plt.subplots(figsize=(9, 4.8))
    for offset_index, model_name in enumerate(MODEL_ORDER):
        model_data = (
            data.loc[data["model_name"] == model_name]
            .set_index("test_source_name")
            .reindex(source_order)
        )
        ax.bar(
            x_positions + (offset_index - 1) * width,
            model_data["normalized_rmse"],
            width=width,
            label=MODEL_LABELS[model_name],
            color=MODEL_COLORS[model_name],
        )
    ax.set_xticks(x_positions)
    ax.set_xticklabels(source_order, rotation=20, ha="right")
    ax.set_ylabel("Normalized RMSE")
    ax.set_title("Cross-Regime Normalized RMSE")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def save_cross_regime_skill_plot(metrics: pd.DataFrame) -> Path:
    output_path = FIGURE_DIR / "cross_regime_skill_vs_baseline.png"
    data = metrics.loc[
        (metrics["evaluation_mode"] == "cross_regime_holdout")
        & (metrics["model_name"].isin(["linear_regression", "ridge_regression"]))
    ].copy()
    source_order = ordered_sources(data.rename(columns={"test_source_name": "source_name"}))
    model_order = ["linear_regression", "ridge_regression"]
    width = 0.28
    x_positions = np.arange(len(source_order))
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    for offset_index, model_name in enumerate(model_order):
        model_data = (
            data.loc[data["model_name"] == model_name]
            .set_index("test_source_name")
            .reindex(source_order)
        )
        ax.bar(
            x_positions + (offset_index - 0.5) * width,
            model_data["rmse_reduction_vs_baseline_pct"],
            width=width,
            label=MODEL_LABELS[model_name],
            color=MODEL_COLORS[model_name],
        )
    ax.axhline(0.0, color="k", linestyle="--", linewidth=1)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(source_order, rotation=20, ha="right")
    ax.set_ylabel("RMSE Reduction vs Mean Baseline (%)")
    ax.set_title("Cross-Regime Skill vs Mean Baseline")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def save_cross_regime_pred_vs_true_plot(predictions: pd.DataFrame) -> Path:
    output_path = FIGURE_DIR / "cross_regime_pred_vs_true.png"
    data = predictions.loc[predictions["evaluation_mode"] == "cross_regime_holdout"].copy()
    source_order = ordered_sources(data.rename(columns={"test_source_name": "source_name"}))
    fig, axes = plt.subplots(1, len(source_order), figsize=(15, 4.8), sharex=False, sharey=False)
    if len(source_order) == 1:
        axes = [axes]
    for ax, source_name in zip(axes, source_order):
        source_data = data.loc[data["test_source_name"] == source_name]
        for model_name in MODEL_ORDER:
            group = source_data.loc[source_data["model_name"] == model_name]
            ax.scatter(
                group["y_true_k"],
                group["y_pred_k"],
                s=28,
                alpha=0.8,
                label=MODEL_LABELS[model_name],
                color=MODEL_COLORS[model_name],
            )
        diagonal_min = min(source_data["y_true_k"].min(), source_data["y_pred_k"].min())
        diagonal_max = max(source_data["y_true_k"].max(), source_data["y_pred_k"].max())
        ax.plot([diagonal_min, diagonal_max], [diagonal_min, diagonal_max], "k--", lw=1)
        ax.set_title(source_name)
        ax.set_xlabel("True Lower-Layer Mean Temperature (K)")
        ax.set_ylabel("Predicted Lower-Layer Mean Temperature (K)")
        ax.grid(True, alpha=0.3)
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def save_cross_regime_residuals_plot(predictions: pd.DataFrame) -> Path:
    output_path = FIGURE_DIR / "cross_regime_residuals.png"
    data = predictions.loc[predictions["evaluation_mode"] == "cross_regime_holdout"].copy()
    source_order = ordered_sources(data.rename(columns={"test_source_name": "source_name"}))
    fig, axes = plt.subplots(1, len(source_order), figsize=(15, 4.8), sharey=True)
    if len(source_order) == 1:
        axes = [axes]
    for ax, source_name in zip(axes, source_order):
        source_data = data.loc[data["test_source_name"] == source_name]
        for model_name in MODEL_ORDER:
            group = source_data.loc[source_data["model_name"] == model_name]
            ax.scatter(
                group["y_true_k"],
                group["residual_k"],
                s=28,
                alpha=0.8,
                label=MODEL_LABELS[model_name],
                color=MODEL_COLORS[model_name],
            )
        ax.axhline(0.0, color="k", linestyle="--", linewidth=1)
        ax.set_title(source_name)
        ax.set_xlabel("True Lower-Layer Mean Temperature (K)")
        ax.set_ylabel("Residual (Prediction - Truth) (K)")
        ax.grid(True, alpha=0.3)
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def print_summary_table(metrics: pd.DataFrame) -> None:
    print("Cross-regime retrieval summary:")
    print("Mode | Test source | Model | RMSE | MAE | Bias | Target std | NRMSE | Skill")
    for row in metrics.itertuples(index=False):
        print(
            f"{row.evaluation_mode} | {row.test_source_name} | {row.model_name} | "
            f"{row.rmse_k:.3f} K | {row.mae_k:.3f} K | {row.bias_k:+.3f} K | "
            f"{row.target_std_k:.3f} K | {row.normalized_rmse:.3f} | "
            f"{row.rmse_reduction_vs_baseline_pct:.1f}%"
        )


def main() -> None:
    observations = pd.read_csv(INPUT_PATH)
    observations = add_degraded_observations(observations)
    retrieval_dataset = pivot_observations(observations)
    era5_dataset = retrieval_dataset.loc[
        ~retrieval_dataset["region_name"].map(lambda value: str(value).startswith("synthetic_"))
    ].copy()

    in_distribution_metrics, _ = evaluate_in_distribution_by_source(era5_dataset)
    cross_regime_metrics, cross_regime_predictions = evaluate_cross_regime(era5_dataset)
    metrics = pd.concat([in_distribution_metrics, cross_regime_metrics], ignore_index=True)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(OUTPUT_PATH, index=False)

    output_paths = [
        save_cross_regime_rmse_plot(metrics),
        save_cross_regime_normalized_rmse_plot(metrics),
        save_cross_regime_skill_plot(metrics),
        save_cross_regime_pred_vs_true_plot(cross_regime_predictions),
        save_cross_regime_residuals_plot(cross_regime_predictions),
        OUTPUT_PATH,
    ]

    print_summary_table(metrics)
    for output_path in output_paths:
        print(output_path)


if __name__ == "__main__":
    main()
