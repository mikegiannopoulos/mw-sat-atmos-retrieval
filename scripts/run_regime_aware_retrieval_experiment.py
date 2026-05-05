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
    PRIMARY_TARGET,
    add_degraded_observations,
    compute_metrics,
    is_synthetic_region,
    pivot_observations,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = (
    PROJECT_ROOT / "data" / "processed" / "experiments" / "regime_aware_retrieval_metrics.csv"
)
FIGURE_DIR = PROJECT_ROOT / "reports" / "figures" / "experiments"

FEATURE_COLUMNS = ["tb_obs_50_3", "tb_obs_52_8", "tb_obs_54_4"]
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
SETUP_ORDER = ["global", "source_name", "climate_regime", "surface_type"]
SETUP_LABELS = {
    "global": "Global",
    "source_name": "Source-specific",
    "climate_regime": "Climate-regime-aware",
    "surface_type": "Surface-type-aware",
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


def source_std(dataset: pd.DataFrame) -> float:
    return float(np.std(dataset[PRIMARY_TARGET].to_numpy(dtype=float), ddof=0))


def rmse(values: pd.Series) -> float:
    return float(np.sqrt(np.mean(np.square(values.to_numpy(dtype=float)))))


def setup_group_description(dataset: pd.DataFrame, grouping_column: str) -> str:
    distinct_sources = sorted(pd.unique(dataset["source_name"]))
    if grouping_column == "source_name":
        return "One model per source_name."
    if grouping_column == "climate_regime":
        if len(distinct_sources) == dataset[grouping_column].nunique():
            return "Currently equivalent to source_name because each source has a unique climate_regime."
        return "One model per climate_regime."
    if grouping_column == "surface_type":
        source_counts = dataset.groupby(grouping_column)["source_name"].nunique().to_dict()
        if any(count < 2 for count in source_counts.values()):
            return (
                "Surface-type grouping is partially limited because at least one surface type "
                "contains only one distinct source."
            )
        return "One model per surface_type."
    return "Single global model across all ERA5 sources."


def build_metric_row(
    setup_type: str,
    group_name: str,
    model_name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_std_k: float,
    baseline_rmse_k: float,
    n_samples: int,
    notes: str,
) -> dict[str, object]:
    metrics = compute_metrics(y_true, y_pred)
    return {
        "setup_type": setup_type,
        "group_name": group_name,
        "model_name": model_name,
        "target_name": PRIMARY_TARGET,
        "n_samples": n_samples,
        **metrics,
        "target_std_k": target_std_k,
        "normalized_rmse": metrics["rmse_k"] / target_std_k,
        "rmse_reduction_vs_baseline_pct": 100.0
        * (1.0 - metrics["rmse_k"] / baseline_rmse_k),
        "notes": notes,
    }


def evaluate_global_model(
    era5_dataset: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    x = era5_dataset[FEATURE_COLUMNS].to_numpy(dtype=float)
    y = era5_dataset[PRIMARY_TARGET].to_numpy(dtype=float)
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

    baseline_rmse_k = compute_metrics(y, predictions["mean_baseline"])["rmse_k"]
    target_std_k = source_std(era5_dataset)
    metric_rows = [
        build_metric_row(
            setup_type="global",
            group_name="all_era5",
            model_name=model_name,
            y_true=y,
            y_pred=predictions[model_name],
            target_std_k=target_std_k,
            baseline_rmse_k=baseline_rmse_k,
            n_samples=len(era5_dataset),
            notes=setup_group_description(era5_dataset, "global"),
        )
        for model_name in MODEL_ORDER
    ]

    prediction_rows: list[dict[str, object]] = []
    for model_name in MODEL_ORDER:
        for row_index, prediction in enumerate(predictions[model_name]):
            row = era5_dataset.iloc[row_index]
            prediction_rows.append(
                {
                    "setup_type": "global",
                    "group_name": "all_era5",
                    "model_name": model_name,
                    "profile_id": row["profile_id"],
                    "source_name": row["source_name"],
                    "climate_regime": row["climate_regime"],
                    "surface_type": row["surface_type"],
                    "valid_time": row["valid_time"],
                    "y_true_k": float(y[row_index]),
                    "y_pred_k": float(prediction),
                    "residual_k": float(prediction - y[row_index]),
                }
            )

    return pd.DataFrame(metric_rows), pd.DataFrame(prediction_rows)


def evaluate_grouped_models(
    era5_dataset: pd.DataFrame,
    grouping_column: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []
    notes = setup_group_description(era5_dataset, grouping_column)

    pooled_predictions = {model_name: [] for model_name in MODEL_ORDER}
    pooled_truth = []
    pooled_rows = []

    for group_name, group in era5_dataset.groupby(grouping_column, sort=True):
        x = group[FEATURE_COLUMNS].to_numpy(dtype=float)
        y = group[PRIMARY_TARGET].to_numpy(dtype=float)
        group_predictions = {model_name: np.zeros(len(group), dtype=float) for model_name in MODEL_ORDER}

        for test_index in range(len(group)):
            train_mask = np.ones(len(group), dtype=bool)
            train_mask[test_index] = False
            x_train = x[train_mask]
            y_train = y[train_mask]
            x_test = x[[test_index]]
            for model_name in MODEL_ORDER:
                group_predictions[model_name][test_index] = fit_model_predict(
                    model_name, x_train, y_train, x_test
                )[0]

        baseline_rmse_k = compute_metrics(y, group_predictions["mean_baseline"])["rmse_k"]
        target_std_k = source_std(group)
        for model_name in MODEL_ORDER:
            metric_rows.append(
                build_metric_row(
                    setup_type=grouping_column,
                    group_name=str(group_name),
                    model_name=model_name,
                    y_true=y,
                    y_pred=group_predictions[model_name],
                    target_std_k=target_std_k,
                    baseline_rmse_k=baseline_rmse_k,
                    n_samples=len(group),
                    notes=notes,
                )
            )
            pooled_predictions[model_name].extend(group_predictions[model_name].tolist())
        pooled_truth.extend(y.tolist())
        pooled_rows.extend(group.index.tolist())

        for model_name in MODEL_ORDER:
            for row_index, prediction in enumerate(group_predictions[model_name]):
                row = group.iloc[row_index]
                prediction_rows.append(
                    {
                        "setup_type": grouping_column,
                        "group_name": str(group_name),
                        "model_name": model_name,
                        "profile_id": row["profile_id"],
                        "source_name": row["source_name"],
                        "climate_regime": row["climate_regime"],
                        "surface_type": row["surface_type"],
                        "valid_time": row["valid_time"],
                        "y_true_k": float(y[row_index]),
                        "y_pred_k": float(prediction),
                        "residual_k": float(prediction - y[row_index]),
                    }
                )

    pooled_truth_array = np.asarray(pooled_truth, dtype=float)
    pooled_target_std_k = source_std(era5_dataset)
    pooled_baseline_rmse_k = compute_metrics(
        pooled_truth_array,
        np.asarray(pooled_predictions["mean_baseline"], dtype=float),
    )["rmse_k"]
    for model_name in MODEL_ORDER:
        metric_rows.append(
            build_metric_row(
                setup_type=grouping_column,
                group_name="all_groups_pooled",
                model_name=model_name,
                y_true=pooled_truth_array,
                y_pred=np.asarray(pooled_predictions[model_name], dtype=float),
                target_std_k=pooled_target_std_k,
                baseline_rmse_k=pooled_baseline_rmse_k,
                n_samples=len(era5_dataset),
                notes=notes,
            )
        )

    return pd.DataFrame(metric_rows), pd.DataFrame(prediction_rows)


def plot_metric_by_setup(
    metrics: pd.DataFrame,
    metric_column: str,
    output_name: str,
    ylabel: str,
    title: str,
    model_subset: list[str] | None = None,
) -> Path:
    output_path = FIGURE_DIR / output_name
    data = metrics.loc[metrics["group_name"] == "all_groups_pooled"].copy()
    data = pd.concat(
        [
            metrics.loc[
                (metrics["setup_type"] == "global") & (metrics["group_name"] == "all_era5")
            ],
            data,
        ],
        ignore_index=True,
    )
    if model_subset is not None:
        data = data.loc[data["model_name"].isin(model_subset)].copy()
        model_order = model_subset
    else:
        model_order = MODEL_ORDER
    setup_order = [setup for setup in SETUP_ORDER if setup in set(data["setup_type"])]
    width = 0.22 if len(model_order) == 3 else 0.28
    x_positions = np.arange(len(setup_order))
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    center_offset = (len(model_order) - 1) / 2
    for offset_index, model_name in enumerate(model_order):
        model_data = (
            data.loc[data["model_name"] == model_name]
            .set_index("setup_type")
            .reindex(setup_order)
        )
        ax.bar(
            x_positions + (offset_index - center_offset) * width,
            model_data[metric_column],
            width=width,
            label=MODEL_LABELS[model_name],
            color=MODEL_COLORS[model_name],
        )
    ax.set_xticks(x_positions)
    ax.set_xticklabels([SETUP_LABELS[setup] for setup in setup_order], rotation=15, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def save_bias_by_group_plot(metrics: pd.DataFrame) -> Path:
    output_path = FIGURE_DIR / "regime_aware_bias_by_group.png"
    data = metrics.loc[
        (metrics["group_name"] != "all_groups_pooled")
        & (metrics["group_name"] != "all_era5")
        & (metrics["model_name"].isin(["linear_regression", "ridge_regression"]))
    ].copy()
    data["label"] = data["setup_type"] + ":" + data["group_name"]
    x_positions = np.arange(len(data["label"].unique()))
    labels = list(dict.fromkeys(data["label"]))
    width = 0.35
    fig, ax = plt.subplots(figsize=(12.5, 5.2))
    for offset_index, model_name in enumerate(["linear_regression", "ridge_regression"]):
        model_data = (
            data.loc[data["model_name"] == model_name]
            .set_index("label")
            .reindex(labels)
        )
        ax.bar(
            x_positions + (offset_index - 0.5) * width,
            model_data["bias_k"],
            width=width,
            label=MODEL_LABELS[model_name],
            color=MODEL_COLORS[model_name],
        )
    ax.axhline(0.0, color="k", linestyle="--", linewidth=1)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylabel("Bias (K)")
    ax.set_title("Regime-Aware Retrieval Bias by Group")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def save_pred_vs_true_plot(predictions: pd.DataFrame) -> Path:
    output_path = FIGURE_DIR / "regime_aware_pred_vs_true.png"
    ridge_predictions = predictions.loc[predictions["model_name"] == "ridge_regression"].copy()
    setup_order = [setup for setup in SETUP_ORDER if setup in set(ridge_predictions["setup_type"])]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5), sharex=False, sharey=False)
    axes = axes.flatten()
    for ax, setup_type in zip(axes, setup_order):
        setup_data = ridge_predictions.loc[ridge_predictions["setup_type"] == setup_type]
        for source_name, source_group in setup_data.groupby("source_name", sort=True):
            ax.scatter(
                source_group["y_true_k"],
                source_group["y_pred_k"],
                s=26,
                alpha=0.8,
                label=source_name,
            )
        diagonal_min = min(setup_data["y_true_k"].min(), setup_data["y_pred_k"].min())
        diagonal_max = max(setup_data["y_true_k"].max(), setup_data["y_pred_k"].max())
        ax.plot([diagonal_min, diagonal_max], [diagonal_min, diagonal_max], "k--", lw=1)
        ax.set_title(f"{SETUP_LABELS[setup_type]} (ridge)")
        ax.set_xlabel("True Lower-Layer Mean Temperature (K)")
        ax.set_ylabel("Predicted Lower-Layer Mean Temperature (K)")
        ax.grid(True, alpha=0.3)
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def save_residuals_plot(predictions: pd.DataFrame) -> Path:
    output_path = FIGURE_DIR / "regime_aware_residuals.png"
    ridge_predictions = predictions.loc[predictions["model_name"] == "ridge_regression"].copy()
    setup_order = [setup for setup in SETUP_ORDER if setup in set(ridge_predictions["setup_type"])]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5), sharey=True)
    axes = axes.flatten()
    for ax, setup_type in zip(axes, setup_order):
        setup_data = ridge_predictions.loc[ridge_predictions["setup_type"] == setup_type]
        for source_name, source_group in setup_data.groupby("source_name", sort=True):
            ax.scatter(
                source_group["y_true_k"],
                source_group["residual_k"],
                s=26,
                alpha=0.8,
                label=source_name,
            )
        ax.axhline(0.0, color="k", linestyle="--", linewidth=1)
        ax.set_title(f"{SETUP_LABELS[setup_type]} (ridge)")
        ax.set_xlabel("True Lower-Layer Mean Temperature (K)")
        ax.set_ylabel("Residual (Prediction - Truth) (K)")
        ax.grid(True, alpha=0.3)
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def print_summary(metrics: pd.DataFrame) -> None:
    print("Regime-aware retrieval summary:")
    print("Setup | Group | Model | RMSE | MAE | Bias | Target std | NRMSE | Skill | Notes")
    for row in metrics.itertuples(index=False):
        print(
            f"{row.setup_type} | {row.group_name} | {row.model_name} | "
            f"{row.rmse_k:.3f} K | {row.mae_k:.3f} K | {row.bias_k:+.3f} K | "
            f"{row.target_std_k:.3f} K | {row.normalized_rmse:.3f} | "
            f"{row.rmse_reduction_vs_baseline_pct:.1f}% | {row.notes}"
        )


def main() -> None:
    observations = pd.read_csv(INPUT_PATH)
    observations = add_degraded_observations(observations)
    retrieval_dataset = pivot_observations(observations)
    era5_dataset = retrieval_dataset.loc[
        ~retrieval_dataset["region_name"].map(is_synthetic_region)
    ].copy()

    metrics_frames = []
    prediction_frames = []

    global_metrics, global_predictions = evaluate_global_model(era5_dataset)
    metrics_frames.append(global_metrics)
    prediction_frames.append(global_predictions)

    for grouping_column in ("source_name", "climate_regime", "surface_type"):
        grouped_metrics, grouped_predictions = evaluate_grouped_models(
            era5_dataset, grouping_column
        )
        metrics_frames.append(grouped_metrics)
        prediction_frames.append(grouped_predictions)

    metrics = pd.concat(metrics_frames, ignore_index=True)
    predictions = pd.concat(prediction_frames, ignore_index=True)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(OUTPUT_PATH, index=False)

    output_paths = [
        plot_metric_by_setup(
            metrics,
            "rmse_k",
            "regime_aware_rmse.png",
            "RMSE (K)",
            "Regime-Aware Retrieval RMSE",
        ),
        plot_metric_by_setup(
            metrics,
            "normalized_rmse",
            "regime_aware_normalized_rmse.png",
            "Normalized RMSE",
            "Regime-Aware Normalized RMSE",
        ),
        plot_metric_by_setup(
            metrics,
            "rmse_reduction_vs_baseline_pct",
            "regime_aware_skill_vs_baseline.png",
            "RMSE Reduction vs Mean Baseline (%)",
            "Regime-Aware Skill vs Mean Baseline",
            model_subset=["linear_regression", "ridge_regression"],
        ),
        save_bias_by_group_plot(metrics),
        save_pred_vs_true_plot(predictions),
        save_residuals_plot(predictions),
        OUTPUT_PATH,
    ]

    print_summary(metrics)
    for output_path in output_paths:
        print(output_path)


if __name__ == "__main__":
    main()
