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
    feature_sets,
    is_synthetic_region,
    pivot_observations,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "reports" / "figures" / "experiments"
RESIDUAL_OUTPUT_PATH = (
    PROJECT_ROOT / "data" / "processed" / "experiments" / "retrieval_residuals.csv"
)
METRICS_BY_SOURCE_OUTPUT_PATH = (
    PROJECT_ROOT / "data" / "processed" / "experiments" / "retrieval_metrics_by_source.csv"
)
SOURCE_PROFILE_SUMMARY_OUTPUT_PATH = (
    PROJECT_ROOT / "data" / "processed" / "experiments" / "source_profile_summary.csv"
)
NORMALIZED_METRICS_BY_SOURCE_OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "experiments"
    / "retrieval_metrics_normalized_by_source.csv"
)
PRIMARY_NOISE_CASE = "baseline_nedt"
PRIMARY_FEATURE_CASE = "full_3channel"
PRIMARY_MODELS = ["linear_regression", "ridge_regression"]
PRIMARY_ERA5_SOURCES = [
    "winter_midlatitude_maritime_sample",
    "lower_latitude_maritime_2020",
]
SOURCE_STYLE_MAP = {
    "winter_midlatitude_maritime_sample": {
        "label": "Winter midlatitude maritime",
        "color": "tab:blue",
        "marker": "o",
    },
    "lower_latitude_maritime_2020": {
        "label": "Lower-latitude maritime",
        "color": "tab:orange",
        "marker": "s",
    },
    "synthetic_profiles": {
        "label": "Synthetic OOD",
        "color": "tab:red",
        "marker": "^",
    },
}


def fit_predict_linear_model(
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


def build_residual_dataset(dataset: pd.DataFrame) -> pd.DataFrame:
    era5_dataset = dataset.loc[~dataset["region_name"].map(is_synthetic_region)].copy()
    synthetic_dataset = dataset.loc[dataset["region_name"].map(is_synthetic_region)].copy()

    candidate_metadata_columns = [
        "profile_index",
        "profile_id",
        "valid_time",
        "latitude",
        "longitude",
        "region_name",
        "source_name",
        "climate_regime",
        "surface_type",
        "season_label",
        "sample_group",
        "surface_temperature_k",
        "min_temperature_k",
        "max_temperature_k",
        "mean_temperature_k",
        "lower_layer_mean_temperature_k",
        "upper_layer_mean_temperature_k",
        "mean_h2o_vmr",
        "lower_layer_mean_h2o_vmr",
        "upper_layer_mean_h2o_vmr",
        "min_h2o_vmr",
        "max_h2o_vmr",
    ]
    metadata_columns = [
        column for column in candidate_metadata_columns if column in dataset.columns
    ]

    rows: list[dict[str, object]] = []
    for noise_case, noise_prefix in (
        ("baseline_nedt", "tb_obs"),
        ("degraded_nedt", "tb_obs_degraded"),
    ):
        for feature_case, feature_columns in feature_sets(noise_prefix).items():
            x_era5 = era5_dataset[feature_columns].to_numpy(dtype=float)
            y_era5 = era5_dataset[PRIMARY_TARGET].to_numpy(dtype=float)
            loo = LeaveOneOut()
            for model_name in ("mean_baseline", "linear_regression", "ridge_regression"):
                era5_predictions = np.zeros(len(era5_dataset), dtype=float)
                for train_index, test_index in loo.split(x_era5):
                    era5_predictions[test_index] = fit_predict_linear_model(
                        model_name,
                        x_era5[train_index],
                        y_era5[train_index],
                        x_era5[test_index],
                    )

                for row_index, prediction in enumerate(era5_predictions):
                    row = {
                        column: era5_dataset.iloc[row_index][column]
                        for column in metadata_columns
                    }
                    row.update(
                        {
                            "evaluation_split": "era5_leave_one_profile_out",
                            "profile_class": "ERA5",
                            "noise_case": noise_case,
                            "feature_case": feature_case,
                            "model_name": model_name,
                            "target_name": PRIMARY_TARGET,
                            "y_true_k": float(y_era5[row_index]),
                            "y_pred_k": float(prediction),
                            "residual_k": float(prediction - y_era5[row_index]),
                        }
                    )
                    for feature_name in feature_columns:
                        row[feature_name] = float(
                            era5_dataset.iloc[row_index][feature_name]
                        )
                    rows.append(row)

                x_synth = synthetic_dataset[feature_columns].to_numpy(dtype=float)
                y_synth = synthetic_dataset[PRIMARY_TARGET].to_numpy(dtype=float)
                synthetic_predictions = fit_predict_linear_model(
                    model_name, x_era5, y_era5, x_synth
                )
                for row_index, prediction in enumerate(synthetic_predictions):
                    row = {
                        column: synthetic_dataset.iloc[row_index][column]
                        for column in metadata_columns
                    }
                    row.update(
                        {
                            "evaluation_split": "synthetic_stress_test",
                            "profile_class": "Synthetic",
                            "noise_case": noise_case,
                            "feature_case": feature_case,
                            "model_name": model_name,
                            "target_name": PRIMARY_TARGET,
                            "y_true_k": float(y_synth[row_index]),
                            "y_pred_k": float(prediction),
                            "residual_k": float(prediction - y_synth[row_index]),
                        }
                    )
                    for feature_name in feature_columns:
                        row[feature_name] = float(
                            synthetic_dataset.iloc[row_index][feature_name]
                        )
                    rows.append(row)

    residuals = pd.DataFrame(rows)
    residuals.to_csv(RESIDUAL_OUTPUT_PATH, index=False)
    return residuals


def filter_primary_case(residuals: pd.DataFrame, model_name: str) -> pd.DataFrame:
    return residuals.loc[
        (residuals["noise_case"] == PRIMARY_NOISE_CASE)
        & (residuals["feature_case"] == PRIMARY_FEATURE_CASE)
        & (residuals["model_name"] == model_name)
    ].copy()


def source_style(source_name: str) -> dict[str, object]:
    return SOURCE_STYLE_MAP.get(
        source_name,
        {"label": source_name, "color": "0.45", "marker": "o"},
    )


def ordered_sources(data: pd.DataFrame) -> list[str]:
    present_sources = list(pd.unique(data["source_name"]))
    ordered = [
        source_name
        for source_name in PRIMARY_ERA5_SOURCES + ["synthetic_profiles"]
        if source_name in present_sources
    ]
    ordered.extend(
        source_name for source_name in present_sources if source_name not in ordered
    )
    return ordered


def rmse(values: pd.Series) -> float:
    return float(np.sqrt(np.mean(np.square(values.to_numpy(dtype=float)))))


def safe_std(values: pd.Series) -> float:
    return float(np.std(values.to_numpy(dtype=float), ddof=0))


def summarize_source_profiles(dataset: pd.DataFrame) -> pd.DataFrame:
    channel_columns = ["tb_obs_50_3", "tb_obs_52_8", "tb_obs_54_4"]
    rows: list[dict[str, object]] = []
    for source_name, group in dataset.groupby("source_name"):
        row: dict[str, object] = {
            "source_name": source_name,
            "profile_count": int(len(group)),
        }
        for prefix, column in (
            ("lower_layer_temperature", "lower_layer_mean_temperature_k"),
            ("upper_layer_temperature", "upper_layer_mean_temperature_k"),
            ("lower_layer_h2o_vmr", "lower_layer_mean_h2o_vmr"),
        ):
            values = group[column]
            row[f"{prefix}_mean"] = float(values.mean())
            row[f"{prefix}_std"] = safe_std(values)
            row[f"{prefix}_min"] = float(values.min())
            row[f"{prefix}_max"] = float(values.max())
            row[f"{prefix}_range"] = float(values.max() - values.min())
        for channel_column in channel_columns:
            channel_label = channel_column.replace("tb_obs_", "tb_")
            values = group[channel_column]
            row[f"{channel_label}_mean"] = float(values.mean())
            row[f"{channel_label}_std"] = safe_std(values)
            row[f"{channel_label}_min"] = float(values.min())
            row[f"{channel_label}_max"] = float(values.max())
            row[f"{channel_label}_range"] = float(values.max() - values.min())
        rows.append(row)

    summary = pd.DataFrame(rows).sort_values("source_name")
    summary.to_csv(SOURCE_PROFILE_SUMMARY_OUTPUT_PATH, index=False)
    return summary


def summarize_metrics_by_source(residuals: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        residuals.groupby(
            [
                "evaluation_split",
                "noise_case",
                "feature_case",
                "model_name",
                "source_name",
            ]
        )
        .agg(
            count=("residual_k", "size"),
            mae_k=("residual_k", lambda values: float(np.mean(np.abs(values)))),
            bias_k=("residual_k", "mean"),
            rmse_k=("residual_k", rmse),
        )
        .reset_index()
        .sort_values(
            ["evaluation_split", "noise_case", "feature_case", "model_name", "source_name"]
        )
    )
    grouped.to_csv(METRICS_BY_SOURCE_OUTPUT_PATH, index=False)
    return grouped


def summarize_normalized_metrics_by_source(
    metrics_by_source: pd.DataFrame,
    source_profile_summary: pd.DataFrame,
) -> pd.DataFrame:
    baseline = metrics_by_source.loc[
        metrics_by_source["model_name"] == "mean_baseline",
        [
            "evaluation_split",
            "noise_case",
            "feature_case",
            "source_name",
            "rmse_k",
        ],
    ].rename(columns={"rmse_k": "mean_baseline_rmse_k"})
    normalized = metrics_by_source.merge(
        source_profile_summary[
            ["source_name", "lower_layer_temperature_std", "lower_layer_temperature_range"]
        ],
        on="source_name",
        how="left",
    ).merge(
        baseline,
        on=["evaluation_split", "noise_case", "feature_case", "source_name"],
        how="left",
    )
    normalized["target_std_k"] = normalized["lower_layer_temperature_std"]
    normalized["target_range_k"] = normalized["lower_layer_temperature_range"]
    normalized["normalized_rmse"] = (
        normalized["rmse_k"] / normalized["target_std_k"]
    )
    normalized["rmse_reduction_vs_baseline_pct"] = 100.0 * (
        1.0 - normalized["rmse_k"] / normalized["mean_baseline_rmse_k"]
    )
    normalized.to_csv(NORMALIZED_METRICS_BY_SOURCE_OUTPUT_PATH, index=False)
    return normalized


def scatter_by_profile_class(
    ax: plt.Axes, data: pd.DataFrame, x_column: str, y_column: str
) -> None:
    for profile_class, color, marker in (
        ("ERA5", "0.45", "o"),
        ("Synthetic", "tab:red", "^"),
    ):
        group = data.loc[data["profile_class"] == profile_class]
        ax.scatter(
            group[x_column],
            group[y_column],
            c=color,
            marker=marker,
            s=40,
            alpha=0.85,
            label=profile_class,
        )


def scatter_by_source(
    ax: plt.Axes, data: pd.DataFrame, x_column: str, y_column: str
) -> None:
    for source_name in ordered_sources(data):
        group = data.loc[data["source_name"] == source_name]
        style = source_style(source_name)
        ax.scatter(
            group[x_column],
            group[y_column],
            c=style["color"],
            marker=style["marker"],
            s=42,
            alpha=0.8,
            label=style["label"],
        )


def save_rmse_by_source_plot(metrics_by_source: pd.DataFrame) -> Path:
    output_path = OUTPUT_DIR / "retrieval_rmse_by_source.png"
    data = metrics_by_source.loc[
        (metrics_by_source["noise_case"] == PRIMARY_NOISE_CASE)
        & (metrics_by_source["feature_case"] == PRIMARY_FEATURE_CASE)
    ].copy()
    source_order = ordered_sources(data)
    model_order = ["mean_baseline", "linear_regression", "ridge_regression"]
    model_labels = {
        "mean_baseline": "Mean baseline",
        "linear_regression": "Linear regression",
        "ridge_regression": "Ridge regression",
    }
    width = 0.22
    x_positions = np.arange(len(source_order))
    fig, ax = plt.subplots(figsize=(9, 4.8))
    for offset_index, model_name in enumerate(model_order):
        model_data = (
            data.loc[data["model_name"] == model_name]
            .set_index("source_name")
            .reindex(source_order)
        )
        ax.bar(
            x_positions + (offset_index - 1) * width,
            model_data["rmse_k"],
            width=width,
            label=model_labels[model_name],
        )
    ax.set_xticks(x_positions)
    ax.set_xticklabels(
        [source_style(source_name)["label"] for source_name in source_order],
        rotation=20,
        ha="right",
    )
    ax.set_ylabel("RMSE (K)")
    ax.set_title("Retrieval RMSE by Source")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def save_target_variability_by_source_plot(source_profile_summary: pd.DataFrame) -> Path:
    output_path = OUTPUT_DIR / "target_variability_by_source.png"
    source_order = ordered_sources(source_profile_summary)
    data = source_profile_summary.set_index("source_name").reindex(source_order)
    x_positions = np.arange(len(source_order))
    colors = [source_style(source_name)["color"] for source_name in source_order]
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.bar(
        x_positions,
        data["lower_layer_temperature_std"],
        color=colors,
        alpha=0.8,
        label="Std dev",
    )
    ax.errorbar(
        x_positions,
        data["lower_layer_temperature_mean"],
        yerr=0.5 * data["lower_layer_temperature_range"],
        fmt="k_",
        linewidth=1.4,
        capsize=4,
        label="Mean ± 0.5 range",
    )
    ax.set_xticks(x_positions)
    ax.set_xticklabels(
        [source_style(source_name)["label"] for source_name in source_order],
        rotation=20,
        ha="right",
    )
    ax.set_ylabel("Lower-Layer Mean Temperature (K)")
    ax.set_title("Target Variability by Source")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def save_rmse_vs_target_spread_by_source_plot(
    normalized_metrics_by_source: pd.DataFrame,
) -> Path:
    output_path = OUTPUT_DIR / "retrieval_rmse_vs_target_spread_by_source.png"
    data = normalized_metrics_by_source.loc[
        (normalized_metrics_by_source["noise_case"] == PRIMARY_NOISE_CASE)
        & (normalized_metrics_by_source["feature_case"] == PRIMARY_FEATURE_CASE)
        & (
            normalized_metrics_by_source["model_name"].isin(
                ["mean_baseline", "linear_regression", "ridge_regression"]
            )
        )
    ].copy()
    model_labels = {
        "mean_baseline": "Mean baseline",
        "linear_regression": "Linear regression",
        "ridge_regression": "Ridge regression",
    }
    model_markers = {
        "mean_baseline": "D",
        "linear_regression": "o",
        "ridge_regression": "s",
    }
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    for row in data.itertuples(index=False):
        style = source_style(row.source_name)
        ax.scatter(
            row.target_std_k,
            row.rmse_k,
            c=style["color"],
            marker=model_markers[row.model_name],
            s=70,
            alpha=0.85,
        )
        ax.annotate(
            f"{style['label']} | {model_labels[row.model_name]}",
            (row.target_std_k, row.rmse_k),
            textcoords="offset points",
            xytext=(5, 4),
            fontsize=8,
        )
    ax.set_xlabel("Target Standard Deviation (K)")
    ax.set_ylabel("RMSE (K)")
    ax.set_title("Retrieval RMSE vs Target Spread by Source")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def save_normalized_rmse_by_source_plot(
    normalized_metrics_by_source: pd.DataFrame,
) -> Path:
    output_path = OUTPUT_DIR / "retrieval_normalized_rmse_by_source.png"
    data = normalized_metrics_by_source.loc[
        (normalized_metrics_by_source["noise_case"] == PRIMARY_NOISE_CASE)
        & (normalized_metrics_by_source["feature_case"] == PRIMARY_FEATURE_CASE)
    ].copy()
    source_order = ordered_sources(data)
    model_order = ["mean_baseline", "linear_regression", "ridge_regression"]
    model_labels = {
        "mean_baseline": "Mean baseline",
        "linear_regression": "Linear regression",
        "ridge_regression": "Ridge regression",
    }
    width = 0.22
    x_positions = np.arange(len(source_order))
    fig, ax = plt.subplots(figsize=(9, 4.8))
    for offset_index, model_name in enumerate(model_order):
        model_data = (
            data.loc[data["model_name"] == model_name]
            .set_index("source_name")
            .reindex(source_order)
        )
        ax.bar(
            x_positions + (offset_index - 1) * width,
            model_data["normalized_rmse"],
            width=width,
            label=model_labels[model_name],
        )
    ax.set_xticks(x_positions)
    ax.set_xticklabels(
        [source_style(source_name)["label"] for source_name in source_order],
        rotation=20,
        ha="right",
    )
    ax.set_ylabel("Normalized RMSE (RMSE / target std)")
    ax.set_title("Normalized Retrieval RMSE by Source")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def save_skill_vs_baseline_by_source_plot(
    normalized_metrics_by_source: pd.DataFrame,
) -> Path:
    output_path = OUTPUT_DIR / "retrieval_skill_vs_baseline_by_source.png"
    data = normalized_metrics_by_source.loc[
        (normalized_metrics_by_source["noise_case"] == PRIMARY_NOISE_CASE)
        & (normalized_metrics_by_source["feature_case"] == PRIMARY_FEATURE_CASE)
        & (
            normalized_metrics_by_source["model_name"].isin(
                ["linear_regression", "ridge_regression"]
            )
        )
    ].copy()
    source_order = ordered_sources(data)
    model_order = ["linear_regression", "ridge_regression"]
    model_labels = {
        "linear_regression": "Linear regression",
        "ridge_regression": "Ridge regression",
    }
    width = 0.28
    x_positions = np.arange(len(source_order))
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    for offset_index, model_name in enumerate(model_order):
        model_data = (
            data.loc[data["model_name"] == model_name]
            .set_index("source_name")
            .reindex(source_order)
        )
        ax.bar(
            x_positions + (offset_index - 0.5) * width,
            model_data["rmse_reduction_vs_baseline_pct"],
            width=width,
            label=model_labels[model_name],
        )
    ax.axhline(0.0, color="k", linestyle="--", linewidth=1)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(
        [source_style(source_name)["label"] for source_name in source_order],
        rotation=20,
        ha="right",
    )
    ax.set_ylabel("RMSE Reduction vs Mean Baseline (%)")
    ax.set_title("Retrieval Skill vs Mean Baseline by Source")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def save_predicted_vs_true_plot(residuals: pd.DataFrame) -> Path:
    output_path = OUTPUT_DIR / "retrieval_pred_vs_true.png"
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharex=True, sharey=True)
    for ax, model_name, title in zip(
        axes,
        PRIMARY_MODELS,
        ["Linear Regression", "Ridge Regression"],
    ):
        data = filter_primary_case(residuals, model_name)
        scatter_by_profile_class(ax, data, "y_true_k", "y_pred_k")
        diagonal_min = min(data["y_true_k"].min(), data["y_pred_k"].min())
        diagonal_max = max(data["y_true_k"].max(), data["y_pred_k"].max())
        ax.plot([diagonal_min, diagonal_max], [diagonal_min, diagonal_max], "k--", lw=1)
        ax.set_title(title)
        ax.set_xlabel("True Lower-Layer Mean Temperature (K)")
        ax.set_ylabel("Predicted Lower-Layer Mean Temperature (K)")
        ax.grid(True, alpha=0.3)
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def save_predicted_vs_true_by_source_plot(residuals: pd.DataFrame) -> Path:
    output_path = OUTPUT_DIR / "retrieval_pred_vs_true_by_source.png"
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharex=True, sharey=True)
    for ax, model_name, title in zip(
        axes,
        PRIMARY_MODELS,
        ["Linear Regression", "Ridge Regression"],
    ):
        data = filter_primary_case(residuals, model_name)
        scatter_by_source(ax, data, "y_true_k", "y_pred_k")
        diagonal_min = min(data["y_true_k"].min(), data["y_pred_k"].min())
        diagonal_max = max(data["y_true_k"].max(), data["y_pred_k"].max())
        ax.plot([diagonal_min, diagonal_max], [diagonal_min, diagonal_max], "k--", lw=1)
        ax.set_title(title)
        ax.set_xlabel("True Lower-Layer Mean Temperature (K)")
        ax.set_ylabel("Predicted Lower-Layer Mean Temperature (K)")
        ax.grid(True, alpha=0.3)
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def save_residual_vs_truth_plot(residuals: pd.DataFrame) -> Path:
    output_path = OUTPUT_DIR / "retrieval_residual_vs_truth.png"
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)
    for ax, model_name, title in zip(
        axes,
        PRIMARY_MODELS,
        ["Linear Regression", "Ridge Regression"],
    ):
        data = filter_primary_case(residuals, model_name)
        scatter_by_profile_class(ax, data, "y_true_k", "residual_k")
        ax.axhline(0.0, color="k", linestyle="--", linewidth=1)
        ax.set_title(title)
        ax.set_xlabel("True Lower-Layer Mean Temperature (K)")
        ax.set_ylabel("Residual (Prediction - Truth) (K)")
        ax.grid(True, alpha=0.3)
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def save_residual_vs_truth_by_source_plot(residuals: pd.DataFrame) -> Path:
    output_path = OUTPUT_DIR / "retrieval_residual_vs_truth_by_source.png"
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)
    for ax, model_name, title in zip(
        axes,
        PRIMARY_MODELS,
        ["Linear Regression", "Ridge Regression"],
    ):
        data = filter_primary_case(residuals, model_name)
        scatter_by_source(ax, data, "y_true_k", "residual_k")
        ax.axhline(0.0, color="k", linestyle="--", linewidth=1)
        ax.set_title(title)
        ax.set_xlabel("True Lower-Layer Mean Temperature (K)")
        ax.set_ylabel("Residual (Prediction - Truth) (K)")
        ax.grid(True, alpha=0.3)
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def save_residual_vs_humidity_plot(residuals: pd.DataFrame) -> Path:
    output_path = OUTPUT_DIR / "retrieval_residual_vs_humidity.png"
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)
    for ax, model_name, title in zip(
        axes,
        PRIMARY_MODELS,
        ["Linear Regression", "Ridge Regression"],
    ):
        data = filter_primary_case(residuals, model_name)
        scatter_by_profile_class(ax, data, "lower_layer_mean_h2o_vmr", "residual_k")
        ax.axhline(0.0, color="k", linestyle="--", linewidth=1)
        ax.set_xscale("log")
        ax.set_title(title)
        ax.set_xlabel("Lower-Layer Mean H2O VMR")
        ax.set_ylabel("Residual (Prediction - Truth) (K)")
        ax.grid(True, alpha=0.3)
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def save_residual_vs_humidity_by_source_plot(residuals: pd.DataFrame) -> Path:
    output_path = OUTPUT_DIR / "retrieval_residual_vs_humidity_by_source.png"
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)
    for ax, model_name, title in zip(
        axes,
        PRIMARY_MODELS,
        ["Linear Regression", "Ridge Regression"],
    ):
        data = filter_primary_case(residuals, model_name)
        scatter_by_source(ax, data, "lower_layer_mean_h2o_vmr", "residual_k")
        ax.axhline(0.0, color="k", linestyle="--", linewidth=1)
        ax.set_xscale("log")
        ax.set_title(title)
        ax.set_xlabel("Lower-Layer Mean H2O VMR")
        ax.set_ylabel("Residual (Prediction - Truth) (K)")
        ax.grid(True, alpha=0.3)
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def save_residual_vs_channels_plot(residuals: pd.DataFrame) -> Path:
    output_path = OUTPUT_DIR / "retrieval_residual_vs_channels.png"
    channel_columns = [
        ("tb_obs_50_3", "50.3 GHz Tb_obs (K)"),
        ("tb_obs_52_8", "52.8 GHz Tb_obs (K)"),
        ("tb_obs_54_4", "54.4 GHz Tb_obs (K)"),
    ]
    fig, axes = plt.subplots(3, 2, figsize=(10, 11), sharey=True)
    for column_index, (feature_name, xlabel) in enumerate(channel_columns):
        for model_index, (model_name, title) in enumerate(
            zip(PRIMARY_MODELS, ["Linear Regression", "Ridge Regression"])
        ):
            ax = axes[column_index, model_index]
            data = filter_primary_case(residuals, model_name)
            scatter_by_profile_class(ax, data, feature_name, "residual_k")
            ax.axhline(0.0, color="k", linestyle="--", linewidth=1)
            if column_index == 0:
                ax.set_title(title)
            ax.set_xlabel(xlabel)
            ax.set_ylabel("Residual (K)")
            ax.grid(True, alpha=0.3)
    axes[0, 0].legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def save_channel_importance_by_source_plot(metrics_by_source: pd.DataFrame) -> Path:
    output_path = OUTPUT_DIR / "retrieval_channel_importance_by_source.png"
    data = metrics_by_source.loc[
        (metrics_by_source["noise_case"] == PRIMARY_NOISE_CASE)
        & (metrics_by_source["model_name"] == "ridge_regression")
        & (
            metrics_by_source["feature_case"].isin(
                ["full_3channel", "drop_50_3", "drop_52_8", "drop_54_4"]
            )
        )
    ].copy()
    feature_order = ["full_3channel", "drop_50_3", "drop_52_8", "drop_54_4"]
    source_order = ordered_sources(data)
    width = 0.22
    x_positions = np.arange(len(feature_order))
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    for offset_index, source_name in enumerate(source_order):
        source_data = (
            data.loc[data["source_name"] == source_name]
            .set_index("feature_case")
            .reindex(feature_order)
        )
        style = source_style(source_name)
        ax.bar(
            x_positions + (offset_index - (len(source_order) - 1) / 2) * width,
            source_data["rmse_k"],
            width=width,
            label=style["label"],
            color=style["color"],
        )
    ax.set_xticks(x_positions)
    ax.set_xticklabels(feature_order, rotation=20, ha="right")
    ax.set_ylabel("RMSE (K)")
    ax.set_title("Channel-Configuration RMSE by Source (Ridge, Baseline NEΔT)")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def save_residual_distribution_by_source_plot(residuals: pd.DataFrame) -> Path:
    output_path = OUTPUT_DIR / "retrieval_residual_distribution_by_source.png"
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.8), sharey=True)
    for ax, model_name, title in zip(
        axes,
        PRIMARY_MODELS,
        ["Linear Regression", "Ridge Regression"],
    ):
        data = filter_primary_case(residuals, model_name)
        source_order = ordered_sources(data)
        box_data = [
            data.loc[data["source_name"] == source_name, "residual_k"]
            for source_name in source_order
        ]
        ax.boxplot(
            box_data,
            tick_labels=[source_style(source_name)["label"] for source_name in source_order],
        )
        ax.axhline(0.0, color="k", linestyle="--", linewidth=1)
        ax.set_title(title)
        ax.set_ylabel("Residual (K)")
        ax.grid(True, axis="y", alpha=0.3)
        ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def save_residual_by_group_plot(residuals: pd.DataFrame) -> Path:
    output_path = OUTPUT_DIR / "retrieval_residual_by_group.png"
    data = filter_primary_case(residuals, "ridge_regression")
    grouped = (
        data.groupby("region_name", as_index=False)
        .agg(
            rmse_k=("residual_k", rmse),
            bias_k=("residual_k", "mean"),
        )
        .sort_values("region_name")
    )
    colors = [
        "tab:red" if is_synthetic_region(region_name) else "0.55"
        for region_name in grouped["region_name"]
    ]
    x_positions = np.arange(len(grouped))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharex=True)
    axes[0].bar(x_positions, grouped["bias_k"], color=colors)
    axes[0].axhline(0.0, color="k", linestyle="--", linewidth=1)
    axes[0].set_title("Mean Bias by Region")
    axes[0].set_ylabel("Bias (K)")
    axes[0].grid(True, axis="y", alpha=0.3)

    axes[1].bar(x_positions, grouped["rmse_k"], color=colors)
    axes[1].set_title("RMSE by Region")
    axes[1].set_ylabel("RMSE (K)")
    axes[1].grid(True, axis="y", alpha=0.3)

    for ax in axes:
        ax.set_xticks(x_positions)
        ax.set_xticklabels(grouped["region_name"], rotation=35, ha="right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def save_residual_by_channel_config_plot(residuals: pd.DataFrame) -> Path:
    output_path = OUTPUT_DIR / "retrieval_residual_by_channel_config.png"
    data = residuals.loc[
        (residuals["noise_case"] == PRIMARY_NOISE_CASE)
        & (residuals["model_name"] == "ridge_regression")
    ].copy()
    feature_order = ["full_3channel", "drop_50_3", "drop_52_8", "drop_54_4"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True)
    for ax, split_name, title in zip(
        axes,
        ["era5_leave_one_profile_out", "synthetic_stress_test"],
        ["ERA5 Leave-One-Profile-Out", "Synthetic Stress Test"],
    ):
        split_data = data.loc[data["evaluation_split"] == split_name]
        box_data = [
            split_data.loc[split_data["feature_case"] == feature_case, "residual_k"]
            for feature_case in feature_order
        ]
        ax.boxplot(box_data, tick_labels=feature_order)
        ax.axhline(0.0, color="k", linestyle="--", linewidth=1)
        ax.set_title(title)
        ax.set_xlabel("Channel Configuration")
        ax.set_ylabel("Residual (K)")
        ax.grid(True, axis="y", alpha=0.3)
        ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def print_source_metric_summary(metrics_by_source: pd.DataFrame) -> None:
    data = metrics_by_source.loc[
        (metrics_by_source["noise_case"] == PRIMARY_NOISE_CASE)
        & (metrics_by_source["feature_case"] == PRIMARY_FEATURE_CASE)
        & (
            metrics_by_source["model_name"].isin(
                ["mean_baseline", "linear_regression", "ridge_regression"]
            )
        )
    ].copy()
    print("Retrieval metrics by source:")
    print("Split | Source | Model | RMSE | MAE | Bias")
    for row in data.itertuples(index=False):
        print(
            f"{row.evaluation_split} | {row.source_name} | {row.model_name} | "
            f"{row.rmse_k:.3f} K | {row.mae_k:.3f} K | {row.bias_k:+.3f} K"
        )


def print_variability_normalized_summary(
    source_profile_summary: pd.DataFrame,
    normalized_metrics_by_source: pd.DataFrame,
) -> None:
    print("Source profile variability summary:")
    print(
        "Source | Profiles | Lower-layer T std | Lower-layer T range | "
        "Upper-layer T std | Lower-layer H2O std"
    )
    for row in source_profile_summary.itertuples(index=False):
        print(
            f"{row.source_name} | {row.profile_count} | "
            f"{row.lower_layer_temperature_std:.3f} K | "
            f"{row.lower_layer_temperature_range:.3f} K | "
            f"{row.upper_layer_temperature_std:.3f} K | "
            f"{row.lower_layer_h2o_vmr_std:.3e}"
        )

    data = normalized_metrics_by_source.loc[
        (normalized_metrics_by_source["noise_case"] == PRIMARY_NOISE_CASE)
        & (normalized_metrics_by_source["feature_case"] == PRIMARY_FEATURE_CASE)
        & (
            normalized_metrics_by_source["model_name"].isin(
                ["mean_baseline", "linear_regression", "ridge_regression"]
            )
        )
    ].copy()
    print("Variability-normalized retrieval metrics by source:")
    print("Source | Model | RMSE | Target std | Normalized RMSE | Skill vs baseline")
    for row in data.itertuples(index=False):
        print(
            f"{row.source_name} | {row.model_name} | "
            f"{row.rmse_k:.3f} K | {row.target_std_k:.3f} K | "
            f"{row.normalized_rmse:.3f} | {row.rmse_reduction_vs_baseline_pct:.1f}%"
        )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    observations = pd.read_csv(INPUT_PATH)
    observations = add_degraded_observations(observations)
    retrieval_dataset = pivot_observations(observations)
    residuals = build_residual_dataset(retrieval_dataset)
    source_profile_summary = summarize_source_profiles(retrieval_dataset)
    metrics_by_source = summarize_metrics_by_source(residuals)
    normalized_metrics_by_source = summarize_normalized_metrics_by_source(
        metrics_by_source, source_profile_summary
    )
    print_source_metric_summary(metrics_by_source)
    print_variability_normalized_summary(
        source_profile_summary, normalized_metrics_by_source
    )

    output_paths = [
        save_rmse_by_source_plot(metrics_by_source),
        save_target_variability_by_source_plot(source_profile_summary),
        save_rmse_vs_target_spread_by_source_plot(normalized_metrics_by_source),
        save_normalized_rmse_by_source_plot(normalized_metrics_by_source),
        save_skill_vs_baseline_by_source_plot(normalized_metrics_by_source),
        save_predicted_vs_true_plot(residuals),
        save_predicted_vs_true_by_source_plot(residuals),
        save_residual_vs_truth_plot(residuals),
        save_residual_vs_truth_by_source_plot(residuals),
        save_residual_vs_humidity_plot(residuals),
        save_residual_vs_humidity_by_source_plot(residuals),
        save_residual_vs_channels_plot(residuals),
        save_channel_importance_by_source_plot(metrics_by_source),
        save_residual_distribution_by_source_plot(residuals),
        save_residual_by_group_plot(residuals),
        save_residual_by_channel_config_plot(residuals),
        RESIDUAL_OUTPUT_PATH,
        METRICS_BY_SOURCE_OUTPUT_PATH,
        SOURCE_PROFILE_SUMMARY_OUTPUT_PATH,
        NORMALIZED_METRICS_BY_SOURCE_OUTPUT_PATH,
    ]

    for output_path in output_paths:
        print(output_path)


if __name__ == "__main__":
    main()
