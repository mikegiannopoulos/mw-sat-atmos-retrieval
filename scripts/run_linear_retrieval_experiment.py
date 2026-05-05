from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import LeaveOneOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = (
    PROJECT_ROOT / "data" / "processed" / "experiments" / "multi_profile_observations.csv"
)
OUTPUT_PATH = (
    PROJECT_ROOT / "data" / "processed" / "experiments" / "linear_retrieval_results.csv"
)
MULTI_PROFILE_NOISE_BASE_SEED = 100
CHANNEL_FREQUENCIES_GHZ = [50.3, 52.8, 54.4]
PRIMARY_TARGET = "lower_layer_mean_temperature_k"


def frequency_feature_name(frequency_ghz: float, prefix: str) -> str:
    frequency_label = str(frequency_ghz).replace(".", "_")
    return f"{prefix}_{frequency_label}"


def is_synthetic_region(region_name: str) -> bool:
    return str(region_name).startswith("synthetic_")


def add_degraded_observations(observations: pd.DataFrame) -> pd.DataFrame:
    observations = observations.copy()
    degraded_values = pd.Series(index=observations.index, dtype=float)
    for profile_index, group in observations.groupby("profile_index", sort=False):
        group = group.sort_values("frequency_ghz")
        rng = np.random.default_rng(MULTI_PROFILE_NOISE_BASE_SEED + int(profile_index))
        degraded_noise = rng.normal(
            loc=0.0,
            scale=(2.0 * group["nedt_k"]).to_numpy(),
            size=len(group),
        )
        degraded_tb_obs = group["tb_true_k"].to_numpy() + degraded_noise
        degraded_values.loc[group.index] = degraded_tb_obs
    observations["tb_obs_degraded_k"] = degraded_values
    return observations


def pivot_observations(observations: pd.DataFrame) -> pd.DataFrame:
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
        column for column in candidate_metadata_columns if column in observations.columns
    ]
    pivot_rows: list[dict[str, object]] = []
    for profile_id, group in observations.groupby("profile_id", sort=True):
        group = group.sort_values("frequency_ghz")
        row = {column: group[column].iloc[0] for column in metadata_columns}
        for channel_row in group.itertuples(index=False):
            row[frequency_feature_name(channel_row.frequency_ghz, "tb_obs")] = (
                channel_row.tb_obs_k
            )
            row[frequency_feature_name(channel_row.frequency_ghz, "tb_obs_degraded")] = (
                channel_row.tb_obs_degraded_k
            )
        pivot_rows.append(row)
    return pd.DataFrame(pivot_rows).sort_values("profile_index").reset_index(drop=True)


def regression_models() -> dict[str, object]:
    return {
        "mean_baseline": None,
        "linear_regression": LinearRegression(),
        "ridge_regression": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
    }


def feature_sets(noise_prefix: str) -> dict[str, list[str]]:
    features = [
        frequency_feature_name(frequency_ghz, noise_prefix)
        for frequency_ghz in CHANNEL_FREQUENCIES_GHZ
    ]
    return {
        "full_3channel": features,
        "drop_50_3": [
            feature_name
            for feature_name in features
            if not feature_name.endswith("50_3")
        ],
        "drop_52_8": [
            feature_name
            for feature_name in features
            if not feature_name.endswith("52_8")
        ],
        "drop_54_4": [
            feature_name
            for feature_name in features
            if not feature_name.endswith("54_4")
        ],
    }


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "rmse_k": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae_k": float(mean_absolute_error(y_true, y_pred)),
        "bias_k": float(np.mean(y_pred - y_true)),
    }


def run_leave_one_profile_out(
    dataset: pd.DataFrame, feature_columns: list[str], target_column: str
) -> dict[str, dict[str, float]]:
    x = dataset[feature_columns].to_numpy(dtype=float)
    y = dataset[target_column].to_numpy(dtype=float)
    loo = LeaveOneOut()
    predictions = {model_name: np.zeros(len(dataset), dtype=float) for model_name in regression_models()}

    for train_index, test_index in loo.split(x):
        x_train, x_test = x[train_index], x[test_index]
        y_train = y[train_index]

        predictions["mean_baseline"][test_index] = y_train.mean()

        linear_model = LinearRegression()
        linear_model.fit(x_train, y_train)
        predictions["linear_regression"][test_index] = linear_model.predict(x_test)

        ridge_model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
        ridge_model.fit(x_train, y_train)
        predictions["ridge_regression"][test_index] = ridge_model.predict(x_test)

    return {
        model_name: compute_metrics(y, model_predictions)
        for model_name, model_predictions in predictions.items()
    }


def run_synthetic_stress_test(
    train_dataset: pd.DataFrame,
    test_dataset: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
) -> dict[str, dict[str, float]]:
    x_train = train_dataset[feature_columns].to_numpy(dtype=float)
    y_train = train_dataset[target_column].to_numpy(dtype=float)
    x_test = test_dataset[feature_columns].to_numpy(dtype=float)
    y_test = test_dataset[target_column].to_numpy(dtype=float)

    mean_predictions = np.full(len(test_dataset), y_train.mean(), dtype=float)

    linear_model = LinearRegression()
    linear_model.fit(x_train, y_train)
    linear_predictions = linear_model.predict(x_test)

    ridge_model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    ridge_model.fit(x_train, y_train)
    ridge_predictions = ridge_model.predict(x_test)

    return {
        "mean_baseline": compute_metrics(y_test, mean_predictions),
        "linear_regression": compute_metrics(y_test, linear_predictions),
        "ridge_regression": compute_metrics(y_test, ridge_predictions),
    }


def evaluate_retrieval_cases(dataset: pd.DataFrame) -> pd.DataFrame:
    era5_dataset = dataset.loc[~dataset["region_name"].map(is_synthetic_region)].copy()
    synthetic_dataset = dataset.loc[dataset["region_name"].map(is_synthetic_region)].copy()

    rows: list[dict[str, object]] = []
    for noise_case, noise_prefix in (
        ("baseline_nedt", "tb_obs"),
        ("degraded_nedt", "tb_obs_degraded"),
    ):
        for feature_case, feature_columns in feature_sets(noise_prefix).items():
            era5_metrics = run_leave_one_profile_out(
                era5_dataset, feature_columns, PRIMARY_TARGET
            )
            for model_name, metrics in era5_metrics.items():
                rows.append(
                    {
                        "evaluation_split": "era5_leave_one_profile_out",
                        "noise_case": noise_case,
                        "feature_case": feature_case,
                        "model_name": model_name,
                        "target_name": PRIMARY_TARGET,
                        "n_samples": len(era5_dataset),
                        **metrics,
                    }
                )

            synthetic_metrics = run_synthetic_stress_test(
                era5_dataset, synthetic_dataset, feature_columns, PRIMARY_TARGET
            )
            for model_name, metrics in synthetic_metrics.items():
                rows.append(
                    {
                        "evaluation_split": "synthetic_stress_test",
                        "noise_case": noise_case,
                        "feature_case": feature_case,
                        "model_name": model_name,
                        "target_name": PRIMARY_TARGET,
                        "n_samples": len(synthetic_dataset),
                        **metrics,
                    }
                )

    return pd.DataFrame(rows)


def print_summary_table(results: pd.DataFrame) -> None:
    print("Linear retrieval experiment summary:")
    print("Split | Noise case | Feature case | Model | RMSE | MAE | Bias")
    for row in results.itertuples(index=False):
        print(
            f"{row.evaluation_split} | {row.noise_case} | {row.feature_case} | "
            f"{row.model_name} | {row.rmse_k:.3f} K | {row.mae_k:.3f} K | "
            f"{row.bias_k:+.3f} K"
        )


def main() -> None:
    observations = pd.read_csv(INPUT_PATH)
    observations = add_degraded_observations(observations)
    retrieval_dataset = pivot_observations(observations)
    results = evaluate_retrieval_cases(retrieval_dataset)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUTPUT_PATH, index=False)
    print_summary_table(results)
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()
