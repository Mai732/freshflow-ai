from pathlib import Path
import json

import lightgbm as lgb
import numpy as np
import pandas as pd

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


BASE_DIR = Path(__file__).resolve().parents[1]

TRAINING_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "forecast_training_data.csv"
)

FEATURE_CONFIG_FILE = (
    BASE_DIR
    / "models"
    / "forecast_feature_config.json"
)

MODEL_FILE = (
    BASE_DIR
    / "models"
    / "demand_forecast_model.txt"
)

METADATA_FILE = (
    BASE_DIR
    / "models"
    / "lightgbm_metadata.json"
)

METRICS_FILE = (
    BASE_DIR
    / "models"
    / "forecast_metrics.json"
)


def load_data():
    if not TRAINING_FILE.exists():
        raise FileNotFoundError(
            "\nforecast_training_data.csv "
            "was not found.\n\n"
            "Run:\n"
            "python forecasting/features.py"
        )

    if not FEATURE_CONFIG_FILE.exists():
        raise FileNotFoundError(
            "\nforecast_feature_config.json "
            "was not found."
        )

    df = pd.read_csv(
        TRAINING_FILE,
        parse_dates=["transaction_date"],
    )

    with open(
        FEATURE_CONFIG_FILE,
        "r",
        encoding="utf-8",
    ) as file:
        config = json.load(file)

    return df, config


def chronological_split(df):
    df = (
        df.sort_values("transaction_date")
        .reset_index(drop=True)
    )

    unique_dates = sorted(
        df["transaction_date"]
        .dropna()
        .unique()
    )

    if len(unique_dates) >= 5:
        split_position = int(
            len(unique_dates) * 0.80
        )

        split_position = max(
            1,
            min(
                split_position,
                len(unique_dates) - 1,
            ),
        )

        cutoff_date = unique_dates[
            split_position
        ]

        train_df = df[
            df["transaction_date"] < cutoff_date
        ].copy()

        test_df = df[
            df["transaction_date"] >= cutoff_date
        ].copy()

    else:
        split_index = int(
            len(df) * 0.80
        )

        train_df = (
            df.iloc[:split_index]
            .copy()
        )

        test_df = (
            df.iloc[split_index:]
            .copy()
        )

    if train_df.empty or test_df.empty:
        raise ValueError(
            "Train/test split failed."
        )

    return train_df, test_df


def create_category_mappings(
    train_df,
    categorical_features,
):
    mappings = {}

    for column in categorical_features:
        values = (
            train_df[column]
            .fillna("UNKNOWN")
            .astype(str)
            .str.strip()
        )

        unique_values = (
            values
            .drop_duplicates()
            .tolist()
        )

        mapping = {
            value: index
            for index, value in enumerate(
                unique_values
            )
        }

        mappings[column] = mapping

    return mappings


def encode_categories(
    df,
    categorical_features,
    mappings,
):
    df = df.copy()

    for column in categorical_features:
        values = (
            df[column]
            .fillna("UNKNOWN")
            .astype(str)
            .str.strip()
        )

        mapping = mappings[column]

        df[column] = (
            values
            .map(mapping)
            .fillna(-1)
            .astype("int32")
        )

    return df


def calculate_numeric_medians(
    train_df,
    numeric_features,
):
    medians = {}

    for column in numeric_features:
        numeric = pd.to_numeric(
            train_df[column],
            errors="coerce",
        )

        median = numeric.median()

        if pd.isna(median):
            median = 0.0

        medians[column] = float(median)

    return medians


def clean_numeric_features(
    df,
    numeric_features,
    medians,
):
    df = df.copy()

    for column in numeric_features:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

        df[column] = (
            df[column]
            .fillna(
                medians[column]
            )
            .astype("float32")
        )

    return df


def calculate_wape(
    actual,
    predicted,
):
    actual = np.asarray(actual)
    predicted = np.asarray(predicted)

    denominator = (
        np.abs(actual)
        .sum()
    )

    if denominator == 0:
        return None

    return float(
        np.abs(
            actual - predicted
        ).sum()
        / denominator
        * 100
    )


def main():
    print("\nLoading forecasting data...")

    df, config = load_data()

    feature_columns = config[
        "feature_columns"
    ]

    categorical_features = config[
        "categorical_features"
    ]

    numeric_features = config[
        "numeric_features"
    ]

    target = config["target"]

    train_df, test_df = (
        chronological_split(df)
    )

    print("\n======================================")
    print("TRAIN / TEST SPLIT")
    print("======================================")

    print(
        f"Training rows: "
        f"{len(train_df):,}"
    )

    print(
        f"Testing rows : "
        f"{len(test_df):,}"
    )

    print("\nTraining period:")

    print(
        train_df["transaction_date"].min(),
        "→",
        train_df["transaction_date"].max(),
    )

    print("\nTesting period:")

    print(
        test_df["transaction_date"].min(),
        "→",
        test_df["transaction_date"].max(),
    )

    category_mappings = create_category_mappings(
        train_df,
        categorical_features,
    )

    train_df = encode_categories(
        train_df,
        categorical_features,
        category_mappings,
    )

    test_df = encode_categories(
        test_df,
        categorical_features,
        category_mappings,
    )

    numeric_medians = calculate_numeric_medians(
        train_df,
        numeric_features,
    )

    train_df = clean_numeric_features(
        train_df,
        numeric_features,
        numeric_medians,
    )

    test_df = clean_numeric_features(
        test_df,
        numeric_features,
        numeric_medians,
    )

    X_train = train_df[
        feature_columns
    ].copy()

    X_test = test_df[
        feature_columns
    ].copy()

    y_train = pd.to_numeric(
        train_df[target],
        errors="coerce",
    )

    y_test = pd.to_numeric(
        test_df[target],
        errors="coerce",
    )

    model = lgb.LGBMRegressor(
        objective="regression",
        n_estimators=1000,
        learning_rate=0.04,
        num_leaves=31,
        max_depth=-1,
        min_child_samples=30,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.1,
        reg_lambda=0.2,
        random_state=42,
        n_jobs=-1,
        verbosity=-1,
    )

    print("\n======================================")
    print("TRAINING LIGHTGBM")
    print("======================================\n")

    model.fit(
        X_train,
        y_train,
        categorical_feature=categorical_features,
        eval_set=[
            (
                X_test,
                y_test,
            )
        ],
        eval_metric="mae",
        callbacks=[
            lgb.early_stopping(
                stopping_rounds=50,
                verbose=True,
            ),
            lgb.log_evaluation(
                period=100
            ),
        ],
    )

    predictions = model.predict(
        X_test,
        num_iteration=model.best_iteration_,
    )

    predictions = np.maximum(
        predictions,
        0,
    )

    mae = mean_absolute_error(
        y_test,
        predictions,
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            predictions,
        )
    )

    r2 = r2_score(
        y_test,
        predictions,
    )

    wape = calculate_wape(
        y_test,
        predictions,
    )

    baseline_value = float(
        y_train.mean()
    )

    baseline_predictions = np.full(
        len(y_test),
        baseline_value,
    )

    baseline_mae = mean_absolute_error(
        y_test,
        baseline_predictions,
    )

    if baseline_mae > 0:
        improvement = (
            (
                baseline_mae - mae
            )
            / baseline_mae
            * 100
        )
    else:
        improvement = 0.0

    print("\n======================================")
    print("MODEL RESULTS")
    print("======================================")

    print(
        f"MAE  : {mae:.3f}"
    )

    print(
        f"RMSE : {rmse:.3f}"
    )

    if wape is not None:
        print(
            f"WAPE : {wape:.2f}%"
        )
    else:
        print(
            "WAPE : unavailable"
        )

    print(
        f"R²   : {r2:.3f}"
    )

    print(
        f"\nBaseline MAE: "
        f"{baseline_mae:.3f}"
    )

    print(
        "Improvement over baseline: "
        f"{improvement:.2f}%"
    )

    importances = pd.DataFrame(
        {
            "feature": feature_columns,
            "importance": model.feature_importances_,
        }
    )

    importances = (
        importances
        .sort_values(
            "importance",
            ascending=False,
        )
    )

    print("\n======================================")
    print("TOP FEATURES")
    print("======================================")

    print(
        importances
        .head(15)
        .to_string(
            index=False
        )
    )

    MODEL_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    model.booster_.save_model(
        str(MODEL_FILE)
    )

    metadata = {
        "target": target,
        "feature_columns": feature_columns,
        "categorical_features": categorical_features,
        "numeric_features": numeric_features,
        "category_mappings": category_mappings,
        "numeric_medians": numeric_medians,
        "best_iteration": int(
            model.best_iteration_
            or model.n_estimators
        ),
    }

    with open(
        METADATA_FILE,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata,
            file,
            indent=2,
        )

    metrics = {
        "mae": float(mae),
        "rmse": float(rmse),
        "wape": (
            None
            if wape is None
            else float(wape)
        ),
        "r2": float(r2),
        "baseline_mae": float(
            baseline_mae
        ),
        "improvement_percent": float(
            improvement
        ),
    }

    with open(
        METRICS_FILE,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metrics,
            file,
            indent=2,
        )

    print(
        "\nModel saved:"
    )
    print(MODEL_FILE)

    print(
        "\nMetadata saved:"
    )
    print(METADATA_FILE)

    print(
        "\nMetrics saved:"
    )
    print(METRICS_FILE)


if __name__ == "__main__":
    main()