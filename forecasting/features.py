from pathlib import Path
import json

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "cleaned_data.csv"
)

OUTPUT_FILE = (
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


TARGET = "units_sold"

POSSIBLE_CATEGORICAL_FEATURES = [
    "product_id",
    "product_name",
    "category",
    "product_category",
    "store_id",
    "store_location",
    "location",
    "storage_type",
    "storage_condition",
]

POSSIBLE_NUMERIC_FEATURES = [
    "initial_quantity",
    "selling_price",
    "base_price",
    "cost_price",
    "discount",
    "discount_percentage",
    "shelf_life_days",
    "days_to_expiry",
    "days_remaining_at_purchase",
    "storage_temperature",
    "storage_temperature_c",
    "temperature_deviation",
]

DATE_FEATURES = [
    "year",
    "month",
    "day_of_month",
    "day_of_week",
    "week_of_year",
    "is_weekend",
]


def load_data():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            "\ncleaned_data.csv was not found.\n"
            "Run first:\n"
            "python clean_data.py"
        )

    df = pd.read_csv(
        INPUT_FILE,
        parse_dates=["transaction_date"],
    )

    print("\n======================================")
    print("LOADED CLEAN DATA")

    print(f"Rows   : {len(df):,}")
    print(f"Columns: {len(df.columns):,}")

    return df


def create_date_features(df):
    df = df.copy()

    df["year"] = (
        df["transaction_date"]
        .dt.year
    )

    df["month"] = (
        df["transaction_date"]
        .dt.month
    )

    df["day_of_month"] = (
        df["transaction_date"]
        .dt.day
    )

    df["day_of_week"] = (
        df["transaction_date"]
        .dt.dayofweek
    )

    df["week_of_year"] = (
        df["transaction_date"]
        .dt.isocalendar()
        .week
        .astype(int)
    )

    df["is_weekend"] = (
        df["day_of_week"] >= 5
    ).astype(int)

    return df


def select_available_features(df):
    categorical_features = [
        column
        for column in POSSIBLE_CATEGORICAL_FEATURES
        if column in df.columns
    ]

    numeric_features = [
        column
        for column in POSSIBLE_NUMERIC_FEATURES
        if column in df.columns
    ]

    numeric_features += DATE_FEATURES

    # Remove duplicates while keeping the original order
    categorical_features = list(
        dict.fromkeys(
            categorical_features
        )
    )

    numeric_features = list(
        dict.fromkeys(
            numeric_features
        )
    )

    return (
        categorical_features,
        numeric_features,
    )


def prepare_ml_data(
    df,
    categorical_features,
    numeric_features,
):
    df = df.copy()

    if TARGET not in df.columns:
        raise ValueError(
            f"Target column '{TARGET}' "
            "was not found."
        )

    # Make sure the target is numeric
    df[TARGET] = pd.to_numeric(
        df[TARGET],
        errors="coerce",
    )

    # Remove rows that are missing the target or transaction date
    df = df.dropna(
        subset=[
            TARGET,
            "transaction_date",
        ]
    ).copy()

    # Demand cannot be negative
    df[TARGET] = (
        df[TARGET]
        .clip(lower=0)
    )

    # Clean categorical values
    for column in categorical_features:
        df[column] = (
            df[column]
            .fillna("UNKNOWN")
            .astype(str)
            .str.strip()
        )

        df.loc[
            df[column] == "",
            column,
        ] = "UNKNOWN"

    # Convert numeric features
    for column in numeric_features:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    return df


def save_output(
    df,
    categorical_features,
    numeric_features,
):
    feature_columns = (
        categorical_features
        + numeric_features
    )

    columns_to_save = (
        ["transaction_date"]
        + feature_columns
        + [TARGET]
    )

    output_df = df[
        columns_to_save
    ].copy()

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    FEATURE_CONFIG_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    config = {
        "target": TARGET,
        "feature_columns": feature_columns,
        "categorical_features": categorical_features,
        "numeric_features": numeric_features,
    }

    with open(
        FEATURE_CONFIG_FILE,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            config,
            file,
            indent=4,
        )

    print("\n======================================")
    print("FORECAST FEATURES READY")

    print(
        f"\nRows available: "
        f"{len(output_df):,}"
    )

    print("\nCategorical features:")

    for feature in categorical_features:
        print(" -", feature)

    print("\nNumeric features:")

    for feature in numeric_features:
        print(" -", feature)

    print(
        "\nTraining data saved to:"
    )
    print(OUTPUT_FILE)

    print(
        "\nFeature configuration saved to:"
    )
    print(FEATURE_CONFIG_FILE)


def main():
    df = load_data()

    df = create_date_features(df)

    (
        categorical_features,
        numeric_features,
    ) = select_available_features(df)

    print(
        "\nNumber of categorical features:",
        len(categorical_features),
    )

    print(
        "Number of numeric features:",
        len(numeric_features),
    )

    df = prepare_ml_data(
        df,
        categorical_features,
        numeric_features,
    )

    save_output(
        df,
        categorical_features,
        numeric_features,
    )


if __name__ == "__main__":
    main()