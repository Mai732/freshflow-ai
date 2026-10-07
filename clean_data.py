from pathlib import Path
import re

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = PROCESSED_DIR / "cleaned_data.csv"


def clean_column_name(column_name):
    """
    Convert column names such as:

    Product ID
    Product-ID
    PRODUCT_ID

    into:

    product_id
    """
    column_name = str(column_name).strip().lower()

    # Replace non-alphanumeric characters with underscores
    column_name = re.sub(r"[^a-z0-9]+", "_", column_name)

    # Remove repeated underscores
    column_name = re.sub(r"_+", "_", column_name)

    return column_name.strip("_")


def find_csv():
    """Find the first CSV file inside data/raw/."""
    csv_files = list(RAW_DIR.glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(
            "\nNo CSV file was found inside data/raw/\n"
            "Put your Kaggle CSV inside:\n"
            "data/raw/"
        )

    print(f"\nUsing dataset: {csv_files[0].name}")

    return csv_files[0]


def find_column(df, possible_names):
    """
    Search for a column using several possible names.

    Returns None if no matching column exists.
    """
    for name in possible_names:
        if name in df.columns:
            return name

    return None


def load_data():
    csv_path = find_csv()

    print("\nLoading dataset...")

    df = pd.read_csv(csv_path)

    print(f"Original rows: {len(df):,}")
    print(f"Original columns: {len(df.columns)}")

    return df


def clean_data(df):
    print("\nCleaning column names...")

    df.columns = [
        clean_column_name(column)
        for column in df.columns
    ]

    print("\nColumns found:")

    for column in df.columns:
        print(f" - {column}")

    # Remove duplicate rows
    before = len(df)

    df = df.drop_duplicates().copy()

    removed = before - len(df)

    print(f"\nDuplicate rows removed: {removed:,}")

    # Find the main columns used by the project
    product_id_col = find_column(
        df,
        [
            "product_id",
            "productid",
            "item_id",
            "itemid"
        ]
    )

    product_name_col = find_column(
        df,
        [
            "product_name",
            "product",
            "item_name"
        ]
    )

    category_col = find_column(
        df,
        [
            "product_category",
            "category",
            "product_type"
        ]
    )

    transaction_date_col = find_column(
        df,
        [
            "transaction_date",
            "date",
            "sales_date",
            "sale_date"
        ]
    )

    expiration_date_col = find_column(
        df,
        [
            "expiration_date",
            "expiry_date",
            "expiry"
        ]
    )

    units_sold_col = find_column(
        df,
        [
            "units_sold",
            "quantity_sold",
            "sales_quantity",
            "sold_quantity"
        ]
    )

    units_wasted_col = find_column(
        df,
        [
            "units_wasted",
            "quantity_wasted",
            "waste_quantity",
            "wasted_units"
        ]
    )

    initial_quantity_col = find_column(
        df,
        [
            "initial_quantity",
            "initial_stock",
            "stock_quantity",
            "quantity"
        ]
    )

    selling_price_col = find_column(
        df,
        [
            "selling_price",
            "sale_price",
            "price"
        ]
    )

    waste_percentage_col = find_column(
        df,
        [
            "waste_percentage",
            "waste_percent",
            "waste_rate"
        ]
    )

    print("\nImportant column mapping:")

    mappings = {
        "product_id": product_id_col,
        "product_name": product_name_col,
        "category": category_col,
        "transaction_date": transaction_date_col,
        "expiration_date": expiration_date_col,
        "units_sold": units_sold_col,
        "units_wasted": units_wasted_col,
        "initial_quantity": initial_quantity_col,
        "selling_price": selling_price_col,
        "waste_percentage": waste_percentage_col,
    }

    for standard_name, actual_name in mappings.items():
        print(
            f"{standard_name:22} -> "
            f"{actual_name if actual_name else 'NOT FOUND'}"
        )

    # Rename the main columns to consistent names
    rename_map = {}

    for standard_name, actual_name in mappings.items():
        if actual_name is not None:
            rename_map[actual_name] = standard_name

    df = df.rename(columns=rename_map)

    # Make sure the minimum required columns exist
    required = [
        "product_id",
        "transaction_date",
        "units_sold"
    ]

    missing_required = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing_required:
        raise ValueError(
            "\nThe following required columns could not be found:\n"
            + ", ".join(missing_required)
            + "\n\nLook at the printed column names and update "
              "the mapping in clean_data.py."
        )

    # Convert date columns
    date_columns = [
        "transaction_date",
        "expiration_date"
    ]

    for column in date_columns:
        if column in df.columns:
            df[column] = pd.to_datetime(
                df[column],
                errors="coerce"
            )

    # Convert numeric columns
    possible_numeric_columns = [
        "units_sold",
        "units_wasted",
        "initial_quantity",
        "selling_price",
        "waste_percentage",
        "shelf_life_days",
        "days_remaining_at_purchase",
        "storage_temperature",
        "storage_temperature_c",
        "temperature_deviation",
        "base_price",
        "cost_price",
        "revenue",
        "profit",
        "waste_cost"
    ]

    for column in possible_numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(
                df[column],
                errors="coerce"
            )

    # Remove rows missing the core fields
    df = df.dropna(
        subset=[
            "product_id",
            "transaction_date",
            "units_sold"
        ]
    ).copy()

    # Sales cannot be negative
    df["units_sold"] = df["units_sold"].clip(lower=0)

    # Create derived features

    # Number of days between transaction and expiration
    if "expiration_date" in df.columns:
        df["days_to_expiry"] = (
            df["expiration_date"]
            - df["transaction_date"]
        ).dt.days

    # Estimate remaining inventory
    if (
        "initial_quantity" in df.columns
        and "units_sold" in df.columns
    ):
        wasted = (
            df["units_wasted"]
            if "units_wasted" in df.columns
            else 0
        )

        df["remaining_inventory"] = (
            df["initial_quantity"]
            - df["units_sold"]
            - wasted
        )

        df["remaining_inventory"] = (
            df["remaining_inventory"]
            .clip(lower=0)
        )

    # Calculate waste percentage if the dataset does not already contain it
    if (
        "waste_percentage" not in df.columns
        and "units_wasted" in df.columns
        and "initial_quantity" in df.columns
    ):
        denominator = (
            df["initial_quantity"]
            .replace(0, np.nan)
        )

        df["waste_percentage"] = (
            df["units_wasted"]
            / denominator
            * 100
        )

    # Add basic date features
    df["year"] = df["transaction_date"].dt.year
    df["month"] = df["transaction_date"].dt.month
    df["day_of_week"] = df["transaction_date"].dt.dayofweek
    df["is_weekend"] = (
        df["day_of_week"] >= 5
    ).astype(int)

    # Keep the dataset ordered by product and date
    df = df.sort_values(
        [
            "product_id",
            "transaction_date"
        ]
    ).reset_index(drop=True)

    return df


def save_data(df):
    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("CLEANING COMPLETE")
    print("---------------------------------------")

    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")

    print(
        f"\nClean dataset saved to:\n"
        f"{OUTPUT_FILE}"
    )

    print("\nMissing values in important columns:")

    important = [
        "product_id",
        "transaction_date",
        "units_sold",
        "expiration_date",
        "initial_quantity",
        "units_wasted"
    ]

    for column in important:
        if column in df.columns:
            print(
                f"{column:22}: "
                f"{df[column].isna().sum():,}"
            )


if __name__ == "__main__":
    data = load_data()
    cleaned_data = clean_data(data)
    save_data(cleaned_data)