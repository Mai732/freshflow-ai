from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "data" / "processed" / "cleaned_data.csv"


def main():
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            "cleaned_data.csv does not exist.\n"
            "Run: python clean_data.py"
        )

    df = pd.read_csv(
        DATA_FILE,
        parse_dates=["transaction_date"]
    )

    print("\n======================================")
    print("DATASET OVERVIEW:")

    print(f"\nRows: {len(df):,}")
    print(f"Columns: {len(df.columns):,}")

    print("\nColumn names:")
    for column in df.columns:
        print(f" - {column}")

    # Product information
    print("\n======================================")
    print("PRODUCT INFORMATION")

    print(
        "\nUnique products:",
        df["product_id"].nunique()
    )

    if "category" in df.columns:
        print(
            "Unique categories:",
            df["category"].nunique()
        )

        print("\nTop categories:")

        print(
            df["category"]
            .value_counts()
            .head(10)
        )

    # Date range
    print("DATE RANGE:")
    print("======================================")

    print(
        "Start:",
        df["transaction_date"].min()
    )

    print(
        "End:",
        df["transaction_date"].max()
    )

    # Sales
    print("SALES")
    print("======================================")

    print(
        "Total units sold:",
        round(df["units_sold"].sum(), 2)
    )

    print(
        "Average units sold per record:",
        round(df["units_sold"].mean(), 2)
    )

    product_sales = (
        df.groupby("product_id")["units_sold"]
        .sum()
        .sort_values(ascending=False)
    )

    print("\nTop 10 products by units sold:")
    print(product_sales.head(10))

    # Waste
    if "units_wasted" in df.columns:
        print("\n======================================")
        print("WASTE")
        print("======================================")

        print(
            "Total units wasted:",
            round(df["units_wasted"].sum(), 2)
        )

        product_waste = (
            df.groupby("product_id")["units_wasted"]
            .sum()
            .sort_values(ascending=False)
        )

        print("\nTop 10 products by waste:")
        print(product_waste.head(10))

    if "waste_percentage" in df.columns:
        print(
            "\nAverage waste percentage:",
            round(
                df["waste_percentage"].mean(),
                2
            )
        )

    # Missing values
    print("\n======================================")
    print("MISSING VALUES:")

    missing = (
        df.isna()
        .sum()
        .sort_values(ascending=False)
    )

    print(missing.head(20))


if __name__ == "__main__":
    main()