from pathlib import Path
import json
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]

if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))


from live_inventory import (
    get_product,
    get_live_inventory_snapshot,
    load_live_sales,
)


HISTORICAL_DATA_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "cleaned_data.csv"
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


def load_resources():
    if not HISTORICAL_DATA_FILE.exists():
        raise FileNotFoundError(
            "cleaned_data.csv was not found."
        )

    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            "LightGBM model was not found.\n"
            "Run:\n"
            "python forecasting/train.py"
        )

    if not METADATA_FILE.exists():
        raise FileNotFoundError(
            "lightgbm_metadata.json was not found."
        )

    historical_df = pd.read_csv(
        HISTORICAL_DATA_FILE,
        parse_dates=["transaction_date"],
    )

    historical_df["product_id"] = (
        historical_df["product_id"]
        .astype(str)
        .str.strip()
    )

    with open(
        METADATA_FILE,
        "r",
        encoding="utf-8",
    ) as file:
        metadata = json.load(file)

    model = lgb.Booster(
        model_file=str(MODEL_FILE)
    )

    return (
        model,
        historical_df,
        metadata,
    )


def get_historical_product_data(
    historical_df,
    product_id,
):
    product_id = str(product_id).strip()

    product_data = historical_df[
        historical_df["product_id"]
        == product_id
    ].copy()

    if product_data.empty:
        return None

    product_data = (
        product_data
        .sort_values("transaction_date")
    )

    return product_data


def get_live_snapshot_row(product_id):
    snapshot = get_live_inventory_snapshot()

    if snapshot.empty:
        return None

    snapshot["product_id"] = (
        snapshot["product_id"]
        .astype(str)
        .str.strip()
    )

    match = snapshot[
        snapshot["product_id"]
        == str(product_id).strip()
    ]

    if match.empty:
        return None

    return match.iloc[-1].to_dict()


def build_product_context(
    historical_df,
    product_id,
):
    product_id = str(product_id).strip()

    historical_data = get_historical_product_data(
        historical_df,
        product_id,
    )

    live_product = get_product(
        product_id
    )

    live_snapshot = get_live_snapshot_row(
        product_id
    )

    if (
        historical_data is None
        and live_product is None
    ):
        raise ValueError(
            f"Product '{product_id}' was not found "
            "in either historical data or live inventory."
        )

    context = {}

    if historical_data is not None:
        latest_historical = historical_data.iloc[-1]
        context.update(
            latest_historical.to_dict()
        )

    # Live product information should override older historical values.
    if live_product is not None:
        for key, value in live_product.items():
            if (
                value is not None
                and str(value).lower() != "nan"
            ):
                context[key] = value

    # Current live inventory has the highest priority.
    if live_snapshot is not None:
        context["initial_quantity"] = live_snapshot.get(
            "current_quantity",
            context.get(
                "initial_quantity",
                0,
            ),
        )

        context["current_quantity"] = live_snapshot.get(
            "current_quantity",
            0,
        )

        context["days_to_expiry"] = live_snapshot.get(
            "days_to_expiry",
            context.get(
                "days_to_expiry",
                7,
            ),
        )

        if live_snapshot.get("selling_price") is not None:
            context["selling_price"] = live_snapshot.get(
                "selling_price"
            )

        if live_snapshot.get("cost_price") is not None:
            context["cost_price"] = live_snapshot.get(
                "cost_price"
            )

        if live_snapshot.get("storage_type") is not None:
            context["storage_type"] = live_snapshot.get(
                "storage_type"
            )

        for field in [
            "product_name",
            "category",
            "shelf_life_days",
        ]:
            if live_snapshot.get(field) is not None:
                context[field] = live_snapshot.get(
                    field
                )

    # Support both category column names used by different datasets.
    if (
        "category" in context
        and "product_category" not in context
    ):
        context["product_category"] = context[
            "category"
        ]

    if (
        "product_category" in context
        and "category" not in context
    ):
        context["category"] = context[
            "product_category"
        ]

    context["product_id"] = product_id

    # Start from today for live inventory.
    # Otherwise continue from the end of the historical dataset.
    if live_snapshot is not None:
        base_date = (
            pd.Timestamp.today()
            .normalize()
        )

    elif historical_data is not None:
        base_date = historical_data[
            "transaction_date"
        ].max()

    else:
        base_date = (
            pd.Timestamp.today()
            .normalize()
        )

    return (
        context,
        base_date,
        historical_data,
        live_snapshot,
    )


def encode_category(
    feature,
    value,
    metadata,
):
    mappings = metadata.get(
        "category_mappings",
        {},
    )

    mapping = mappings.get(
        feature,
        {},
    )

    if (
        value is None
        or pd.isna(value)
    ):
        value = "UNKNOWN"

    value = str(value).strip()

    if not value:
        value = "UNKNOWN"

    # New product IDs or unseen categories are mapped to -1.
    return int(
        mapping.get(
            value,
            -1,
        )
    )


def build_future_row(
    context,
    prediction_date,
    step,
    metadata,
):
    feature_columns = metadata[
        "feature_columns"
    ]

    categorical_features = metadata[
        "categorical_features"
    ]

    numeric_features = metadata[
        "numeric_features"
    ]

    numeric_medians = metadata[
        "numeric_medians"
    ]

    future = {}

    for feature in feature_columns:
        future[feature] = context.get(
            feature
        )

    # Calendar features for the future prediction date
    future["year"] = prediction_date.year
    future["month"] = prediction_date.month
    future["day_of_month"] = prediction_date.day
    future["day_of_week"] = prediction_date.dayofweek

    future["week_of_year"] = int(
        prediction_date
        .isocalendar()
        .week
    )

    future["is_weekend"] = int(
        prediction_date.dayofweek >= 5
    )

    # Reduce days remaining until expiry as the forecast moves forward.
    if "days_to_expiry" in feature_columns:
        original_days = context.get(
            "days_to_expiry",
            numeric_medians.get(
                "days_to_expiry",
                7,
            ),
        )

        try:
            original_days = float(
                original_days
            )

            if np.isnan(original_days):
                raise ValueError

        except Exception:
            original_days = float(
                numeric_medians.get(
                    "days_to_expiry",
                    7,
                )
            )

        future["days_to_expiry"] = max(
            0,
            original_days - step,
        )

    for feature in categorical_features:
        future[feature] = encode_category(
            feature,
            future.get(feature),
            metadata,
        )

    for feature in numeric_features:
        value = future.get(feature)

        try:
            value = float(value)

            if np.isnan(value):
                raise ValueError

        except Exception:
            value = float(
                numeric_medians.get(
                    feature,
                    0,
                )
            )

        future[feature] = value

    row = pd.DataFrame(
        [future]
    )

    return row[
        feature_columns
    ]


def get_recent_live_sales_summary(
    product_id,
    days=7,
):
    sales = load_live_sales()

    if sales.empty:
        return {
            "total_live_sales": 0.0,
            "average_daily_live_sales": 0.0,
            "live_sale_days": 0,
        }

    sales["product_id"] = (
        sales["product_id"]
        .astype(str)
        .str.strip()
    )

    sales["sale_date"] = pd.to_datetime(
        sales["sale_date"],
        errors="coerce",
    )

    sales["quantity_sold"] = pd.to_numeric(
        sales["quantity_sold"],
        errors="coerce",
    ).fillna(0)

    product_sales = sales[
        sales["product_id"]
        == str(product_id).strip()
    ].copy()

    if product_sales.empty:
        return {
            "total_live_sales": 0.0,
            "average_daily_live_sales": 0.0,
            "live_sale_days": 0,
        }

    cutoff = (
        pd.Timestamp.today()
        .normalize()
        - pd.Timedelta(
            days=days - 1
        )
    )

    recent = product_sales[
        product_sales["sale_date"]
        >= cutoff
    ].copy()

    if recent.empty:
        return {
            "total_live_sales": 0.0,
            "average_daily_live_sales": 0.0,
            "live_sale_days": 0,
        }

    daily = (
        recent.groupby(
            "sale_date"
        )["quantity_sold"]
        .sum()
    )

    return {
        "total_live_sales": float(
            daily.sum()
        ),
        "average_daily_live_sales": float(
            daily.mean()
        ),
        "live_sale_days": int(
            len(daily)
        ),
    }


def predict_next_days(
    product_id,
    days=7,
):
    if days < 1:
        raise ValueError(
            "Forecast days must be at least 1."
        )

    days = min(
        int(days),
        30,
    )

    (
        model,
        historical_df,
        metadata,
    ) = load_resources()

    (
        context,
        base_date,
        historical_data,
        live_snapshot,
    ) = build_product_context(
        historical_df,
        product_id,
    )

    forecasts = []

    for step in range(
        1,
        days + 1,
    ):
        prediction_date = (
            base_date
            + pd.Timedelta(
                days=step
            )
        )

        X_future = build_future_row(
            context=context,
            prediction_date=prediction_date,
            step=step,
            metadata=metadata,
        )

        prediction = float(
            model.predict(
                X_future,
                num_iteration=metadata.get(
                    "best_iteration"
                ),
            )[0]
        )

        prediction = max(
            0.0,
            prediction,
        )

        forecasts.append(
            {
                "date": prediction_date.date(),
                "predicted_demand": round(
                    prediction,
                    2,
                ),
            }
        )

    result = pd.DataFrame(
        forecasts
    )

    return result


def predict_total_demand(
    product_id,
    days,
):
    forecast = predict_next_days(
        product_id=product_id,
        days=days,
    )

    return float(
        forecast[
            "predicted_demand"
        ].sum()
    )


if __name__ == "__main__":
    product_id = input(
        "Enter product ID: "
    ).strip()

    days = int(
        input(
            "How many days to forecast? "
        )
    )

    forecast = predict_next_days(
        product_id,
        days,
    )

    print(
        "\n======================================"
    )

    print(
        "LIVE-AWARE DEMAND FORECAST"
    )

    print(
        "======================================\n"
    )

    print(
        forecast.to_string(
            index=False
        )
    )

    print(
        "\nTotal predicted demand:",
        round(
            forecast[
                "predicted_demand"
            ].sum(),
            2,
        ),
    )

    live_sales = get_recent_live_sales_summary(
        product_id
    )

    print(
        "\nRecent live sales:"
    )

    print(
        live_sales
    )