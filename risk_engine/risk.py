from pathlib import Path
import sys

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]

if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))


from forecasting.predict import (
    predict_total_demand,
    get_recent_live_sales_summary,
)

from live_inventory import (
    get_live_inventory_snapshot,
    get_product,
)


HISTORICAL_DATA_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "cleaned_data.csv"
)


def calculate_expiry_score(days_to_expiry):
    if days_to_expiry <= 1:
        return 1.00

    if days_to_expiry <= 3:
        return 0.80

    if days_to_expiry <= 7:
        return 0.50

    if days_to_expiry <= 14:
        return 0.25

    return 0.10


def calculate_expected_surplus(
    current_quantity,
    predicted_demand,
):
    return max(
        0.0,
        float(current_quantity) - float(predicted_demand),
    )


def calculate_surplus_score(
    current_quantity,
    expected_surplus,
):
    if current_quantity <= 0:
        return 0.0

    return float(
        np.clip(
            expected_surplus / current_quantity,
            0,
            1,
        )
    )


def load_historical_data():
    if not HISTORICAL_DATA_FILE.exists():
        return pd.DataFrame()

    df = pd.read_csv(
        HISTORICAL_DATA_FILE,
        parse_dates=["transaction_date"],
    )

    df["product_id"] = (
        df["product_id"]
        .astype(str)
        .str.strip()
    )

    return df


def get_historical_product_data(
    historical_df,
    product_id,
):
    if historical_df.empty:
        return pd.DataFrame()

    result = historical_df[
        historical_df["product_id"]
        == str(product_id).strip()
    ].copy()

    return result


def get_live_product_snapshot(product_id):
    snapshot = get_live_inventory_snapshot()

    if snapshot.empty:
        return None

    snapshot["product_id"] = (
        snapshot["product_id"]
        .astype(str)
        .str.strip()
    )

    result = snapshot[
        snapshot["product_id"]
        == str(product_id).strip()
    ]

    if result.empty:
        return None

    return result.iloc[-1].to_dict()


def historical_slow_moving_score(product_data):
    if product_data.empty:
        # No history available, so treat the risk as unknown.
        return 0.50

    if (
        "initial_quantity" not in product_data.columns
        or "units_sold" not in product_data.columns
    ):
        return 0.50

    initial = pd.to_numeric(
        product_data["initial_quantity"],
        errors="coerce",
    )

    sold = pd.to_numeric(
        product_data["units_sold"],
        errors="coerce",
    )

    valid = initial > 0

    if not valid.any():
        return 0.50

    sell_through = (
        sold[valid]
        / initial[valid]
    )

    sell_through = sell_through.clip(
        lower=0,
        upper=1,
    )

    mean_sell_through = float(
        sell_through.mean()
    )

    return float(
        np.clip(
            1 - mean_sell_through,
            0,
            1,
        )
    )


def live_slow_moving_score(
    product_id,
    predicted_demand,
    forecast_days,
):
    live_sales = get_recent_live_sales_summary(
        product_id=product_id,
        days=7,
    )

    live_sale_days = live_sales[
        "live_sale_days"
    ]

    # Fall back to historical behavior if there are no live sales yet.
    if live_sale_days <= 0:
        return None

    observed_daily_sales = live_sales[
        "average_daily_live_sales"
    ]

    expected_daily_sales = (
        predicted_demand
        / max(1, forecast_days)
    )

    if expected_daily_sales <= 0:
        return 0.0

    ratio = (
        observed_daily_sales
        / expected_daily_sales
    )

    # If live sales meet or exceed expected demand,
    # the product is not considered slow-moving.
    if ratio >= 1:
        return 0.0

    return float(
        np.clip(
            1 - ratio,
            0,
            1,
        )
    )


def calculate_slow_moving_score(
    product_id,
    product_data,
    predicted_demand,
    forecast_days,
):
    live_score = live_slow_moving_score(
        product_id=product_id,
        predicted_demand=predicted_demand,
        forecast_days=forecast_days,
    )

    # Prefer live sales whenever they are available.
    if live_score is not None:
        return live_score

    return historical_slow_moving_score(
        product_data
    )


def calculate_historical_waste_score(
    historical_df,
    product_data,
    category=None,
):
    # Use product-specific waste history first.
    if not product_data.empty:
        if "waste_percentage" in product_data.columns:
            waste = pd.to_numeric(
                product_data["waste_percentage"],
                errors="coerce",
            ).dropna()

            if not waste.empty:
                value = float(waste.mean())

                if value > 1:
                    value = value / 100

                return float(
                    np.clip(
                        value,
                        0,
                        1,
                    )
                )

        if (
            "units_wasted" in product_data.columns
            and "initial_quantity" in product_data.columns
        ):
            wasted = pd.to_numeric(
                product_data["units_wasted"],
                errors="coerce",
            )

            initial = pd.to_numeric(
                product_data["initial_quantity"],
                errors="coerce",
            )

            valid = initial > 0

            if valid.any():
                ratio = (
                    wasted[valid]
                    / initial[valid]
                )

                return float(
                    np.clip(
                        ratio.mean(),
                        0,
                        1,
                    )
                )

    # For a new product, use waste history from the same category.
    if category and not historical_df.empty:
        category_column = None

        if "category" in historical_df.columns:
            category_column = "category"

        elif "product_category" in historical_df.columns:
            category_column = "product_category"

        if category_column:
            category_data = historical_df[
                historical_df[category_column]
                .astype(str)
                .str.lower()
                == str(category).lower()
            ]

            if not category_data.empty:
                if "waste_percentage" in category_data.columns:
                    waste = pd.to_numeric(
                        category_data["waste_percentage"],
                        errors="coerce",
                    ).dropna()

                    if not waste.empty:
                        value = float(waste.mean())

                        if value > 1:
                            value = value / 100

                        return float(
                            np.clip(
                                value,
                                0,
                                1,
                            )
                        )

    return 0.0


def classify_risk(risk_score):
    if risk_score >= 0.80:
        return "CRITICAL"

    if risk_score >= 0.60:
        return "HIGH"

    if risk_score >= 0.30:
        return "MEDIUM"

    return "LOW"


def calculate_waste_risk(
    current_quantity,
    predicted_demand,
    days_to_expiry,
    slow_moving_score,
    historical_waste_score,
):
    expected_surplus = calculate_expected_surplus(
        current_quantity,
        predicted_demand,
    )

    surplus_score = calculate_surplus_score(
        current_quantity,
        expected_surplus,
    )

    expiry_score = calculate_expiry_score(
        days_to_expiry
    )

    # Surplus has the highest weight because predicted
    # unsold inventory is the strongest waste signal.
    risk_score = (
        0.50 * surplus_score
        + 0.25 * expiry_score
        + 0.15 * slow_moving_score
        + 0.10 * historical_waste_score
    )

    risk_score = float(
        np.clip(
            risk_score,
            0,
            1,
        )
    )

    return {
        "current_quantity": round(
            float(current_quantity),
            2,
        ),
        "predicted_demand": round(
            float(predicted_demand),
            2,
        ),
        "expected_surplus": round(
            float(expected_surplus),
            2,
        ),
        "days_to_expiry": int(
            days_to_expiry
        ),
        "surplus_score": round(
            surplus_score,
            3,
        ),
        "expiry_score": round(
            expiry_score,
            3,
        ),
        "slow_moving_score": round(
            slow_moving_score,
            3,
        ),
        "historical_waste_score": round(
            historical_waste_score,
            3,
        ),
        "risk_score": round(
            risk_score,
            3,
        ),
        "risk_level": classify_risk(
            risk_score
        ),
    }


def analyze_product(product_id):
    product_id = str(product_id).strip()

    historical_df = load_historical_data()

    product_data = get_historical_product_data(
        historical_df,
        product_id,
    )

    live_snapshot = get_live_product_snapshot(
        product_id
    )

    live_product = get_product(
        product_id
    )

    if (
        product_data.empty
        and live_product is None
    ):
        raise ValueError(
            f"Product '{product_id}' does not exist."
        )

    product_name = None
    category = None

    if live_product:
        product_name = live_product.get(
            "product_name"
        )

        category = live_product.get(
            "category"
        )

    # Use historical information when live product details are missing.
    if not product_data.empty:
        latest_historical = (
            product_data
            .sort_values("transaction_date")
            .iloc[-1]
        )

        if (
            not product_name
            or str(product_name).lower() == "nan"
        ):
            if "product_name" in latest_historical.index:
                product_name = latest_historical[
                    "product_name"
                ]

        if (
            not category
            or str(category).lower() == "nan"
        ):
            if "category" in latest_historical.index:
                category = latest_historical[
                    "category"
                ]

            elif "product_category" in latest_historical.index:
                category = latest_historical[
                    "product_category"
                ]

    # Live inventory is always preferred for the current state.
    if live_snapshot is not None:
        current_quantity = float(
            live_snapshot.get(
                "current_quantity",
                0,
            )
        )

        raw_days = live_snapshot.get(
            "days_to_expiry",
            7,
        )

        try:
            days_to_expiry = int(
                raw_days
            )
        except Exception:
            days_to_expiry = 7

    else:
        # Fall back to the latest historical record.
        latest = (
            product_data
            .sort_values("transaction_date")
            .iloc[-1]
        )

        if (
            "remaining_inventory" in latest.index
            and pd.notna(
                latest["remaining_inventory"]
            )
        ):
            current_quantity = float(
                latest["remaining_inventory"]
            )

        else:
            initial = float(
                latest.get(
                    "initial_quantity",
                    0,
                )
            )

            sold = float(
                latest.get(
                    "units_sold",
                    0,
                )
            )

            wasted = float(
                latest.get(
                    "units_wasted",
                    0,
                )
            )

            current_quantity = max(
                0,
                initial - sold - wasted,
            )

        if (
            "days_to_expiry" in latest.index
            and pd.notna(
                latest["days_to_expiry"]
            )
        ):
            days_to_expiry = int(
                latest["days_to_expiry"]
            )

        else:
            days_to_expiry = 7

    # Keep the real expiry value, while guaranteeing at least
    # one forecast day for the forecasting model.
    actual_days_to_expiry = days_to_expiry

    forecast_days = max(
        1,
        min(
            max(
                1,
                days_to_expiry,
            ),
            30,
        ),
    )

    predicted_demand = predict_total_demand(
        product_id=product_id,
        days=forecast_days,
    )

    slow_moving_score = calculate_slow_moving_score(
        product_id=product_id,
        product_data=product_data,
        predicted_demand=predicted_demand,
        forecast_days=forecast_days,
    )

    historical_waste_score = (
        calculate_historical_waste_score(
            historical_df=historical_df,
            product_data=product_data,
            category=category,
        )
    )

    risk_days = max(
        0,
        actual_days_to_expiry,
    )

    result = calculate_waste_risk(
        current_quantity=current_quantity,
        predicted_demand=predicted_demand,
        days_to_expiry=risk_days,
        slow_moving_score=slow_moving_score,
        historical_waste_score=historical_waste_score,
    )

    result["product_id"] = product_id

    if product_name:
        result["product_name"] = str(
            product_name
        )

    if category:
        result["category"] = str(
            category
        )

    live_sales = get_recent_live_sales_summary(
        product_id,
        days=7,
    )

    result["recent_live_sales_7d"] = round(
        live_sales["total_live_sales"],
        2,
    )

    result["live_sale_days"] = live_sales[
        "live_sale_days"
    ]

    result["data_source"] = (
        "LIVE_INVENTORY"
        if live_snapshot is not None
        else "HISTORICAL_DATA"
    )

    return result


if __name__ == "__main__":
    product_id = input(
        "Enter product ID: "
    ).strip()

    result = analyze_product(
        product_id
    )

    print(
        "\n======================================"
    )
    print(
        "LIVE-AWARE WASTE RISK ANALYSIS"
    )
    print(
        "======================================"
    )

    for key, value in result.items():
        print(
            f"{key:30}: "
            f"{value}"
        )