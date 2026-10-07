from pathlib import Path
import sys
from functools import lru_cache

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]

if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))


from agents.workflow import run_product_analysis
from forecasting.predict import predict_next_days
from risk_engine.risk import analyze_product

from risk_engine.recommendations import (
    recommend_discount,
    recommend_order_adjustment,
    choose_action,
)

from rag.retriever import retrieve_guidance

from live_inventory import (
    get_live_inventory_snapshot,
    get_all_products,
)


HISTORICAL_DATA_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "cleaned_data.csv"
)


@lru_cache(maxsize=1)
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


def get_historical_snapshot():
    df = load_historical_data().copy()

    if df.empty:
        return pd.DataFrame()

    df = df.sort_values(
        "transaction_date"
    )

    latest = (
        df.groupby(
            "product_id",
            as_index=False,
        )
        .tail(1)
        .copy()
    )

    # Estimate the current quantity from the latest historical record.
    if "remaining_inventory" in latest.columns:
        latest["current_quantity"] = pd.to_numeric(
            latest["remaining_inventory"],
            errors="coerce",
        ).fillna(0)

    elif "initial_quantity" in latest.columns:
        initial = pd.to_numeric(
            latest["initial_quantity"],
            errors="coerce",
        ).fillna(0)

        sold = pd.to_numeric(
            latest.get(
                "units_sold",
                0,
            ),
            errors="coerce",
        ).fillna(0)

        if "units_wasted" in latest.columns:
            wasted = pd.to_numeric(
                latest["units_wasted"],
                errors="coerce",
            ).fillna(0)
        else:
            wasted = 0

        latest["current_quantity"] = (
            initial
            - sold
            - wasted
        ).clip(lower=0)

    else:
        latest["current_quantity"] = 0

    if "days_to_expiry" in latest.columns:
        latest["days_to_expiry"] = pd.to_numeric(
            latest["days_to_expiry"],
            errors="coerce",
        )

    elif "expiration_date" in latest.columns:
        latest["expiration_date"] = pd.to_datetime(
            latest["expiration_date"],
            errors="coerce",
        )

        latest["days_to_expiry"] = (
            latest["expiration_date"]
            - latest["transaction_date"]
        ).dt.days

    else:
        latest["days_to_expiry"] = None

    latest["data_source"] = "HISTORICAL"

    return latest


def get_latest_snapshot():
    historical = get_historical_snapshot()
    live = get_live_inventory_snapshot()

    if not live.empty:
        live = live.copy()

        live["product_id"] = (
            live["product_id"]
            .astype(str)
            .str.strip()
        )

        live["data_source"] = "LIVE"

    if live.empty:
        return historical

    if historical.empty:
        return live

    # A live record replaces the historical current-state row for
    # the same product, while the historical dataset remains untouched.
    live_ids = set(
        live["product_id"]
        .astype(str)
        .tolist()
    )

    historical_without_live = historical[
        ~historical["product_id"]
        .astype(str)
        .isin(live_ids)
    ].copy()

    combined = pd.concat(
        [
            historical_without_live,
            live,
        ],
        ignore_index=True,
        sort=False,
    )

    return combined


def analyze_product_tool(product_id):
    result = run_product_analysis(
        product_id
    )

    return result.model_dump()


def predict_demand_tool(
    product_id,
    days=7,
):
    days = max(
        1,
        min(
            int(days),
            30,
        ),
    )

    forecast = predict_next_days(
        product_id=product_id,
        days=days,
    )

    records = forecast.copy()
    records["date"] = (
        records["date"]
        .astype(str)
    )

    total = float(
        forecast["predicted_demand"]
        .sum()
    )

    return {
        "product_id": str(product_id),
        "forecast_days": days,
        "total_predicted_demand": round(
            total,
            2,
        ),
        "daily_forecast": records.to_dict(
            orient="records"
        ),
    }


def expiring_products_tool(
    days=3,
    limit=10,
):
    from live_inventory import (
        load_live_inventory,
        get_all_products,
    )

    days = max(
        1,
        min(
            int(days),
            30,
        ),
    )

    limit = max(
        1,
        min(
            int(limit),
            50,
        ),
    )

    live_inventory = (
        load_live_inventory()
        .copy()
    )

    live_results = []

    if not live_inventory.empty:
        live_inventory["quantity"] = pd.to_numeric(
            live_inventory["quantity"],
            errors="coerce",
        ).fillna(0)

        live_inventory["expiration_date"] = pd.to_datetime(
            live_inventory["expiration_date"],
            errors="coerce",
        )

        today = (
            pd.Timestamp.today()
            .normalize()
        )

        live_inventory["days_to_expiry"] = (
            live_inventory["expiration_date"]
            - today
        ).dt.days

        # Keep only stock that is still available.
        live_inventory = live_inventory[
            live_inventory["quantity"] > 0
        ].copy()

        # Find batches expiring between today and the requested horizon.
        expiring_live = live_inventory[
            live_inventory["days_to_expiry"].between(
                0,
                days,
                inclusive="both",
            )
        ].copy()

        if not expiring_live.empty:
            products = get_all_products()

            if not products.empty:
                useful_columns = [
                    column
                    for column in [
                        "product_id",
                        "product_name",
                        "category",
                    ]
                    if column in products.columns
                ]

                expiring_live = expiring_live.merge(
                    products[
                        useful_columns
                    ].drop_duplicates(
                        subset=["product_id"],
                        keep="last",
                    ),
                    on="product_id",
                    how="left",
                )

            # Batches are kept separate because they may expire on different dates.
            expiring_live["data_source"] = "LIVE"

            expiring_live = expiring_live.sort_values(
                [
                    "days_to_expiry",
                    "expiration_date",
                ],
                ascending=[
                    True,
                    True,
                ],
            )

            for _, row in expiring_live.iterrows():
                item = {
                    "product_id": str(
                        row["product_id"]
                    ),
                    "batch_id": str(
                        row.get(
                            "batch_id",
                            "",
                        )
                    ),
                    "current_quantity": float(
                        row["quantity"]
                    ),
                    "expiration_date": (
                        row["expiration_date"]
                        .date()
                        .isoformat()
                    ),
                    "days_to_expiry": int(
                        row["days_to_expiry"]
                    ),
                    "data_source": "LIVE",
                }

                if (
                    "product_name" in row.index
                    and pd.notna(
                        row["product_name"]
                    )
                ):
                    item["product_name"] = str(
                        row["product_name"]
                    )

                if (
                    "category" in row.index
                    and pd.notna(
                        row["category"]
                    )
                ):
                    item["category"] = str(
                        row["category"]
                    )

                live_results.append(item)

    if live_results:
        return live_results[:limit]

    # Fall back to historical data when no live batches match.
    snapshot = get_historical_snapshot().copy()

    if snapshot.empty:
        return []

    snapshot["current_quantity"] = pd.to_numeric(
        snapshot["current_quantity"],
        errors="coerce",
    ).fillna(0)

    snapshot["days_to_expiry"] = pd.to_numeric(
        snapshot["days_to_expiry"],
        errors="coerce",
    )

    snapshot = snapshot[
        snapshot["current_quantity"] > 0
    ]

    snapshot = snapshot[
        snapshot["days_to_expiry"].notna()
    ]

    snapshot = snapshot[
        snapshot["days_to_expiry"].between(
            0,
            days,
            inclusive="both",
        )
    ]

    snapshot = (
        snapshot.sort_values(
            [
                "days_to_expiry",
                "current_quantity",
            ],
            ascending=[
                True,
                False,
            ],
        )
        .head(limit)
    )

    columns = [
        "product_id",
        "current_quantity",
        "days_to_expiry",
    ]

    for column in [
        "product_name",
        "category",
        "product_category",
        "expiration_date",
        "data_source",
    ]:
        if column in snapshot.columns:
            columns.append(column)

    result = snapshot[
        columns
    ].copy()

    if "expiration_date" in result.columns:
        result["expiration_date"] = (
            result["expiration_date"]
            .astype(str)
        )

    return result.to_dict(
        orient="records"
    )


def get_risk_candidate_ids(count=15):
    snapshot = (
        get_latest_snapshot()
        .copy()
    )

    if snapshot.empty:
        return []

    snapshot["current_quantity"] = pd.to_numeric(
        snapshot["current_quantity"],
        errors="coerce",
    ).fillna(0)

    snapshot = snapshot[
        snapshot["current_quantity"] > 0
    ]

    snapshot["days_to_expiry_sort"] = pd.to_numeric(
        snapshot.get(
            "days_to_expiry",
            9999,
        ),
        errors="coerce",
    ).fillna(9999)

    snapshot = snapshot.sort_values(
        [
            "days_to_expiry_sort",
            "current_quantity",
        ],
        ascending=[
            True,
            False,
        ],
    )

    return (
        snapshot["product_id"]
        .astype(str)
        .head(count)
        .tolist()
    )


def highest_risk_products_tool(limit=5):
    limit = max(
        1,
        min(
            int(limit),
            10,
        ),
    )

    candidate_count = min(
        20,
        max(
            10,
            limit * 2,
        ),
    )

    product_ids = get_risk_candidate_ids(
        candidate_count
    )

    results = []

    for product_id in product_ids:
        try:
            result = analyze_product(
                product_id
            )

            results.append(result)

        except Exception as error:
            print(
                f"Skipping {product_id}: "
                f"{error}"
            )

    results = sorted(
        results,
        key=lambda item: item.get(
            "risk_score",
            0,
        ),
        reverse=True,
    )

    return results[:limit]


def discount_candidates_tool(limit=5):
    limit = max(
        1,
        min(
            int(limit),
            10,
        ),
    )

    risk_results = highest_risk_products_tool(
        limit=max(
            8,
            limit * 2,
        )
    )

    candidates = []

    for result in risk_results:
        discount = recommend_discount(
            risk_level=result["risk_level"],
            days_to_expiry=result["days_to_expiry"],
            expected_surplus=result["expected_surplus"],
            current_quantity=result["current_quantity"],
        )

        if discount <= 0:
            continue

        adjustment = recommend_order_adjustment(
            expected_surplus=result["expected_surplus"],
            risk_level=result["risk_level"],
        )

        action = choose_action(
            risk_level=result["risk_level"],
            days_to_expiry=result["days_to_expiry"],
            discount=discount,
        )

        candidate = result.copy()

        candidate["recommended_discount"] = discount
        candidate["recommended_order_adjustment"] = adjustment
        candidate["recommended_action"] = action

        candidates.append(candidate)

    return candidates[:limit]


def fda_search_tool(
    query,
    top_k=5,
):
    top_k = max(
        1,
        min(
            int(top_k),
            8,
        ),
    )

    results = retrieve_guidance(
        query,
        top_k=top_k,
    )

    cleaned = []

    for item in results:
        cleaned.append(
            {
                "source": item.get("source"),
                "page": item.get("page"),
                "text": item.get(
                    "text",
                    "",
                ),
                "distance": item.get(
                    "distance"
                ),
            }
        )

    return cleaned