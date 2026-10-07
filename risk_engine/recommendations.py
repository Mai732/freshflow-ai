from pathlib import Path
import sys


BASE_DIR = Path(__file__).resolve().parents[1]

if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))


from risk_engine.risk import analyze_product


def recommend_discount(
    risk_level,
    days_to_expiry,
    expected_surplus,
    current_quantity,
):
    if current_quantity <= 0:
        return 0

    surplus_ratio = expected_surplus / current_quantity

    if risk_level == "CRITICAL":
        if days_to_expiry <= 1:
            return 50
        elif surplus_ratio >= 0.60:
            return 40
        else:
            return 30

    elif risk_level == "HIGH":
        if days_to_expiry <= 2:
            return 30
        else:
            return 20

    elif risk_level == "MEDIUM":
        return 10

    return 0


def recommend_order_adjustment(
    expected_surplus,
    risk_level,
):
    expected_surplus = max(0, expected_surplus)

    if risk_level in [
        "CRITICAL",
        "HIGH",
    ]:
        # Reduce the next order by roughly the expected surplus
        return -int(
            round(expected_surplus)
        )

    elif risk_level == "MEDIUM":
        return -int(
            round(
                expected_surplus * 0.50
            )
        )

    return 0


def choose_action(
    risk_level,
    days_to_expiry,
    discount,
):
    if risk_level == "CRITICAL":
        return (
            "URGENT_DISCOUNT_FEFO_"
            "AND_REDUCE_ORDER"
        )

    elif risk_level == "HIGH":
        return (
            "DISCOUNT_FEFO_"
            "AND_REDUCE_ORDER"
        )

    elif risk_level == "MEDIUM":
        if discount > 0:
            return (
                "SMALL_PROMOTION_"
                "AND_MONITOR"
            )

        return "MONITOR"

    return "NO_ACTION"


def create_reason(result):
    quantity = result["current_quantity"]
    demand = result["predicted_demand"]
    surplus = result["expected_surplus"]
    days = result["days_to_expiry"]
    level = result["risk_level"]

    return (
        f"The product currently has "
        f"{quantity:.2f} units available. "
        f"Predicted demand before expiry "
        f"is approximately {demand:.2f} units, "
        f"leaving an expected surplus of "
        f"{surplus:.2f} units. "
        f"The product has {days} day(s) "
        f"until expiry, producing a "
        f"{level} waste-risk classification."
    )


def generate_recommendation(product_id):
    result = analyze_product(product_id)

    discount = recommend_discount(
        risk_level=result["risk_level"],
        days_to_expiry=result["days_to_expiry"],
        expected_surplus=result["expected_surplus"],
        current_quantity=result["current_quantity"],
    )

    order_adjustment = recommend_order_adjustment(
        expected_surplus=result["expected_surplus"],
        risk_level=result["risk_level"],
    )

    action = choose_action(
        risk_level=result["risk_level"],
        days_to_expiry=result["days_to_expiry"],
        discount=discount,
    )

    result["recommended_discount"] = discount
    result["recommended_order_adjustment"] = order_adjustment
    result["recommended_action"] = action
    result["reason"] = create_reason(result)

    return result


if __name__ == "__main__":
    product_id = input(
        "Enter product ID: "
    ).strip()

    result = generate_recommendation(
        product_id
    )

    print("\n======================================")
    print("FINAL PRODUCT RECOMMENDATION")

    for key, value in result.items():
        print(
            f"{key:36}: "
            f"{value}"
        )