from typing import Literal

from pydantic import BaseModel, Field


class ProductRecommendation(BaseModel):
    product_id: str
    product_name: str | None = None
    category: str | None = None

    current_quantity: float = Field(ge=0)
    predicted_demand: float = Field(ge=0)
    expected_surplus: float = Field(ge=0)
    days_to_expiry: int = Field(ge=0)

    expiration_risk: Literal[
        "LOW",
        "MEDIUM",
        "HIGH",
        "CRITICAL",
    ]

    waste_risk_score: float = Field(
        ge=0,
        le=1,
    )

    waste_risk: Literal[
        "LOW",
        "MEDIUM",
        "HIGH",
        "CRITICAL",
    ]

    recommended_discount: int = Field(
        ge=0,
        le=100,
    )

    recommended_order_adjustment: int
    recommended_action: str
    reason: str
    rag_evidence: list[str] = []


# Convert days remaining into an expiration-risk label
def classify_expiration_risk(days_to_expiry):
    if days_to_expiry <= 1:
        return "CRITICAL"

    elif days_to_expiry <= 3:
        return "HIGH"

    elif days_to_expiry <= 7:
        return "MEDIUM"

    else:
        return "LOW"


# Build the final validated recommendation object
def build_output(
    risk_result,
    reason,
    rag_evidence,
):
    return ProductRecommendation(
        product_id=risk_result["product_id"],
        product_name=risk_result.get("product_name"),
        category=risk_result.get("category"),

        current_quantity=risk_result["current_quantity"],
        predicted_demand=risk_result["predicted_demand"],
        expected_surplus=risk_result["expected_surplus"],
        days_to_expiry=risk_result["days_to_expiry"],

        expiration_risk=classify_expiration_risk(
            risk_result["days_to_expiry"]
        ),

        waste_risk_score=risk_result["risk_score"],
        waste_risk=risk_result["risk_level"],

        recommended_discount=risk_result[
            "recommended_discount"
        ],

        recommended_order_adjustment=risk_result[
            "recommended_order_adjustment"
        ],

        recommended_action=risk_result[
            "recommended_action"
        ],

        reason=reason,
        rag_evidence=rag_evidence,
    )