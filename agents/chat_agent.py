from pathlib import Path
import re
import sys
from typing import Literal, Optional

from pydantic import BaseModel, Field


BASE_DIR = Path(__file__).resolve().parents[1]

if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))


from agents.recommendation_agent import get_llm

from agents.tools import (
    analyze_product_tool,
    discount_candidates_tool,
    expiring_products_tool,
    fda_search_tool,
    highest_risk_products_tool,
    predict_demand_tool,
)


class RouteDecision(BaseModel):
    intent: Literal[
        "ANALYZE_PRODUCT",
        "PREDICT_DEMAND",
        "EXPIRING_PRODUCTS",
        "HIGH_RISK_PRODUCTS",
        "DISCOUNT_CANDIDATES",
        "FDA_SEARCH",
        "HELP",
    ]

    product_id: Optional[str] = None

    days: int = Field(
        default=3,
        ge=1,
        le=30,
    )

    limit: int = Field(
        default=5,
        ge=1,
        le=10,
    )

    search_query: Optional[str] = None


def extract_product_id(message):
    match = re.findall(
        r"\b[A-Za-z0-9]+_"
        r"[A-Za-z0-9]+_"
        r"[A-Za-z0-9]+\b",
        message,
    )

    return match[0] if match else None


def extract_days(
    message,
    default=3,
):
    patterns = [
        r"within\s+(\d+)\s+days?",
        r"next\s+(\d+)\s+days?",
        r"for\s+(\d+)\s+days?",
        r"(\d+)\s+days?",
    ]

    lower = message.lower()

    for pattern in patterns:
        match = re.search(
            pattern,
            lower,
        )

        if match:
            return min(
                max(
                    int(match.group(1)),
                    1,
                ),
                30,
            )

    return default


def fallback_route(message):
    lower = message.lower()

    product_id = extract_product_id(
        message
    )

    days = extract_days(
        message
    )

    if (
        "discount" in lower
        or "promotion" in lower
    ):
        return RouteDecision(
            intent="DISCOUNT_CANDIDATES"
        )

    if any(
        word in lower
        for word in [
            "expire",
            "expiration",
            "expiry",
        ]
    ):
        if product_id:
            return RouteDecision(
                intent="ANALYZE_PRODUCT",
                product_id=product_id,
            )

        return RouteDecision(
            intent="EXPIRING_PRODUCTS",
            days=days,
            limit=10,
        )

    if (
        "highest risk" in lower
        or "waste risk" in lower
        or "most likely" in lower
    ):
        return RouteDecision(
            intent="HIGH_RISK_PRODUCTS"
        )

    if any(
        word in lower
        for word in [
            "predict",
            "forecast",
            "demand",
        ]
    ):
        return RouteDecision(
            intent="PREDICT_DEMAND",
            product_id=product_id,
            days=days,
        )

    if any(
        word in lower
        for word in [
            "fda",
            "storage",
            "food code",
            "temperature",
            "date marking",
            "food safety",
        ]
    ):
        return RouteDecision(
            intent="FDA_SEARCH",
            search_query=message,
        )

    if product_id:
        return RouteDecision(
            intent="ANALYZE_PRODUCT",
            product_id=product_id,
        )

    return RouteDecision(
        intent="HELP"
    )


def route_message(
    message,
    history=None,
):
    history = history or []

    recent_history = "\n".join(
        f"{item.get('role')}: "
        f"{item.get('content')}"
        for item in history[-6:]
    )

    prompt = f"""
Route this supermarket food-waste question.

Possible intents:

ANALYZE_PRODUCT:
full analysis of one specific product.

PREDICT_DEMAND:
forecast one product.

EXPIRING_PRODUCTS:
find products expiring soon.

HIGH_RISK_PRODUCTS:
find products with highest waste risk.

DISCOUNT_CANDIDATES:
find products that should be discounted.

FDA_SEARCH:
questions about FDA Food Code, storage,
temperature, date marking, handling or disposal.

HELP:
general greeting or explanation of system abilities.

Do not invent product IDs.

Recent conversation:
{recent_history}

Current user message:
{message}
"""

    try:
        llm = get_llm()

        router = llm.with_structured_output(
            RouteDecision
        )

        return router.invoke(
            prompt
        )

    except Exception:
        return fallback_route(
            message
        )


def answer_fda_question(
    question,
    sources,
):
    if not sources:
        return (
            "I could not find relevant FDA "
            "guidance for that question."
        )

    evidence = "\n\n".join(
        (
            f"Source {index}: "
            f"{item['source']} "
            f"page {item['page']}\n"
            f"{item['text']}"
        )
        for index, item in enumerate(
            sources,
            start=1,
        )
    )

    llm = get_llm()

    prompt = f"""
Answer the user's question using only the FDA
Food Code evidence below.

Do not invent regulations.

Question:
{question}

Evidence:
{evidence}

Give a clear and concise answer and mention
relevant page numbers.
"""

    return llm.invoke(
        prompt
    ).content


def chat(
    message,
    history=None,
):
    decision = route_message(
        message,
        history,
    )

    intent = decision.intent

    if intent == "ANALYZE_PRODUCT":
        product_id = (
            decision.product_id
            or extract_product_id(
                message
            )
        )

        if not product_id:
            return {
                "intent": intent,
                "answer": "Please give me the product ID.",
                "data": None,
                "sources": [],
            }

        result = analyze_product_tool(
            product_id
        )

        return {
            "intent": intent,
            "answer": result["reason"],
            "data": result,
            "sources": result.get(
                "rag_evidence",
                [],
            ),
        }

    if intent == "PREDICT_DEMAND":
        product_id = (
            decision.product_id
            or extract_product_id(
                message
            )
        )

        if not product_id:
            return {
                "intent": intent,
                "answer": (
                    "Please tell me which product "
                    "you want to forecast."
                ),
                "data": None,
                "sources": [],
            }

        result = predict_demand_tool(
            product_id,
            decision.days or 7,
        )

        return {
            "intent": intent,
            "answer": (
                f"Predicted demand for "
                f"{product_id} over the next "
                f"{result['forecast_days']} "
                f"day(s) is approximately "
                f"{result['total_predicted_demand']:.2f} "
                f"units."
            ),
            "data": result,
            "sources": [],
        }

    if intent == "EXPIRING_PRODUCTS":
        results = expiring_products_tool(
            decision.days,
            decision.limit,
        )

        return {
            "intent": intent,
            "answer": (
                f"I found {len(results)} "
                f"live inventory batch(es) "
                f"expiring within "
                f"{decision.days} day(s)."
            ),
            "data": results,
            "sources": [],
        }

    if intent == "HIGH_RISK_PRODUCTS":
        results = highest_risk_products_tool(
            decision.limit
        )

        return {
            "intent": intent,
            "answer": (
                f"These are the "
                f"{len(results)} highest-risk "
                f"products currently identified."
            ),
            "data": results,
            "sources": [],
        }

    if intent == "DISCOUNT_CANDIDATES":
        results = discount_candidates_tool(
            decision.limit
        )

        return {
            "intent": intent,
            "answer": (
                f"I found {len(results)} "
                "product(s) that meet the "
                "discount rules."
            ),
            "data": results,
            "sources": [],
        }

    if intent == "FDA_SEARCH":
        sources = fda_search_tool(
            decision.search_query
            or message,
            top_k=5,
        )

        return {
            "intent": intent,
            "answer": answer_fda_question(
                message,
                sources,
            ),
            "data": None,
            "sources": sources,
        }

    return {
        "intent": "HELP",
        "answer": (
            "I can forecast demand, analyze "
            "food-waste risk, find products "
            "approaching expiration, identify "
            "discount candidates, recommend "
            "order adjustments, and search "
            "FDA Food Code guidance."
        ),
        "data": None,
        "sources": [],
    }