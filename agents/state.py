from typing import (
    TypedDict,
    Any
)


class FoodWasteState(
    TypedDict,
    total=False
):

    product_id: str

    risk_result: dict

    rag_query: str

    rag_results: list[dict]

    rag_context: str

    explanation: str

    final_output: Any

    error: str