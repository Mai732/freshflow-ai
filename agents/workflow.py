from pathlib import Path
import sys

from langgraph.graph import (
    StateGraph,
    START,
    END,
)


BASE_DIR = Path(__file__).resolve().parents[1]

if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))


from agents.state import FoodWasteState
from agents.recommendation_agent import generate_explanation
from risk_engine.recommendations import generate_recommendation
from rag.retriever import retrieve_guidance, format_guidance
from schemas.output import build_output


def risk_node(state: FoodWasteState):
    product_id = state["product_id"]

    result = generate_recommendation(
        product_id
    )

    return {
        "risk_result": result
    }


def rag_query_node(state: FoodWasteState):
    result = state["risk_result"]

    product = result.get(
        "product_name",
        result["product_id"],
    )

    category = result.get(
        "category",
        "food product",
    )

    days = result["days_to_expiry"]
    risk = result["risk_level"]

    query = (
        f"FDA guidance relevant to "
        f"{category} food such as {product}, "
        f"with {days} days until expiration "
        f"and {risk} waste risk. "
        f"Retrieve guidance about safe storage, "
        f"date marking, temperature control, "
        f"handling, rotation, holding, "
        f"and disposal where relevant."
    )

    return {
        "rag_query": query
    }


def retrieval_node(state: FoodWasteState):
    query = state["rag_query"]

    results = retrieve_guidance(
        query,
        top_k=5,
    )

    context = format_guidance(
        results
    )

    return {
        "rag_results": results,
        "rag_context": context,
    }


def explanation_node(state: FoodWasteState):
    explanation = generate_explanation(
        risk_result=state["risk_result"],
        rag_context=state["rag_context"],
    )

    return {
        "explanation": explanation
    }


def output_node(state: FoodWasteState):
    rag_evidence = []

    for item in state["rag_results"]:
        evidence = (
            f"{item['source']} "
            f"(page {item['page']}): "
            f"{item['text'][:350]}"
        )

        rag_evidence.append(
            evidence
        )

    final_output = build_output(
        risk_result=state["risk_result"],
        reason=state["explanation"],
        rag_evidence=rag_evidence,
    )

    return {
        "final_output": final_output
    }


def build_workflow():
    workflow = StateGraph(
        FoodWasteState
    )

    workflow.add_node(
        "risk_engine",
        risk_node,
    )

    workflow.add_node(
        "build_rag_query",
        rag_query_node,
    )

    workflow.add_node(
        "retrieve_guidance",
        retrieval_node,
    )

    workflow.add_node(
        "generate_explanation",
        explanation_node,
    )

    workflow.add_node(
        "validate_output",
        output_node,
    )

    workflow.add_edge(
        START,
        "risk_engine",
    )

    workflow.add_edge(
        "risk_engine",
        "build_rag_query",
    )

    workflow.add_edge(
        "build_rag_query",
        "retrieve_guidance",
    )

    workflow.add_edge(
        "retrieve_guidance",
        "generate_explanation",
    )

    workflow.add_edge(
        "generate_explanation",
        "validate_output",
    )

    workflow.add_edge(
        "validate_output",
        END,
    )

    return workflow.compile()


def run_product_analysis(product_id):
    app = build_workflow()

    result = app.invoke(
        {
            "product_id": product_id
        }
    )

    return result["final_output"]


if __name__ == "__main__":
    product_id = input(
        "Enter product ID: "
    ).strip()

    result = run_product_analysis(
        product_id
    )

    print(
        "\n======================================"
    )

    print(
        "FINAL AGENT OUTPUT"
    )

    print(
        result.model_dump_json(
            indent=2
        )
    )