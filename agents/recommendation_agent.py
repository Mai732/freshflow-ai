from pathlib import Path
import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI


BASE_DIR = (
    Path(__file__)
    .resolve()
    .parents[1]
)

load_dotenv(
    BASE_DIR / ".env"
)


def get_llm():
    api_key = os.getenv(
        "OPENAI_API_KEY"
    )

    if not api_key:
        raise ValueError(
            "\nOPENAI_API_KEY was not found.\n"
            "Add it to the .env file."
        )

    return ChatOpenAI(
        model="gpt-6-luna",
    )


def generate_explanation(
    risk_result,
    rag_context,
):
    llm = get_llm()

    prompt = f"""
You are an AI decision-support assistant
for food waste reduction in restaurants
and supermarkets.

IMPORTANT RULES:

1. Do NOT change any numerical values.
2. Do NOT invent new risk scores.
3. Do NOT invent discount percentages.
4. Do NOT invent order adjustments.
5. The deterministic Python Risk Engine
   is the source of truth for numbers.
6. Use FDA evidence only to support
   storage, handling, date-marking,
   prioritization, or waste-prevention
   reasoning.
7. Do not claim that FDA guidance
   directly recommends a specific
   discount percentage unless the
   retrieved text explicitly says so.
8. Keep the explanation concise.

RISK ENGINE RESULT:

{risk_result}

FDA RAG CONTEXT:

{rag_context}

Write a concise professional explanation
for why the recommendation is appropriate.

Mention:
- current inventory
- predicted demand
- expected surplus
- time to expiry
- waste-risk level
- operational recommendation
- relevant FDA guidance when applicable
"""

    response = llm.invoke(
        prompt
    )

    return response.content