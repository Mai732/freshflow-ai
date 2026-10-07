from pathlib import Path
import sys
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


BASE_DIR = Path(__file__).resolve().parents[1]

if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))


from agents.workflow import run_product_analysis
from agents.chat_agent import chat


app = FastAPI(
    title="FreshFlow AI API",
    description="AI Food Waste Reduction System",
    version="2.0.0",
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AnalysisRequest(BaseModel):
    product_id: str


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    message: str
    history: list[ChatMessage] = Field(
        default_factory=list
    )


@app.get("/")
def root():
    return {
        "service": "FreshFlow AI",
        "status": "running",
        "version": "2.0.0",
    }


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


@app.post("/analyze")
def analyze(request: AnalysisRequest):
    try:
        result = run_product_analysis(
            request.product_id
        )

        return result.model_dump()

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=str(error),
        )


@app.post("/chat")
def chat_endpoint(request: ChatRequest):
    try:
        history = [
            message.model_dump()
            for message in request.history
        ]

        result = chat(
            message=request.message,
            history=history,
        )

        return result

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=str(error),
        )