"""
app.py
------

FastAPI application exposing the retrieval‑augmented generation service.  The
server loads a RAG chain on startup and handles incoming queries by routing
them through the chain.  Basic metrics such as inference latency and number
of retrieved documents are pushed to CloudWatch via the helper in
``monitor.py``:contentReference[oaicite:0]{index=0}.
"""

from __future__ import annotations

import logging
import os
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from ..chains import build_rag_chain, RagChain
from .monitor import push_metric


logger = logging.getLogger(__name__)

app = FastAPI(title="AI‑Powered Personalized Learning Assistant")

# Global chain instance set on startup
rag_chain: Optional[RagChain] = None


class QueryRequest(BaseModel):
    """Pydantic model for the incoming query payload."""

    question: str = Field(..., description="The natural language question to answer.")


class QueryResponse(BaseModel):
    """Pydantic model for the outgoing response."""

    answer: str
    sources: Optional[List[str]] = None
    latency_ms: float


@app.on_event("startup")
def startup() -> None:
    """Initialise the retrieval chain when the application starts."""
    global rag_chain
    # Determine the location of the vector store and other settings from the environment
    vector_store_path = os.environ.get("VECTOR_STORE_PATH", "./vector_store")
    use_pinecone = os.environ.get("USE_PINECONE", "false").lower() == "true"
    index_name = os.environ.get("PINECONE_INDEX", "documents")
    pinecone_key = os.environ.get("PINECONE_API_KEY")
    pinecone_env = os.environ.get("PINECONE_ENV")
    model_name = os.environ.get("MODEL_NAME", "gpt-4")
    temperature = float(os.environ.get("TEMPERATURE", "0.0"))
    k = int(os.environ.get("TOP_K", "4"))

    try:
        rag_chain = build_rag_chain(
            vector_store_path=vector_store_path,
            use_pinecone=use_pinecone,
            index_name=index_name,
            pinecone_api_key=pinecone_key,
            pinecone_env=pinecone_env,
            model_name=model_name,
            temperature=temperature,
            k=k,
        )
        logger.info("Loaded RAG chain successfully")
    except Exception as exc:
        logger.exception("Failed to initialise RAG chain: %s", exc)
        raise


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    """Handle a user query by retrieving context and generating an answer."""
    global rag_chain
    if rag_chain is None:
        raise HTTPException(status_code=503, detail="RAG chain not initialised yet")
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question must not be empty")
    try:
        result = rag_chain.answer(question, return_sources=True)
    except Exception as exc:
        logger.exception("Error while generating answer: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))

    # Push latency metric in milliseconds
    latency_ms = result.get("latency", 0.0) * 1000
    push_metric("Latency", latency_ms)
    # Push number of retrieved documents
    sources = result.get("sources", [])
    push_metric("RetrievedCount", len(sources), unit="Count")

    return QueryResponse(
        answer=result["answer"],
        sources=sources,
        latency_ms=latency_ms,
    )
