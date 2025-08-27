"""
Top‑level package for the AI‑Powered Personalized Learning Assistant.

This package exposes the main public classes and functions for building
retrieval‑augmented generation pipelines using LangChain and GPT‑4.  See the
README for details on how to ingest documents, build the RAG chain and deploy
the API.
"""

# Re‑export convenient symbols for easy access
from .data_ingestion.ingestor import DocumentIngestor  # noqa: F401
from .chains.rag_chain import build_rag_chain  # noqa: F401
from .vector_store.store import get_vector_store  # noqa: F401
