"""
Chain definitions for retrieval‑augmented generation.

The :mod:`ai_personalized_learning_assistant.chains` package exposes helpers
for constructing LangChain RAG chains.  These chains combine a retriever (for
fetching relevant context from a vector store) with a language model to
generate answers.
"""

from .rag_chain import build_rag_chain  # noqa: F401
