"""
optimization.py
----------------

This module contains simple utilities for running hyper‑parameter sweeps over
the RAG pipeline.  It provides basic functions to vary the number of retrieved
documents or chunk sizes and record metrics such as latency and answer length.

Note: These functions are intentionally lightweight and avoid external
dependencies like Optuna or Ray.  They are intended as a starting point for
experimentation and can be extended with more sophisticated optimisation
libraries if needed.
"""

from __future__ import annotations

import json
import time
from typing import Any, Dict, Iterable, List, Sequence

from ..chains import RagChain


def evaluate_chain(chain: RagChain, question: str) -> Dict[str, Any]:
    """Evaluate a single question using a RAG chain and record metrics.

    Parameters
    ----------
    chain: RagChain
        The retrieval‑augmented generation chain.
    question: str
        The question to ask.

    Returns
    -------
    dict
        A dictionary with keys ``"answer"``, ``"latency"`` (seconds),
        ``"length"`` (characters in the answer) and ``"retrieved"`` (number of
        source documents).
    """
    result = chain.answer(question, return_sources=True)
    answer = result["answer"]
    latency = result.get("latency", 0.0)
    sources = result.get("sources", [])
    return {
        "answer": answer,
        "latency": latency,
        "length": len(answer),
        "retrieved": len(sources),
    }


def sweep_k(
    chain: RagChain,
    k_values: Sequence[int],
    *,
    questions: Iterable[str],
    save_path: str | None = None,
) -> List[Dict[str, Any]]:
    """Run a sweep over different retrieval depths (k) and collect metrics.

    For each ``k`` in ``k_values`` the chain’s retriever is reconfigured to
    retrieve that many documents.  Each question in ``questions`` is asked and
    the metrics are recorded.

    Parameters
    ----------
    chain: RagChain
        The chain to evaluate.
    k_values: sequence of int
        Values of ``k`` (number of retrieved documents) to test.
    questions: iterable of str
        List of questions to evaluate.
    save_path: str or None, default None
        Optional path to write the results to a JSON file.

    Returns
    -------
    list of dict
        A list of result dictionaries.  Each entry contains the tested ``k``
        value, the question and the measured metrics.
    """
    results: List[Dict[str, Any]] = []
    for k in k_values:
        # Set new k on the chain's retriever
        try:
            chain.chain.retriever.search_kwargs["k"] = k
        except Exception:
            # Fallback for older versions of LangChain where search_kwargs isn't mutable
            chain.chain.retriever = chain.chain.retriever.with_search_kwargs({"k": k})
        for question in questions:
            metrics = evaluate_chain(chain, question)
            metrics.update({"k": k, "question": question})
            results.append(metrics)
    if save_path:
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
    return results


def sweep_chunk_size(
    ingestor,
    chunk_sizes: Sequence[int],
    *,
    directory: str,
    vector_store_base: str,
    questions: Iterable[str],
    embedding,
    model_name: str = "gpt-4",
    top_k: int = 4,
    save_path: str | None = None,
) -> List[Dict[str, Any]]:
    """Example function to test different chunk sizes during ingestion.

    This function re‑ingests the documents for each chunk size, builds a new
    vector store, constructs a RAG chain and evaluates the provided questions.
    Because this process can be expensive it should only be used on small
    datasets or for illustrative purposes.
    """
    from ..chains import build_rag_chain

    results: List[Dict[str, Any]] = []
    for cs in chunk_sizes:
        # Ingest with the specified chunk size
        vector_store_path = f"{vector_store_base}_cs{cs}"
        ingestor.ingest_from_directory(directory, vector_store_path=vector_store_path, chunk_size=cs)
        # Build a new chain
        chain = build_rag_chain(
            vector_store_path=vector_store_path,
            embedding=embedding,
            model_name=model_name,
            k=top_k,
        )
        for q in questions:
            metrics = evaluate_chain(chain, q)
            metrics.update({"chunk_size": cs, "question": q})
            results.append(metrics)
    if save_path:
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
    return results
