"""
Experiment utilities for hyper‑parameter optimisation.

This subpackage provides simple routines for exploring different parameter
settings of the RAG pipeline.  For instance, you can vary the number of
documents retrieved (``k``) or the chunk size used during ingestion and
measure the impact on latency or answer quality.
"""

from .optimization import sweep_k, evaluate_chain  # noqa: F401
