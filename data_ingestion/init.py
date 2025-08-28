"""
Data ingestion subpackage.

The purpose of this subpackage is to provide utilities for loading raw
documents, splitting them into manageable chunks and converting them into
embeddings that can be stored in a vector database.  The main entry point
for consumers of this package is :class:`DocumentIngestor`.
"""

from .ingestor import DocumentIngestor  # noqa: F401
