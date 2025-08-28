"""
ingestor.py
-------------

This module defines the :class:`DocumentIngestor`, responsible for loading
documents from various sources (directories of PDFs/Markdowns or simple strings),
splitting them into chunks and creating vector embeddings.  The ingested
embeddings can be stored in a local FAISS index or a remote Pinecone index.

The ingestion process follows the data indexing stage of a RAG pipeline, which
involves loading data, splitting it into smaller pieces, converting it to
embeddings and saving the vectors into a searchable store:contentReference[oaicite:3]{index=3}.

Example usage::

    from ai_personalized_learning_assistant.data_ingestion import DocumentIngestor

    ingestor = DocumentIngestor()
    ingestor.ingest_from_directory("./docs", vector_store_path="./index")
"""

from __future__ import annotations

import os
import pathlib
from dataclasses import dataclass
from typing import Iterable, List, Optional

from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS, Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.embeddings.base import Embeddings

try:
    from langchain.document_loaders import (
        PyPDFLoader,
        UnstructuredMarkdownLoader,
        TextLoader,
    )
except ImportError:
    # Fall back to generic DirectoryLoader components if specific loaders are not available
    from langchain.document_loaders import DirectoryLoader

__all__ = ["DocumentIngestor"]


@dataclass
class DocumentIngestor:
    """Helper class for ingesting documents and building a vector store.

    Parameters
    ----------
    embedding_model_name: str
        Name of the OpenAI embedding model to use. Defaults to ``"text-embedding-ada-002"``.
    embedding: Embeddings, optional
        An instance of a LangChain Embeddings implementation.  If provided, this
        overrides ``embedding_model_name`` and allows you to supply a custom
        embedder (useful for testing).
    use_pinecone: bool
        Whether to store the embeddings in a Pinecone index rather than a local
        FAISS index.
    pinecone_api_key: Optional[str]
        API key for Pinecone.  Required if ``use_pinecone`` is true.
    pinecone_env: Optional[str]
        Pinecone environment region.
    index_name: str
        Name of the Pinecone index to use when ``use_pinecone`` is true.
    """

    embedding_model_name: str = "text-embedding-ada-002"
    embedding: Optional[Embeddings] = None
    use_pinecone: bool = False
    pinecone_api_key: Optional[str] = None
    pinecone_env: Optional[str] = None
    index_name: str = "documents"

    def __post_init__(self) -> None:
        # Create the embedding model lazily to avoid unnecessary API calls
        if self.embedding is None:
            # The OpenAI API key is taken from the OPENAI_API_KEY environment variable
            self.embedding = OpenAIEmbeddings(model=self.embedding_model_name)

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------
    def ingest_from_directory(
        self,
        directory: str,
        *,
        vector_store_path: Optional[str] = None,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ) -> FAISS | Pinecone:
        """Load all documents from a directory, split them into chunks, embed and store.

        Parameters
        ----------
        directory: str
            Path to a directory containing documents (PDF, Markdown, text).
        vector_store_path: Optional[str], default None
            Where to persist the FAISS index on disk.  If ``None`` the index
            will exist only in memory.  When using Pinecone this parameter is
            ignored.
        chunk_size: int, default 1000
            Maximum number of characters per text chunk.  See
            :class:`RecursiveCharacterTextSplitter`.
        chunk_overlap: int, default 200
            Overlap size between chunks.

        Returns
        -------
        :class:`langchain.vectorstores.base.VectorStore`
            A vector store containing the embedded document chunks.
        """
        docs = self._load_directory(directory)
        return self._embed_and_store(docs, vector_store_path, chunk_size, chunk_overlap)

    def ingest_from_texts(
        self,
        texts: Iterable[str],
        *,
        vector_store_path: Optional[str] = None,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ) -> FAISS | Pinecone:
        """Create a vector store from an iterable of raw text strings.

        This convenience method is useful for testing because it avoids the need
        to load files from disk.  It splits the provided strings into chunks
        and embeds them.
        """
        docs = [Document(page_content=t) for t in texts]
        return self._embed_and_store(docs, vector_store_path, chunk_size, chunk_overlap)

    # ---------------------------------------------------------------------
    # Internal helpers
    # ---------------------------------------------------------------------
    def _load_directory(self, directory: str) -> List[Document]:
        """Load documents from the specified directory.

        Supports PDF, Markdown and plain text files.  Files with other
        extensions are ignored.
        """
        directory_path = pathlib.Path(directory)
        if not directory_path.exists():
            raise FileNotFoundError(f"Directory {directory} does not exist")

        documents: List[Document] = []
        for path in directory_path.rglob("*"):
            if path.is_dir():
                continue
            ext = path.suffix.lower()
            if ext == ".pdf":
                loader = PyPDFLoader(str(path))
                documents.extend(loader.load())
            elif ext in {".md", ".markdown"}:
                loader = UnstructuredMarkdownLoader(str(path))
                documents.extend(loader.load())
            elif ext in {".txt", ".text"}:
                loader = TextLoader(str(path))
                documents.extend(loader.load())
            else:
                # Unsupported file type; skip
                continue
        return documents

    def _embed_and_store(
        self,
        documents: List[Document],
        vector_store_path: Optional[str],
        chunk_size: int,
        chunk_overlap: int,
    ) -> FAISS | Pinecone:
        """Split, embed and store the supplied documents.

        If ``use_pinecone`` is True a Pinecone index will be created (and
        assumed to already exist on the Pinecone service).  Otherwise a local
        FAISS index will be created and optionally persisted to disk.
        """
        # Split documents into chunks
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        split_docs = splitter.split_documents(documents)

        if self.use_pinecone:
            # Create a Pinecone vector store
            if self.pinecone_api_key is None or self.pinecone_env is None:
                raise ValueError(
                    "Pinecone API key and environment must be provided when use_pinecone=True"
                )
            import pinecone

            pinecone.init(api_key=self.pinecone_api_key, environment=self.pinecone_env)

            if self.index_name not in pinecone.list_indexes():
                # Create an index if it doesn't exist.  Pinecone requires a dimension
                # when creating the index – we obtain it from the embedding dimension
                sample_vector = self.embedding.embed_query("sample")
                dimension = len(sample_vector)
                pinecone.create_index(self.index_name, dimension=dimension)

            vector_store = Pinecone.from_documents(
                split_docs, self.embedding, index_name=self.index_name
            )
        else:
            # Create a FAISS vector store
            vector_store = FAISS.from_documents(split_docs, self.embedding)
            # Persist to disk if a path is provided
            if vector_store_path is not None:
                os.makedirs(vector_store_path, exist_ok=True)
                faiss_path = pathlib.Path(vector_store_path) / "faiss_index"
                vector_store.save_local(str(faiss_path))
        return vector_store
