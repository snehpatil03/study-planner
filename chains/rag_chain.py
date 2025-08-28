"""
rag_chain.py
-------------

Utilities for building and interacting with a retrieval‑augmented generation
(RAG) pipeline.  A RAG pipeline uses a two‑phase process: it first retrieves
relevant information from an external knowledge base (a vector store), then
generates an answer using a large language model:contentReference[oaicite:2]{index=2}.  This module
wraps the necessary LangChain primitives into a simple builder function.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, Optional

from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.embeddings.base import Embeddings

from ..vector_store import store as vs_store


def default_prompt() -> PromptTemplate:
    """Return a default prompt template for the RAG pipeline.

    The prompt instructs the model to answer using only the provided context.
    If the answer cannot be determined from the context, the model should
    indicate that it doesn’t know.
    """
    template = (
        "You are an intelligent assistant. Use the following pieces of context to "
        "answer the question at the end. If the answer is not contained within "
        "the context, say 'I don't know'.\n\n"
        "{context}\n\n"
        "Question: {question}\n"
        "Helpful answer:"
    )
    return PromptTemplate(input_variables=["context", "question"], template=template)


@dataclass
class RagChain:
    """Lightweight wrapper around LangChain's RetrievalQA chain.

    Parameters
    ----------
    chain: RetrievalQA
        The underlying LangChain chain.
    """

    chain: RetrievalQA

    def answer(self, question: str, *, return_sources: bool = False) -> Dict[str, Any]:
        """Answer a question using the retrieval‑augmented chain.

        Parameters
        ----------
        question: str
            The user's question.
        return_sources: bool, default False
            Whether to include the source documents used to construct the answer.

        Returns
        -------
        dict
            A dictionary containing at least the key ``"answer"``.  If
            ``return_sources`` is true, a ``"sources"`` key will also be
            included containing the retrieved documents.
        """
        start = time.perf_counter()
        response = self.chain({"query": question})
        latency = time.perf_counter() - start
        answer = response["result"]
        result: Dict[str, Any] = {"answer": answer, "latency": latency}
        if return_sources:
            # Include the page_content for each source document
            sources = [doc.page_content for doc in response.get("source_documents", [])]
            result["sources"] = sources
        return result


def build_rag_chain(
    *,
    vector_store_path: Optional[str] = None,
    use_pinecone: bool = False,
    index_name: str = "documents",
    embedding: Optional[Embeddings] = None,
    pinecone_api_key: Optional[str] = None,
    pinecone_env: Optional[str] = None,
    model_name: str = "gpt-4",
    temperature: float = 0.0,
    k: int = 4,
    prompt: Optional[PromptTemplate] = None,
) -> RagChain:
    """Construct a retrieval‑augmented generation chain.

    This helper hides away the details of loading the vector store and
    constructing the LangChain ``RetrievalQA`` object.  It returns a
    :class:`RagChain` wrapper with a simple ``answer()`` method.

    Parameters
    ----------
    vector_store_path: str or None
        Path to the FAISS index directory.  Ignored if ``use_pinecone`` is true.
    use_pinecone: bool, default False
        Whether to load a Pinecone index instead of a local FAISS index.
    index_name: str
        Name of the Pinecone index when ``use_pinecone`` is true.
    embedding: Embeddings, optional
        The embedding model to use.  If ``None`` a new OpenAIEmbeddings instance
        will be created.
    pinecone_api_key: Optional[str]
        Pinecone API key required when ``use_pinecone`` is true.
    pinecone_env: Optional[str]
        Pinecone environment region required when ``use_pinecone`` is true.
    model_name: str, default "gpt-4"
        Name of the OpenAI chat model to use.
    temperature: float, default 0.0
        Sampling temperature for the language model.  A low temperature leads to
        more deterministic responses.
    k: int, default 4
        Number of documents to retrieve from the vector store.
    prompt: PromptTemplate, optional
        Custom prompt template.  If ``None``, a default template is used.

    Returns
    -------
    RagChain
        A wrapper exposing an ``answer()`` method.
    """
    # Ensure an embedding model is available
    if embedding is None:
        from langchain.embeddings.openai import OpenAIEmbeddings

        embedding = OpenAIEmbeddings()

    # Load the vector store
    vector_store = vs_store.get_vector_store(
        vector_store_path=vector_store_path,
        use_pinecone=use_pinecone,
        index_name=index_name,
        embedding=embedding,
        pinecone_api_key=pinecone_api_key,
        pinecone_env=pinecone_env,
    )

    # Build the retriever
    retriever = vector_store.as_retriever(search_kwargs={"k": k})

    # Prepare the prompt template
    if prompt is None:
        prompt = default_prompt()

    # Configure the language model
    llm = ChatOpenAI(model=model_name, temperature=temperature)

    # Compose the RetrievalQA chain
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt},
    )
    return RagChain(chain=chain)
