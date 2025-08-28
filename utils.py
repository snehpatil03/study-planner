"""
utils.py
---------

Miscellaneous helper classes and functions used throughout the project.  This
module currently defines simple fake embedding and LLM classes intended for
testing without incurring external API calls.
"""

from __future__ import annotations

import hashlib
import random
from typing import Iterable, List

import numpy as np
from langchain.embeddings.base import Embeddings
from langchain.llms.base import LLM
from langchain.schema import Generation, LLMResult


class FakeEmbeddings(Embeddings):
    """Simple deterministic embedding model for testing purposes.

    The fake embedder hashes the input text to produce a numeric vector.  The
    generated vectors have low dimensionality and no semantic meaning; they
    suffice for exercising the ingestion and retrieval code paths without
    contacting an external API.
    """

    def __init__(self, dimension: int = 8) -> None:
        self.dimension = dimension

    def _hash(self, text: str) -> float:
        return float(int(hashlib.md5(text.encode("utf-8")).hexdigest(), 16) % (10 ** 8))

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self.embed_query(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        random.seed(self._hash(text))
        return [random.random() for _ in range(self.dimension)]


class FakeLLM(LLM):
    """LLM that returns canned answers for testing.

    This fake LLM ignores the prompt and returns a deterministic answer based on
    a hash of the input.  It implements the minimum API required by LangChain.
    """

    def _call(self, prompt: str, stop: Iterable[str] | None = None) -> str:
        # Return the first sentence of the prompt's question as the answer
        # Fall back to a deterministic phrase if no question is found
        answer = ""
        for line in reversed(prompt.split("\n")):
            if line.lower().startswith("question:"):
                answer = line.split(":", 1)[1].strip()
                break
        if not answer:
            # Deterministic pseudorandom answer
            answer = f"Answer_{int(self._hash(prompt)) % 1000}"
        return answer

    def _hash(self, text: str) -> float:
        return float(int(hashlib.sha1(text.encode("utf-8")).hexdigest(), 16) % (10 ** 8))

    @property
    def _identifying_params(self) -> dict:
        return {}

    @property
    def _llm_type(self) -> str:
        return "fake-llm"

    def generate(
        self,
        prompts: List[str],
        stop: Iterable[str] | None = None,
    ) -> LLMResult:
        generations = [Generation(text=self._call(p)) for p in prompts]
        return LLMResult(generations=[generations], llm_output={})
