"""Text embeddings for semantic search, behind a tiny interface.

Lesson (ports & adapters): the service depends on the ``Embedder`` protocol, not on a
vendor. Swapping fastembed for a hosted API, or for a fake in tests, touches one file.
"""

from __future__ import annotations

import asyncio
import hashlib
import math
from typing import Protocol

DIMENSIONS = 384  # must match vector(384) in migrations/001_init.sql


class Embedder(Protocol):
    dimensions: int

    async def embed(self, text: str) -> list[float]: ...


class FastEmbedEmbedder:
    """Runs a small ONNX model locally on CPU: free, private, no API key.

    The model (~70 MB) downloads from Hugging Face on first use and is cached.
    """

    def __init__(self, model_name: str):
        from fastembed import TextEmbedding  # imported lazily: optional dependency

        self._model = TextEmbedding(model_name)
        self.dimensions = DIMENSIONS

    async def embed(self, text: str) -> list[float]:
        # The model call is CPU-bound and synchronous; keep it off the event loop
        # so one embedding doesn't stall every other request on this worker.
        vectors = await asyncio.to_thread(lambda: list(self._model.embed([text])))
        vector = [float(x) for x in vectors[0]]
        if len(vector) != self.dimensions:
            raise RuntimeError(
                f"Model returned {len(vector)} dims; the database column expects {self.dimensions}."
            )
        return vector


class HashingEmbedder:
    """Deterministic bag-of-words hashing vectors. For tests and offline demos only.

    It captures word overlap, not meaning, but it exercises the exact same code path
    (pgvector column, HNSW index, RRF fusion) as a real model.
    """

    dimensions = DIMENSIONS

    async def embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dimensions
        for word in text.lower().split():
            h = int.from_bytes(hashlib.sha256(word.strip(".,!?;:").encode()).digest()[:4], "big")
            vec[h % self.dimensions] += 1.0
        norm = math.sqrt(sum(x * x for x in vec)) or 1.0
        return [x / norm for x in vec]


def build_embedder(kind: str, model_name: str) -> Embedder | None:
    if kind == "fastembed":
        return FastEmbedEmbedder(model_name)
    if kind == "hashing":
        return HashingEmbedder()
    return None
