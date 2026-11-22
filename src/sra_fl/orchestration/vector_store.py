"""FAISS-backed vector store for client information."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import faiss
import numpy as np


@dataclass
class Document:
    doc_id: str
    text: str
    metadata: dict[str, Any]


def _hash_embed(text: str, dim: int = 64) -> np.ndarray:
    """Deterministic lightweight embedding (no model download required)."""
    vec = np.zeros(dim, dtype=np.float32)
    tokens = text.lower().replace(",", " ").replace(":", " ").split()
    if not tokens:
        return vec
    for token in tokens:
        h = hash(token) % dim
        vec[h] += 1.0
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm
    return vec


class VectorStore:
    """Store and retrieve text documents with FAISS inner-product search."""

    def __init__(self, dim: int = 64):
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)
        self.documents: list[Document] = []

    def clear(self) -> None:
        self.index = faiss.IndexFlatIP(self.dim)
        self.documents = []

    def add(self, documents: list[Document]) -> None:
        if not documents:
            return
        vectors = np.vstack([_hash_embed(doc.text, self.dim) for doc in documents])
        self.index.add(vectors)
        self.documents.extend(documents)

    def search(self, query: str, top_k: int = 5) -> list[tuple[Document, float]]:
        if not self.documents:
            return []
        q = _hash_embed(query, self.dim).reshape(1, -1)
        k = min(top_k, len(self.documents))
        scores, indices = self.index.search(q, k)
        out: list[tuple[Document, float]] = []
        for score, idx in zip(scores[0], indices[0], strict=False):
            if idx < 0:
                continue
            out.append((self.documents[int(idx)], float(score)))
        return out

    def __len__(self) -> int:
        return len(self.documents)
