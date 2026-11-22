"""RAG engine: index client records and retrieve relevant context."""

from __future__ import annotations

from typing import Any

from sra_fl.orchestration.vector_store import Document, VectorStore
from sra_fl.utils.logger import get_logger

logger = get_logger(__name__)


class ClientInfoStore:
    """Structured client information store used by RAG."""

    def __init__(self) -> None:
        self.records: dict[int, dict[str, Any]] = {}

    def upsert(self, client_id: int, record: dict[str, Any]) -> None:
        existing = self.records.get(client_id, {})
        existing.update(record)
        existing["client_id"] = int(client_id)
        self.records[int(client_id)] = existing

    def get(self, client_id: int) -> dict[str, Any]:
        return dict(self.records.get(int(client_id), {"client_id": int(client_id)}))

    def all_records(self) -> list[dict[str, Any]]:
        return [dict(v) for _, v in sorted(self.records.items())]

    def to_document(self, client_id: int) -> Document:
        rec = self.get(client_id)
        text = (
            f"client {client_id} device_tier={rec.get('device_tier')} "
            f"network_tier={rec.get('network_tier')} "
            f"computation={rec.get('computation')} bandwidth_kbps={rec.get('bandwidth_kbps')} "
            f"reputation={rec.get('reputation')} available={rec.get('available')} "
            f"privacy_remaining={rec.get('privacy_remaining')} "
            f"history={rec.get('history', '')}"
        )
        return Document(doc_id=f"client-{client_id}", text=text, metadata=rec)


class RAGEngine:
    """Build searchable representations and retrieve client context."""

    def __init__(self, store: ClientInfoStore | None = None):
        self.store = store or ClientInfoStore()
        self.vector_store = VectorStore()

    def reindex(self) -> None:
        self.vector_store.clear()
        docs = [self.store.to_document(cid) for cid in self.store.records]
        self.vector_store.add(docs)
        logger.info("RAG reindexed %s client documents", len(docs))

    def retrieve(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        hits = self.vector_store.search(query, top_k=top_k)
        return [
            {"score": score, "text": doc.text, **doc.metadata}
            for doc, score in hits
        ]
