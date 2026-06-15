from abc import ABC, abstractmethod

from backend.domain.document import DocumentChunk
from backend.domain.retrieval import RetrievedChunk


class VectorStore(ABC):
    @abstractmethod
    async def insert_chunks(self, chunks: list[DocumentChunk]) -> int:
        """Insert chunks and return count inserted."""
        ...

    @abstractmethod
    async def search(
        self,
        embedding: list[float],
        top_k: int = 10,
        score_threshold: float = 0.0,
        filters: dict | None = None,
    ) -> list[RetrievedChunk]:
        """Dense vector search."""
        ...

    @abstractmethod
    async def sparse_search(
        self,
        sparse_embedding: dict[int, float],
        top_k: int = 10,
        score_threshold: float = 0.0,
        filters: dict | None = None,
    ) -> list[RetrievedChunk]:
        """Sparse vector search (BM25)."""
        ...

    @abstractmethod
    async def hybrid_search(
        self,
        dense_embedding: list[float],
        sparse_embedding: dict[int, float],
        top_k: int = 10,
        dense_weight: float = 0.7,
        sparse_weight: float = 0.3,
        score_threshold: float = 0.0,
        filters: dict | None = None,
    ) -> list[RetrievedChunk]:
        """Hybrid dense + sparse search with weighted fusion."""
        ...

    @abstractmethod
    async def delete_document(self, document_id: str) -> bool:
        """Delete all chunks for a document."""
        ...

    @abstractmethod
    async def count(self) -> int:
        """Total chunk count."""
        ...
