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
        """Search for similar chunks."""
        ...

    @abstractmethod
    async def delete_document(self, document_id: str) -> bool:
        """Delete all chunks for a document."""
        ...

    @abstractmethod
    async def count(self) -> int:
        """Total chunk count."""
        ...
