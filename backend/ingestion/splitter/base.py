from abc import ABC, abstractmethod

from backend.domain.document import DocumentChunk


class Splitter(ABC):
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 64):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    @abstractmethod
    def split(self, document_id: str, text: str, metadata: dict | None = None) -> list[DocumentChunk]:
        ...
