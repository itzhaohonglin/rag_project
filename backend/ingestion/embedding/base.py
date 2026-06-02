from abc import ABC, abstractmethod

from backend.domain.embedding import EmbeddingConfig


class EmbeddingProvider(ABC):
    def __init__(self, config: EmbeddingConfig | None = None):
        self.config = config or EmbeddingConfig()

    @abstractmethod
    async def embed(self, texts: list[str]) -> list[list[float]]:
        ...

    @abstractmethod
    async def embed_query(self, text: str) -> list[float]:
        ...

    @property
    @abstractmethod
    def dimension(self) -> int:
        ...
