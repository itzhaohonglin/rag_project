from abc import ABC, abstractmethod

from backend.domain.retrieval import Query, RetrievalResult


class Retriever(ABC):
    @abstractmethod
    async def retrieve(self, query: Query) -> RetrievalResult:
        ...
