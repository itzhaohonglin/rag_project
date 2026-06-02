import time

from backend.core.config import settings
from backend.core.retriever.base import Retriever
from backend.domain.retrieval import Query, RetrievalResult
from backend.ingestion.embedding.base import EmbeddingProvider
from backend.storage.vector_store.base import VectorStore


class VectorRetriever(Retriever):
    def __init__(self, vector_store: VectorStore, embedding_provider: EmbeddingProvider):
        self.vector_store = vector_store
        self.embedding_provider = embedding_provider

    async def retrieve(self, query: Query) -> RetrievalResult:
        start = time.perf_counter()
        query_emb = await self.embedding_provider.embed_query(query.text)
        chunks = await self.vector_store.search(
            embedding=query_emb,
            top_k=query.top_k or settings.retrieval.top_k,
            score_threshold=query.score_threshold or settings.retrieval.score_threshold,
            filters=query.filters,
        )
        elapsed = (time.perf_counter() - start) * 1000
        return RetrievalResult(query=query, chunks=chunks, total_time_ms=elapsed)
