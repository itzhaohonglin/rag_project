import time

from backend.core.config import settings
from backend.core.retriever.base import Retriever
from backend.domain.retrieval import Query, RetrievalResult, RetrievedChunk
from backend.ingestion.embedding.base import EmbeddingProvider
from backend.storage.vector_store.base import VectorStore


class HybridRetriever(Retriever):
    """Vector search + keyword scoring fusion."""

    def __init__(self, vector_store: VectorStore, embedding_provider: EmbeddingProvider):
        self.vector_store = vector_store
        self.embedding_provider = embedding_provider

    def _keyword_score(self, text: str, query_terms: set[str]) -> float:
        text_lower = text.lower()
        hits = sum(1 for t in query_terms if t.lower() in text_lower)
        return hits / max(len(query_terms), 1)

    async def retrieve(self, query: Query) -> RetrievalResult:
        start = time.perf_counter()
        query_emb = await self.embedding_provider.embed_query(query.text)
        vector_chunks = await self.vector_store.search(
            embedding=query_emb,
            top_k=query.top_k or settings.retrieval.top_k,
            score_threshold=query.score_threshold or settings.retrieval.score_threshold,
            filters=query.filters,
        )

        query_terms = set(query.text.lower().split())
        seen = set()
        fused: list[RetrievedChunk] = []
        for c in vector_chunks:
            if c.chunk_id in seen:
                continue
            seen.add(c.chunk_id)
            kw_score = self._keyword_score(c.content, query_terms)
            c.score = 0.7 * c.score + 0.3 * kw_score
            fused.append(c)

        fused.sort(key=lambda x: x.score, reverse=True)
        elapsed = (time.perf_counter() - start) * 1000
        return RetrievalResult(query=query, chunks=fused[: query.top_k], total_time_ms=elapsed)
