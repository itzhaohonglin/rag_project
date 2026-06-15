import time

from backend.core.config import settings
from backend.core.retriever.base import Retriever
from backend.domain.retrieval import Query, RetrievalResult
from backend.ingestion.embedding.bm25_sparse import Bm25SparseEmbedding
from backend.storage.vector_store.base import VectorStore


class SparseRetriever(Retriever):
    """纯稀疏检索（BM25）。"""

    def __init__(self, vector_store: VectorStore, sparse_embedding_provider: Bm25SparseEmbedding):
        self.vector_store = vector_store
        self.sparse_embedding_provider = sparse_embedding_provider

    async def retrieve(self, query: Query) -> RetrievalResult:
        start = time.perf_counter()
        query_sparse = self.sparse_embedding_provider.compute_sparse(query.text, update_stats=False)
        chunks = await self.vector_store.sparse_search(
            sparse_embedding=query_sparse,
            top_k=query.top_k or settings.retrieval.top_k,
            score_threshold=query.score_threshold or settings.retrieval.score_threshold,
            filters=query.filters,
        )
        elapsed = (time.perf_counter() - start) * 1000
        return RetrievalResult(query=query, chunks=chunks, total_time_ms=elapsed)
