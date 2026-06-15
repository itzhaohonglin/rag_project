import time

from backend.core.config import settings
from backend.core.retriever.base import Retriever
from backend.domain.retrieval import Query, RetrievalResult
from backend.ingestion.embedding.base import EmbeddingProvider
from backend.ingestion.embedding.bm25_sparse import Bm25SparseEmbedding
from backend.storage.vector_store.base import VectorStore


class HybridRetriever(Retriever):
    """混合检索：稠密 + 稀疏（BM25）加权融合。"""

    def __init__(
        self,
        vector_store: VectorStore,
        embedding_provider: EmbeddingProvider,
        sparse_embedding_provider: Bm25SparseEmbedding,
    ):
        self.vector_store = vector_store
        self.embedding_provider = embedding_provider
        self.sparse_embedding_provider = sparse_embedding_provider

    async def retrieve(self, query: Query) -> RetrievalResult:
        start = time.perf_counter()

        # 稠密查询向量
        query_emb = await self.embedding_provider.embed_query(query.text)
        # 稀疏查询向量
        query_sparse = self.sparse_embedding_provider.compute_sparse(query.text, update_stats=False)

        dense_weight = settings.retrieval.hybrid_dense_weight
        sparse_weight = settings.retrieval.hybrid_sparse_weight

        chunks = await self.vector_store.hybrid_search(
            dense_embedding=query_emb,
            sparse_embedding=query_sparse,
            top_k=query.top_k or settings.retrieval.top_k,
            dense_weight=dense_weight,
            sparse_weight=sparse_weight,
            score_threshold=query.score_threshold or settings.retrieval.score_threshold,
            filters=query.filters,
        )

        elapsed = (time.perf_counter() - start) * 1000
        return RetrievalResult(query=query, chunks=chunks, total_time_ms=elapsed)
