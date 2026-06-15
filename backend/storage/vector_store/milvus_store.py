import json

from pymilvus import Collection, connections, utility

from backend.core.config import settings
from backend.domain.document import DocumentChunk
from backend.domain.retrieval import RetrievedChunk
from backend.storage.vector_store.base import VectorStore
from backend.storage.vector_store.schema import (
    DEFAULT_COLLECTION_NAME,
    collection_schema,
    dense_index_params,
    sparse_index_params,
)


class MilvusStore(VectorStore):
    def __init__(
        self,
        host: str | None = None,
        port: int | None = None,
        collection_name: str | None = None,
    ):
        self.host = host or settings.milvus.host
        self.port = port or settings.milvus.port
        self.collection_name = collection_name or settings.milvus.collection
        self._collection: Collection | None = None

    async def connect(self):
        connections.connect(host=self.host, port=self.port)
        if utility.has_collection(self.collection_name):
            collection = Collection(self.collection_name)
            # 检查是否有 sparse_embedding 字段，没有则重建
            schema = collection.schema
            field_names = [f.name for f in schema.fields]
            if "sparse_embedding" not in field_names:
                utility.drop_collection(self.collection_name)
                collection = Collection(
                    name=self.collection_name,
                    schema=collection_schema,
                    using="default",
                )
                collection.create_index(field_name="embedding", index_params=dense_index_params)
                collection.create_index(field_name="sparse_embedding", index_params=sparse_index_params)
        else:
            collection = Collection(
                name=self.collection_name,
                schema=collection_schema,
                using="default",
            )
            collection.create_index(field_name="embedding", index_params=dense_index_params)
            collection.create_index(field_name="sparse_embedding", index_params=sparse_index_params)
        collection.load()
        self._collection = collection

    async def insert_chunks(self, chunks: list[DocumentChunk]) -> int:
        if not self._collection:
            await self.connect()
        if not chunks:
            return 0

        dense_embeddings = [c.embedding for c in chunks if c.embedding]
        sparse_embeddings = [c.sparse_embedding or {} for c in chunks]

        entities = [
            [c.id for c in chunks],
            [c.document_id for c in chunks],
            [c.content for c in chunks],
            [c.chunk_index for c in chunks],
            [json.dumps(c.metadata, ensure_ascii=False) for c in chunks],
            dense_embeddings,
            sparse_embeddings,
        ]
        self._collection.insert(entities)
        self._collection.flush()
        return len(chunks)

    async def search(
        self,
        embedding: list[float],
        top_k: int = 10,
        score_threshold: float = 0.0,
        filters: dict | None = None,
    ) -> list[RetrievedChunk]:
        if not self._collection:
            await self.connect()
        self._collection.load()

        search_params = {"metric_type": "COSINE", "params": {"nprobe": 10}}
        expr = None
        if filters and "document_id" in filters:
            expr = f'document_id == "{filters["document_id"]}"'

        results = self._collection.search(
            data=[embedding],
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            expr=expr,
            output_fields=["chunk_id", "document_id", "content", "metadata_json"],
        )

        return self._parse_search_results(results[0], score_threshold)

    async def sparse_search(
        self,
        sparse_embedding: dict[int, float],
        top_k: int = 10,
        score_threshold: float = 0.0,
        filters: dict | None = None,
    ) -> list[RetrievedChunk]:
        if not self._collection:
            await self.connect()
        self._collection.load()

        search_params = {"metric_type": "IP"}
        expr = None
        if filters and "document_id" in filters:
            expr = f'document_id == "{filters["document_id"]}"'

        results = self._collection.search(
            data=[sparse_embedding],
            anns_field="sparse_embedding",
            param=search_params,
            limit=top_k,
            expr=expr,
            output_fields=["chunk_id", "document_id", "content", "metadata_json"],
        )

        return self._parse_search_results(results[0], score_threshold)

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
        """Hybrid search: run dense + sparse separately, then fuse with weighted RRF."""
        dense_chunks = await self.search(
            embedding=dense_embedding,
            top_k=top_k * 2,
            score_threshold=0.0,
            filters=filters,
        )
        sparse_chunks = await self.sparse_search(
            sparse_embedding=sparse_embedding,
            top_k=top_k * 2,
            score_threshold=0.0,
            filters=filters,
        )

        return self._fuse_results(dense_chunks, sparse_chunks, top_k, dense_weight, sparse_weight, score_threshold)

    def _fuse_results(
        self,
        dense: list[RetrievedChunk],
        sparse: list[RetrievedChunk],
        top_k: int,
        dense_weight: float,
        sparse_weight: float,
        score_threshold: float,
    ) -> list[RetrievedChunk]:
        seen: dict[str, list[float]] = {}
        for c in dense:
            seen.setdefault(c.chunk_id, [0.0, 0.0])
            seen[c.chunk_id][0] = c.score
        for c in sparse:
            seen.setdefault(c.chunk_id, [0.0, 0.0])
            seen[c.chunk_id][1] = c.score

        # 构建一个全量映射: chunk_id -> chunk 对象
        chunk_map = {}
        for c in dense + sparse:
            if c.chunk_id not in chunk_map:
                chunk_map[c.chunk_id] = c

        fused: list[RetrievedChunk] = []
        for chunk_id, (d_score, s_score) in seen.items():
            if d_score == 0 and s_score == 0:
                continue
            fused_score = dense_weight * d_score + sparse_weight * s_score
            if fused_score < score_threshold:
                continue
            c = chunk_map[chunk_id]
            c.score = fused_score
            fused.append(c)

        fused.sort(key=lambda x: x.score, reverse=True)
        return fused[:top_k]

    def _parse_search_results(self, hits: list, score_threshold: float) -> list[RetrievedChunk]:
        chunks: list[RetrievedChunk] = []
        for hit in hits:
            if hit.score < score_threshold:
                continue
            metadata = {}
            if "metadata_json" in hit.entity:
                metadata = json.loads(hit.entity.get("metadata_json", "{}"))
            chunks.append(RetrievedChunk(
                chunk_id=hit.entity.get("chunk_id"),
                document_id=hit.entity.get("document_id"),
                content=hit.entity.get("content"),
                score=hit.score,
                metadata=metadata,
            ))
        return chunks

    async def delete_document(self, document_id: str) -> bool:
        if not self._collection:
            await self.connect()
        expr = f'document_id == "{document_id}"'
        self._collection.delete(expr)
        self._collection.flush()
        return True

    async def count(self) -> int:
        if not self._collection:
            await self.connect()
        self._collection.load()
        return self._collection.num_entities
