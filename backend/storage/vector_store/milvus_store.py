import json

from pymilvus import Collection, connections, utility

from backend.core.config import settings
from backend.domain.document import DocumentChunk
from backend.domain.retrieval import RetrievedChunk
from backend.storage.vector_store.base import VectorStore
from backend.storage.vector_store.schema import (
    DEFAULT_COLLECTION_NAME,
    collection_schema,
    index_params,
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
        if not utility.has_collection(self.collection_name):
            collection = Collection(
                name=self.collection_name,
                schema=collection_schema,
                using="default",
            )
            collection.create_index(field_name="embedding", index_params=index_params)
        else:
            collection = Collection(self.collection_name)
            collection.load()
        self._collection = collection

    async def insert_chunks(self, chunks: list[DocumentChunk]) -> int:
        if not self._collection:
            await self.connect()
        if not chunks:
            return 0
        entities = [
            [c.id for c in chunks],
            [c.document_id for c in chunks],
            [c.content for c in chunks],
            [c.chunk_index for c in chunks],
            [json.dumps(c.metadata, ensure_ascii=False) for c in chunks],
            [c.embedding for c in chunks if c.embedding],
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

        chunks: list[RetrievedChunk] = []
        for hit in results[0]:
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
