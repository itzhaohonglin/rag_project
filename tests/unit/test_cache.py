from backend.core.memory.cache import LRUCache
from backend.domain.retrieval import Query, RetrievalResult, RetrievedChunk


class TestLRUCache:
    def test_set_and_get(self):
        cache = LRUCache(capacity=10, ttl_seconds=300)
        result = RetrievalResult(
            query=Query(text="test"),
            chunks=[RetrievedChunk(chunk_id="c1", document_id="d1", content="x", score=1.0)],
        )
        cache.set("test", 10, result)
        cached = cache.get("test", 10)
        assert cached is not None
        assert cached.total_chunks == 1

    def test_cache_miss(self):
        cache = LRUCache(capacity=10, ttl_seconds=300)
        assert cache.get("nonexistent", 10) is None

    def test_cache_eviction(self):
        cache = LRUCache(capacity=2, ttl_seconds=300)
        for i in range(3):
            result = RetrievalResult(
                query=Query(text=str(i)),
                chunks=[RetrievedChunk(chunk_id=f"c{i}", document_id="d1", content="x", score=1.0)],
            )
            cache.set(str(i), 10, result)
        assert cache.get("0", 10) is None
        assert cache.get("2", 10) is not None

    def test_clear(self):
        cache = LRUCache(capacity=10, ttl_seconds=300)
        result = RetrievalResult(query=Query(text="x"), chunks=[])
        cache.set("x", 10, result)
        cache.clear()
        assert cache.get("x", 10) is None
