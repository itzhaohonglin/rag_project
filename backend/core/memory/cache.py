import time
from collections import OrderedDict

from backend.domain.retrieval import RetrievalResult


class LRUCache:
    def __init__(self, capacity: int = 128, ttl_seconds: int = 300):
        self.capacity = capacity
        self.ttl = ttl_seconds
        self._cache: OrderedDict[str, tuple[float, RetrievalResult]] = OrderedDict()

    def _key(self, query_text: str, top_k: int) -> str:
        return f"{query_text}:{top_k}"

    def get(self, query_text: str, top_k: int) -> RetrievalResult | None:
        key = self._key(query_text, top_k)
        if key not in self._cache:
            return None
        ts, result = self._cache[key]
        if time.time() - ts > self.ttl:
            del self._cache[key]
            return None
        self._cache.move_to_end(key)
        return result

    def set(self, query_text: str, top_k: int, result: RetrievalResult):
        key = self._key(query_text, top_k)
        self._cache[key] = (time.time(), result)
        if len(self._cache) > self.capacity:
            self._cache.popitem(last=False)

    def clear(self):
        self._cache.clear()
