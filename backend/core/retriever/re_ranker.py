import json

import httpx

from backend.core.config import settings
from backend.domain.retrieval import Query, RetrievalResult, RetrievedChunk


class ReRanker:
    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or settings.retrieval.rerank_model

    async def rerank(self, query: Query, result: RetrievalResult) -> RetrievalResult:
        if not result.chunks or not settings.retrieval.rerank_enabled:
            return result

        try:
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.post(
                    "http://localhost:9997/v1/rerank",
                    json={
                        "model": self.model_name,
                        "query": query.text,
                        "documents": [c.content for c in result.chunks],
                    },
                )
                resp.raise_for_status()
                data = resp.json()

            reranked = []
            for item in data.get("results", []):
                idx = item["index"]
                result.chunks[idx].score = item.get("relevance_score", result.chunks[idx].score)
                reranked.append(result.chunks[idx])

            reranked.sort(key=lambda x: x.score, reverse=True)
            result.chunks = reranked
        except Exception:
            pass  # fallback to original order on error

        return result
