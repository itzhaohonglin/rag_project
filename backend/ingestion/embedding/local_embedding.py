import httpx

from backend.domain.embedding import EmbeddingConfig
from backend.ingestion.embedding.base import EmbeddingProvider


class LocalEmbeddingProvider(EmbeddingProvider):
    def __init__(self, config: EmbeddingConfig | None = None):
        super().__init__(config or EmbeddingConfig(model_name="BAAI/bge-large-zh-v1.5", dimension=1024))
        self.base_url = config.local_model_path or "http://localhost:9997/v1"

    async def embed(self, texts: list[str]) -> list[list[float]]:
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{self.base_url}/embeddings",
                json={"model": self.config.model_name, "input": texts},
            )
            resp.raise_for_status()
            data = resp.json()
            return [d["embedding"] for d in data["data"]]

    async def embed_query(self, text: str) -> list[float]:
        result = await self.embed([text])
        return result[0]

    @property
    def dimension(self) -> int:
        return self.config.dimension
