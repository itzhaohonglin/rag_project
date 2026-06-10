from openai import AsyncOpenAI

from backend.domain.embedding import EmbeddingConfig
from backend.ingestion.embedding.base import EmbeddingProvider


class OpenAIEmbeddingProvider(EmbeddingProvider):
    def __init__(
        self,
        config: EmbeddingConfig | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
    ):
        super().__init__(config or EmbeddingConfig())
        kwargs = {}
        if base_url:
            kwargs["base_url"] = base_url
        if api_key:
            kwargs["api_key"] = api_key
        self.client = AsyncOpenAI(**kwargs)
        self._dimension = self.config.dimension

    async def embed(self, texts: list[str]) -> list[list[float]]:
        resp = await self.client.embeddings.create(
            model=self.config.model_name,
            input=texts,
        )
        return [d.embedding for d in resp.data]

    async def embed_query(self, text: str) -> list[float]:
        result = await self.embed([text])
        return result[0]

    @property
    def dimension(self) -> int:
        return self._dimension
