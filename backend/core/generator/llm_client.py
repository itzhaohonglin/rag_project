from openai import AsyncOpenAI

from backend.core.config import settings


class LLMClient:
    def __init__(self):
        self.provider = settings.llm.provider
        if self.provider == "openai":
            kwargs = {"api_key": settings.llm.openai.api_key or None}
            if settings.llm.openai.base_url:
                kwargs["base_url"] = settings.llm.openai.base_url
            self.client = AsyncOpenAI(**kwargs)
            self.model = settings.llm.openai.model
            self.temperature = settings.llm.openai.temperature
            self.max_tokens = settings.llm.openai.max_tokens
        else:
            self.client = AsyncOpenAI(
                api_key="not-needed",
                base_url=settings.llm.local.base_url,
            )
            self.model = settings.llm.local.model
            self.temperature = settings.llm.local.temperature
            self.max_tokens = settings.llm.local.max_tokens

    async def generate(self, messages: list[dict], stream: bool = False) -> str:
        if stream:
            return await self._stream(messages)
        resp = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return resp.choices[0].message.content or ""

    async def _stream(self, messages: list[dict]) -> str:
        stream = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            stream=True,
        )
        result = []
        async for chunk in stream:
            if chunk.choices[0].delta.content:
                result.append(chunk.choices[0].delta.content)
        return "".join(result)
