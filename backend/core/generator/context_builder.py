from backend.domain.retrieval import RetrievalResult


class ContextBuilder:
    MAX_TOKENS = 3000
    AVG_CHARS_PER_TOKEN = 4  # rough estimate for Chinese/English mix

    def build(self, result: RetrievalResult, max_tokens: int | None = None) -> list[str]:
        limit = max_tokens or self.MAX_TOKENS
        max_chars = limit * self.AVG_CHARS_PER_TOKEN

        contexts: list[str] = []
        total = 0
        for chunk in result.chunks:
            if total + len(chunk.content) > max_chars:
                break
            contexts.append(chunk.content)
            total += len(chunk.content)

        return contexts
