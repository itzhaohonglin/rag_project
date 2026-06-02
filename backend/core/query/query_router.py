from backend.domain.retrieval import Query


class QueryRouter:
    """Route queries to direct answer vs retrieval-augmented generation."""

    GREETING_PATTERNS = ["hi", "hello", "hey", "你好", "您好"]

    async def route(self, query: Query) -> str:
        """Return 'direct' or 'rag'."""
        text = query.text.strip().lower()
        if text in self.GREETING_PATTERNS or len(text) < 3:
            return "direct"
        return "rag"
