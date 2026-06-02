from backend.domain.retrieval import Query


class QueryProcessor:
    """Query rewriting and expansion."""

    MAX_HISTORY_TURNS = 6

    async def rewrite(
        self,
        query: Query,
        conversation_history: list[dict] | None = None,
    ) -> Query:
        """Rewrite query with conversation context."""
        query.rewritten_text = query.text
        return query

    async def expand(self, query: Query, terms: list[str] | None = None) -> Query:
        """Expand query with synonyms."""
        if terms:
            query.rewritten_text = f"{query.text} {' '.join(terms)}"
            query.rewritten_text = query.rewritten_text.strip()
        return query
