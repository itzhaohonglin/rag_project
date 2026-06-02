from dataclasses import dataclass, field

from backend.domain.enums import RetrievalMode


@dataclass
class Query:
    text: str
    rewritten_text: str | None = None
    mode: RetrievalMode = RetrievalMode.HYBRID
    top_k: int = 10
    score_threshold: float = 0.0
    filters: dict | None = None
    conversation_id: str | None = None


@dataclass
class RetrievedChunk:
    chunk_id: str
    document_id: str
    content: str
    score: float
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "content": self.content,
            "score": self.score,
            "metadata": self.metadata,
            "source": self.metadata.get("source", ""),
        }


@dataclass
class RetrievalResult:
    query: Query
    chunks: list[RetrievedChunk]
    total_time_ms: float = 0.0
    total_chunks: int = 0

    def __post_init__(self):
        self.total_chunks = len(self.chunks)

    def to_dict(self) -> dict:
        return {
            "query": self.query.text,
            "chunks": [c.to_dict() for c in self.chunks],
            "total_chunks": self.total_chunks,
            "total_time_ms": self.total_time_ms,
        }
