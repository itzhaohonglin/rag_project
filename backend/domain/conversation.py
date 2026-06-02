from dataclasses import dataclass, field
from datetime import datetime, timezone
from uuid import uuid4

from backend.domain.enums import MessageRole
from backend.domain.retrieval import RetrievedChunk


@dataclass
class Citation:
    chunk_id: str
    document_id: str
    content: str
    score: float

    @classmethod
    def from_retrieved_chunk(cls, chunk: RetrievedChunk) -> "Citation":
        return cls(
            chunk_id=chunk.chunk_id,
            document_id=chunk.document_id,
            content=chunk.content[:200],
            score=chunk.score,
        )


@dataclass
class Message:
    role: MessageRole
    content: str
    citations: list[Citation] = field(default_factory=list)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict:
        return {
            "role": self.role.value,
            "content": self.content,
            "citations": [
                {"chunk_id": c.chunk_id, "document_id": c.document_id, "content": c.content, "score": c.score}
                for c in self.citations
            ],
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class Conversation:
    id: str = field(default_factory=lambda: uuid4().hex)
    messages: list[Message] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict = field(default_factory=dict)

    def add_message(self, message: Message):
        self.messages.append(message)
        self.updated_at = datetime.now(timezone.utc)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "messages": [m.to_dict() for m in self.messages],
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "metadata": self.metadata,
        }
