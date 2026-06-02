from datetime import datetime, timezone
from uuid import uuid4

from backend.domain.enums import DocumentStatus, DocumentType


class Document:
    def __init__(
        self,
        filename: str,
        source: str,
        document_type: DocumentType = DocumentType.UNKNOWN,
        document_id: str | None = None,
        status: DocumentStatus = DocumentStatus.PENDING,
        metadata: dict | None = None,
        created_at: datetime | None = None,
        updated_at: datetime | None = None,
    ):
        self.id = document_id or uuid4().hex
        self.filename = filename
        self.source = source
        self.document_type = document_type
        self.status = status
        self.metadata = metadata or {}
        self.created_at = created_at or datetime.now(timezone.utc)
        self.updated_at = updated_at or datetime.now(timezone.utc)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "filename": self.filename,
            "source": self.source,
            "document_type": self.document_type.value,
            "status": self.status.value,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Document":
        return cls(
            document_id=data["id"],
            filename=data["filename"],
            source=data["source"],
            document_type=DocumentType(data["document_type"]),
            status=DocumentStatus(data["status"]),
            metadata=data.get("metadata", {}),
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
        )


class DocumentChunk:
    def __init__(
        self,
        document_id: str,
        content: str,
        chunk_index: int,
        chunk_id: str | None = None,
        metadata: dict | None = None,
        embedding: list[float] | None = None,
    ):
        self.id = chunk_id or uuid4().hex
        self.document_id = document_id
        self.content = content
        self.chunk_index = chunk_index
        self.metadata = metadata or {}
        self.embedding = embedding

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "document_id": self.document_id,
            "content": self.content,
            "chunk_index": self.chunk_index,
            "metadata": self.metadata,
        }
