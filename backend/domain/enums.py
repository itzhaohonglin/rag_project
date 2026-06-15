from enum import Enum, auto


class DocumentStatus(Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    READY = "ready"
    FAILED = "failed"
    DELETED = "deleted"


class DocumentType(Enum):
    TEXT = "text"
    PDF = "pdf"
    MARKDOWN = "markdown"
    WORD = "word"
    EXCEL = "excel"
    PPT = "ppt"
    CODE = "code"
    UNKNOWN = "unknown"


class ChunkStrategy(Enum):
    RECURSIVE = "recursive"
    SEMANTIC = "semantic"
    CODE = "code"


class MessageRole(Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class RetrievalMode(Enum):
    DENSE = "dense"
    SPARSE = "sparse"
    HYBRID = "hybrid"
