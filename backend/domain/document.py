from datetime import datetime, timezone
from uuid import uuid4

from backend.domain.enums import DocumentStatus, DocumentType


"""
领域模型 — 文档与块。

本文件定义 RAG 管线下游的核心数据载体，纯 Python 数据类，无 ORM 依赖。

─────────────────────────────────────────────────
Document（文档）
─────────────────────────────────────────────────
一个 Document 代表一份被摄入的原始文件（pdf / txt / md / 代码等），
是文档处理流的起点和结果实体。

字段（按构造参数顺序）：
  filename      原始文件名（含扩展名），用于类型推断和展示
  source        来源标识（上传路径 / URL / 原始内容等），
                物料可追溯的唯一凭据
  document_type 枚举 DocumentType，决定 Splitter 策略
  document_id   UUID hex，不传自动生成
  status        枚举 DocumentStatus，标记处理进度
  metadata      自由附加键值对（如作者、来源 URL、标签）
  created_at    创建时间（UTC），不传自动 now
  updated_at    更新时间（UTC），不传自动 now

关键方法：
  to_dict()     → dict  序列化（枚举转 .value、datetime 转 isoformat）
  from_dict()   → Document  反序列化工厂

─────────────────────────────────────────────────
DocumentChunk（文档块）
─────────────────────────────────────────────────
一个 Document 被 Splitter 切分后产生的片段，是向量化和检索的最小单元。
每个 Chunk 携带所属 document_id，通过 chunk_index 保持顺序。

字段：
  document_id   所属文档 ID
  content       片段文本内容
  chunk_index   片段在文档内的序号（从 0 开始）
  chunk_id      UUID hex，不传自动生成
  metadata      自由附加（如所在章节标题、页码）
  embedding     向量数组（检索时填充，存储时可选）
"""


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
        sparse_embedding: dict[int, float] | None = None,
    ):
        self.id = chunk_id or uuid4().hex
        self.document_id = document_id
        self.content = content
        self.chunk_index = chunk_index
        self.metadata = metadata or {}
        self.embedding = embedding
        self.sparse_embedding = sparse_embedding

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "document_id": self.document_id,
            "content": self.content,
            "chunk_index": self.chunk_index,
            "metadata": self.metadata,
        }
