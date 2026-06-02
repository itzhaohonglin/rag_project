from pydantic import BaseModel


class RetrievedChunkSchema(BaseModel):
    chunk_id: str
    document_id: str
    content: str
    score: float
    metadata: dict | None = None


class QueryRequest(BaseModel):
    query: str
    top_k: int = 10
    mode: str = "hybrid"
    conversation_id: str | None = None
    stream: bool = False


class QueryResponse(BaseModel):
    answer: str
    chunks: list[RetrievedChunkSchema] = []
    total_time_ms: float = 0.0
    citations: list[dict] = []
