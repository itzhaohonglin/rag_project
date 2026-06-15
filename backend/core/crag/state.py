from typing import TypedDict

from backend.domain.conversation import Citation
from backend.domain.retrieval import RetrievedChunk


class CRAGState(TypedDict):
    """CRAG 图状态，贯穿整个流程。"""

    # 输入
    query_text: str
    top_k: int
    stream: bool

    # 处理中
    retrieved_chunks: list[RetrievedChunk]
    relevance_scores: list[float]  # 每个 chunk 的分数 [0,1]
    kept_chunk_indices: list[int]  # 保留的 chunk 索引

    # 输出
    final_answer: str
    citations: list[Citation]
