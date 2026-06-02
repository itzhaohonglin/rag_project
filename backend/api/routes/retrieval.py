import time

from fastapi import APIRouter, Depends

from backend.api.errors import success_response
from backend.api.schemas.retrieval import QueryRequest, QueryResponse, RetrievedChunkSchema
from backend.core.rag_engine import RAGEngine

router = APIRouter()


@router.post("/retrieval/query", response_model=dict)
async def query_retrieval(
    req: QueryRequest,
    engine: RAGEngine = Depends(),
):
    start = time.perf_counter()
    answer, result, citations = await engine.answer(req.query)
    elapsed = (time.perf_counter() - start) * 1000

    chunks = []
    if result:
        chunks = [RetrievedChunkSchema(
            chunk_id=c.chunk_id, document_id=c.document_id,
            content=c.content, score=c.score, metadata=c.metadata,
        ).model_dump() for c in result.chunks]

    return success_response(QueryResponse(
        answer=answer,
        chunks=chunks,
        total_time_ms=elapsed,
        citations=[{"chunk_id": c.chunk_id, "document_id": c.document_id,
                     "content": c.content[:200], "score": c.score} for c in citations],
    ).model_dump())
