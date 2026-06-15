from backend.core.config import settings
from backend.core.crag.evaluator import RelevanceEvaluator
from backend.core.crag.state import CRAGState
from backend.core.generator.context_builder import ContextBuilder
from backend.core.generator.llm_client import LLMClient
from backend.core.generator.prompt_manager import PromptManager
from backend.core.retriever.hybrid_retriever import HybridRetriever
from backend.domain.enums import RetrievalMode
from backend.domain.retrieval import Query


async def retrieve_node(state: CRAGState, retriever: HybridRetriever) -> dict:
    """检索节点 — 调用现有检索器。"""
    query = Query(
        text=state["query_text"],
        top_k=state.get("top_k", settings.retrieval.top_k),
        mode=RetrievalMode(settings.retrieval.retrieval_mode),
    )
    result = await retriever.retrieve(query)
    return {"retrieved_chunks": result.chunks}


async def evaluate_node(state: CRAGState, evaluator: RelevanceEvaluator) -> dict:
    """评估节点 — LLM 判断每个 chunk 相关性。"""
    chunks = state.get("retrieved_chunks", [])
    if not chunks:
        return {"relevance_scores": [], "kept_chunk_indices": []}

    scores = await evaluator.evaluate(state["query_text"], chunks)
    # 打分失败时保底
    if not scores:
        scores = [1.0] * len(chunks)

    # 保留 分数 >= 0.5 的（相关 + 部分相关）
    kept = [i for i, s in enumerate(scores) if s >= 0.5]
    return {"relevance_scores": scores, "kept_chunk_indices": kept}


async def generate_node(state: CRAGState, llm: LLMClient,
                        context_builder: ContextBuilder,
                        prompt_manager: PromptManager) -> dict:
    """生成节点 — 用保留的 chunk 做 RAG 生成。"""
    chunks = state.get("retrieved_chunks", [])
    kept = state.get("kept_chunk_indices", [])
    query_text = state["query_text"]

    kept_chunks = [chunks[i] for i in kept] if kept else []

    if kept_chunks:
        # 构建伪 RetrievalResult 来复用 ContextBuilder
        from backend.domain.retrieval import RetrievalResult
        fake_query = Query(text=query_text, mode=RetrievalMode.HYBRID)
        result = RetrievalResult(query=fake_query, chunks=kept_chunks)
        contexts = context_builder.build(result)
        messages = prompt_manager.build_rag_prompt(query_text, contexts)
    else:
        # 没有相关文档 → 直答（LLM 会说自己不知道）
        messages = prompt_manager.build_direct_prompt(
            f"{query_text}\n\n（注意：没有找到相关资料，如果你不知道答案请直接说明）"
        )

    answer = await llm.generate(messages, stream=state.get("stream", False))

    citations = []
    for c in kept_chunks[:5]:
        from backend.domain.conversation import Citation
        citations.append(Citation.from_retrieved_chunk(c))

    return {"final_answer": answer, "citations": citations}
