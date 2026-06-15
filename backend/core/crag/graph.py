from typing import Literal

from langgraph.graph import END, StateGraph

from backend.core.crag.evaluator import RelevanceEvaluator
from backend.core.crag.nodes import evaluate_node, generate_node, retrieve_node
from backend.core.crag.state import CRAGState
from backend.core.generator.context_builder import ContextBuilder
from backend.core.generator.llm_client import LLMClient
from backend.core.generator.prompt_manager import PromptManager
from backend.core.retriever.hybrid_retriever import HybridRetriever


def _decide_generation(state: CRAGState) -> Literal["generate", "generate"]:
    """条件边决策：不管有没有相关文档，都走 generate_node。
    generate_node 内部分支：有相关→RAG，无相关→直答（说不知道）。
    """
    return "generate"


def build_crag_graph(
    retriever: HybridRetriever,
    llm_client: LLMClient,
    context_builder: ContextBuilder,
    prompt_manager: PromptManager,
):
    """构建 CRAG LangGraph 状态图。"""

    evaluator = RelevanceEvaluator(llm_client)

    graph = StateGraph(CRAGState)

    # 注册节点 — 用 partial 注入依赖
    import functools

    graph.add_node("retrieve", functools.partial(retrieve_node, retriever=retriever))
    graph.add_node("evaluate", functools.partial(evaluate_node, evaluator=evaluator))
    graph.add_node(
        "generate",
        functools.partial(
            generate_node,
            llm=llm_client,
            context_builder=context_builder,
            prompt_manager=prompt_manager,
        ),
    )

    # 连线
    graph.add_edge("retrieve", "evaluate")
    graph.add_conditional_edges("evaluate", _decide_generation)
    graph.add_edge("generate", END)

    graph.set_entry_point("retrieve")

    return graph.compile()
