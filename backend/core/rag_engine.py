from backend.core.config import settings
from backend.core.generator.context_builder import ContextBuilder
from backend.core.generator.llm_client import LLMClient
from backend.core.generator.prompt_manager import PromptManager
from backend.core.memory.cache import LRUCache
from backend.core.memory.conversation_memory import ConversationMemory
from backend.core.query.query_processor import QueryProcessor
from backend.core.query.query_router import QueryRouter
from backend.core.retriever.hybrid_retriever import HybridRetriever
from backend.core.retriever.re_ranker import ReRanker
from backend.core.retriever.sparse_retriever import SparseRetriever
from backend.core.retriever.vector_retriever import VectorRetriever
from backend.domain.conversation import Citation, Conversation, Message
from backend.domain.enums import MessageRole, RetrievalMode
from backend.domain.retrieval import Query, RetrievalResult


class RAGEngine:
    """RAG 核心编排器，统筹一次问答的完整流程。

    流程（answer 方法）：
        ┌─ QueryRouter ─→ "direct" ─→ LLMClient 直答（打招呼/短文本）
        │
        Query ─┤
               └─ QueryRouter ─→ "rag"
                    → QueryProcessor.rewrite()    # 重写查询（结合对话历史）
                    → LRUCache (命中则跳过检索)
                    → [CRAG 模式] CRAG LangGraph：Retrieve → Evaluate → Generate
                    → [普通模式]
                         VectorRetriever / SparseRetriever / HybridRetriever
                       → ReRanker.rerank()        # Cross-Encoder 重排序
                       → ContextBuilder.build()   # 组装上下文（限 max_tokens）
                       → PromptManager.build_rag_prompt() # 拼 system+user 消息
                       → LLMClient.generate()     # 调 LLM 生成答案
                    → 返回 (answer, retrieval_result, citations)
    """

    def __init__(
        self,
        vector_retriever: VectorRetriever,
        sparse_retriever: SparseRetriever,
        hybrid_retriever: HybridRetriever,
        llm_client: LLMClient,
        query_processor: QueryProcessor | None = None,
        query_router: QueryRouter | None = None,
        context_builder: ContextBuilder | None = None,
        prompt_manager: PromptManager | None = None,
        re_ranker: ReRanker | None = None,
        memory: ConversationMemory | None = None,
        cache: LRUCache | None = None,
    ):
        self.vector_retriever = vector_retriever  # 稠密检索（DENSE）
        self.sparse_retriever = sparse_retriever  # BM25 稀疏检索（SPARSE）
        self.hybrid_retriever = hybrid_retriever  # 稠密+稀疏加权融合（HYBRID，默认 7:3）
        self.llm_client = llm_client              # LLM 调用客户端（openai / 本地 vLLM）
        self.query_processor = query_processor or QueryProcessor()
        self.query_router = query_router or QueryRouter()          # 直答/检索路由
        self.context_builder = context_builder or ContextBuilder() # 检索结果→上下文文本
        self.prompt_manager = prompt_manager or PromptManager()    # 拼 system + user 消息
        self.re_ranker = re_ranker or ReRanker()                   # Cross-Encoder 重打分
        self.memory = memory or ConversationMemory()               # 对话滑动窗口裁剪
        self.cache = cache or LRUCache()                           # 检索结果 LRU 缓存
        self._crag_graph = None

    def _get_retriever(self, mode: RetrievalMode):
        """按检索模式选对应的检索器。"""
        if mode == RetrievalMode.DENSE:
            return self.vector_retriever   # 仅稠密向量
        elif mode == RetrievalMode.SPARSE:
            return self.sparse_retriever   # 仅 BM25 稀疏向量
        return self.hybrid_retriever       # 默认：混合检索

    def _ensure_crag_graph(self):
        """懒加载 CRAG LangGraph。"""
        if self._crag_graph is None:
            from backend.core.crag.graph import build_crag_graph

            self._crag_graph = build_crag_graph(
                retriever=self.hybrid_retriever,
                llm_client=self.llm_client,
                context_builder=self.context_builder,
                prompt_manager=self.prompt_manager,
            )
        return self._crag_graph

    async def _answer_with_crag(
        self, query_text: str, top_k: int | None, stream: bool
    ) -> tuple[str, RetrievalResult | None, list[Citation]]:
        """使用 CRAG LangGraph 执行问答。"""
        graph = self._ensure_crag_graph()
        state = await graph.ainvoke({
            "query_text": query_text,
            "top_k": top_k or settings.retrieval.top_k,
            "stream": stream,
            "retrieved_chunks": [],
            "relevance_scores": [],
            "kept_chunk_indices": [],
            "final_answer": "",
            "citations": [],
        })
        return state["final_answer"], None, state.get("citations", [])

    async def answer(
        self,
        query_text: str,
        conversation: Conversation | None = None,
        top_k: int | None = None,
        stream: bool = False,
        mode: RetrievalMode | None = None,
        crag: bool | None = None,
    ) -> tuple[str, RetrievalResult | None, list[Citation]]:
        """执行一次问答，返回 (答案, 检索结果, 引用列表)。

        crag=True 时走 CRAG LangGraph 流程（检索 → 评估 → 过滤 → 生成）。
        crag 默认值来自 settings.crag.enabled（yaml 配置）。
        """
        use_crag = settings.crag.enabled if crag is None else crag
        mode = mode or RetrievalMode(settings.retrieval.retrieval_mode)
        query = Query(text=query_text, top_k=top_k or settings.retrieval.top_k, mode=mode)

        # 第一步：路由判断 → 直答还是检索
        route = await self.query_router.route(query)
        if route == "direct":
            messages = self.prompt_manager.build_direct_prompt(query_text)
            answer = await self.llm_client.generate(messages, stream=stream)
            return answer, None, []

        # 第二步：查询重写（结合对话上下文）
        query = await self.query_processor.rewrite(query)

        # CRAG 模式 → 走 LangGraph
        if use_crag:
            return await self._answer_with_crag(query_text, top_k, stream)

        # 第三步：普通模式 — 检索 + 缓存
        cached = self.cache.get(query.text, query.top_k)
        if cached:
            result = cached
        else:
            retriever = self._get_retriever(mode)
            result = await retriever.retrieve(query)   # Milvus 向量/knn 检索
            result = await self.re_ranker.rerank(query, result)  # Cross-Encoder 重打分
            self.cache.set(query.text, query.top_k, result)

        # 第四步：组装上下文 → 拼 prompt → 调 LLM
        contexts = self.context_builder.build(result)  # 截取 top ctx 到 max_tokens
        messages = self.prompt_manager.build_rag_prompt(query_text, contexts)
        answer = await self.llm_client.generate(messages, stream=stream)

        # 第五步：取前 5 个 chunk 做引用来源
        citations = [Citation.from_retrieved_chunk(c) for c in result.chunks[:5]]

        return answer, result, citations
