from backend.core.config import settings
from backend.core.generator.context_builder import ContextBuilder
from backend.core.generator.llm_client import LLMClient
from backend.core.generator.prompt_manager import PromptManager
from backend.core.memory.cache import LRUCache
from backend.core.memory.conversation_memory import ConversationMemory
from backend.core.query.query_processor import QueryProcessor
from backend.core.query.query_router import QueryRouter
from backend.core.retriever.re_ranker import ReRanker
from backend.core.retriever.vector_retriever import VectorRetriever
from backend.domain.conversation import Citation, Conversation, Message
from backend.domain.enums import MessageRole
from backend.domain.retrieval import Query, RetrievalResult


class RAGEngine:
    def __init__(
        self,
        retriever: VectorRetriever,
        llm_client: LLMClient,
        query_processor: QueryProcessor | None = None,
        query_router: QueryRouter | None = None,
        context_builder: ContextBuilder | None = None,
        prompt_manager: PromptManager | None = None,
        re_ranker: ReRanker | None = None,
        memory: ConversationMemory | None = None,
        cache: LRUCache | None = None,
    ):
        self.retriever = retriever
        self.llm_client = llm_client
        self.query_processor = query_processor or QueryProcessor()
        self.query_router = query_router or QueryRouter()
        self.context_builder = context_builder or ContextBuilder()
        self.prompt_manager = prompt_manager or PromptManager()
        self.re_ranker = re_ranker or ReRanker()
        self.memory = memory or ConversationMemory()
        self.cache = cache or LRUCache()

    async def answer(
        self,
        query_text: str,
        conversation: Conversation | None = None,
        top_k: int | None = None,
        stream: bool = False,
    ) -> tuple[str, RetrievalResult | None, list[Citation]]:
        query = Query(text=query_text, top_k=top_k or settings.retrieval.top_k)

        route = await self.query_router.route(query)
        if route == "direct":
            messages = self.prompt_manager.build_direct_prompt(query_text)
            answer = await self.llm_client.generate(messages, stream=stream)
            return answer, None, []

        query = await self.query_processor.rewrite(query)

        cached = self.cache.get(query.text, query.top_k)
        if cached:
            result = cached
        else:
            result = await self.retriever.retrieve(query)
            result = await self.re_ranker.rerank(query, result)
            self.cache.set(query.text, query.top_k, result)

        contexts = self.context_builder.build(result)
        messages = self.prompt_manager.build_rag_prompt(query_text, contexts)
        answer = await self.llm_client.generate(messages, stream=stream)

        citations = [Citation.from_retrieved_chunk(c) for c in result.chunks[:5]]

        return answer, result, citations
