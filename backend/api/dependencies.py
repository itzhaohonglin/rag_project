from fastapi import Depends
from sqlalchemy.orm import Session

from backend.core.config import settings
from backend.core.generator.context_builder import ContextBuilder
from backend.core.generator.llm_client import LLMClient
from backend.core.generator.prompt_manager import PromptManager
from backend.core.memory.cache import LRUCache
from backend.core.memory.conversation_memory import ConversationMemory
from backend.core.query.query_processor import QueryProcessor
from backend.core.query.query_router import QueryRouter
from backend.core.rag_engine import RAGEngine
from backend.core.retriever.re_ranker import ReRanker
from backend.core.retriever.vector_retriever import VectorRetriever
from backend.ingestion.embedding.openai_embedding import OpenAIEmbeddingProvider
from backend.ingestion.processor.pipeline import IngestionPipeline
from backend.storage.file_store.local_fs import LocalFileStore
from backend.storage.relational_db.base import get_session
from backend.storage.relational_db.conversation_repo import ConversationRepository
from backend.storage.relational_db.document_repo import DocumentRepository
from backend.storage.vector_store.milvus_store import MilvusStore

_engine: RAGEngine | None = None
_vector_store: MilvusStore | None = None
_embedding_provider: OpenAIEmbeddingProvider | None = None


async def get_vector_store() -> MilvusStore:
    global _vector_store
    if _vector_store is None:
        _vector_store = MilvusStore()
        await _vector_store.connect()
    return _vector_store


def get_embedding_provider() -> OpenAIEmbeddingProvider:
    global _embedding_provider
    if _embedding_provider is None:
        _embedding_provider = OpenAIEmbeddingProvider()
    return _embedding_provider


async def get_rag_engine(
    vector_store: MilvusStore = Depends(get_vector_store),
    embedding_provider: OpenAIEmbeddingProvider = Depends(get_embedding_provider),
) -> RAGEngine:
    global _engine
    if _engine is None:
        retriever = VectorRetriever(vector_store, embedding_provider)
        llm = LLMClient()
        _engine = RAGEngine(
            retriever=retriever,
            llm_client=llm,
            query_processor=QueryProcessor(),
            query_router=QueryRouter(),
            context_builder=ContextBuilder(),
            prompt_manager=PromptManager(),
            re_ranker=ReRanker(),
            memory=ConversationMemory(),
            cache=LRUCache(),
        )
    return _engine


def get_document_repo(session: Session = Depends(get_session)) -> DocumentRepository:
    return DocumentRepository(session)


def get_conversation_repo(session: Session = Depends(get_session)) -> ConversationRepository:
    return ConversationRepository(session)


def get_file_store() -> LocalFileStore:
    return LocalFileStore(settings.storage.upload_dir)


def get_ingestion_pipeline(
    embedding_provider: OpenAIEmbeddingProvider = Depends(get_embedding_provider),
) -> IngestionPipeline:
    return IngestionPipeline(embedding_provider=embedding_provider)
