from pathlib import Path

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
from backend.core.retriever.hybrid_retriever import HybridRetriever
from backend.core.retriever.re_ranker import ReRanker
from backend.core.retriever.sparse_retriever import SparseRetriever
from backend.core.retriever.vector_retriever import VectorRetriever
from backend.ingestion.embedding.bm25_sparse import Bm25SparseEmbedding
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
_sparse_embedding_provider: Bm25SparseEmbedding | None = None


async def get_vector_store() -> MilvusStore:
    global _vector_store
    if _vector_store is None:
        _vector_store = MilvusStore()
        await _vector_store.connect()
    return _vector_store


def get_embedding_provider() -> OpenAIEmbeddingProvider:
    global _embedding_provider
    if _embedding_provider is None:
        from backend.domain.embedding import EmbeddingConfig as DomainEmbeddingConfig

        cfg = settings.embedding
        domain_cfg = DomainEmbeddingConfig(
            model_name=cfg.openai.model if cfg.provider == "openai" else "",
            dimension=cfg.dimensions,
            provider=cfg.provider,
        )
        _embedding_provider = OpenAIEmbeddingProvider(
            config=domain_cfg,
            base_url=settings.llm.openai.base_url or None,
            api_key=settings.llm.openai.api_key or None,
        )
    return _embedding_provider


def get_sparse_embedding_provider() -> Bm25SparseEmbedding:
    global _sparse_embedding_provider
    if _sparse_embedding_provider is None:
        state_path = Path(settings.storage.upload_dir).parent / "bm25_state.json"
        _sparse_embedding_provider = Bm25SparseEmbedding(state_path=state_path)
    return _sparse_embedding_provider


async def get_rag_engine(
    vector_store: MilvusStore = Depends(get_vector_store),
    embedding_provider: OpenAIEmbeddingProvider = Depends(get_embedding_provider),
    sparse_provider: Bm25SparseEmbedding = Depends(get_sparse_embedding_provider),
) -> RAGEngine:
    global _engine
    if _engine is None:
        vector_retriever = VectorRetriever(vector_store, embedding_provider)
        sparse_retriever = SparseRetriever(vector_store, sparse_provider)
        hybrid_retriever = HybridRetriever(vector_store, embedding_provider, sparse_provider)
        llm = LLMClient()
        _engine = RAGEngine(
            vector_retriever=vector_retriever,
            sparse_retriever=sparse_retriever,
            hybrid_retriever=hybrid_retriever,
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
    sparse_provider: Bm25SparseEmbedding = Depends(get_sparse_embedding_provider),
) -> IngestionPipeline:
    return IngestionPipeline(
        embedding_provider=embedding_provider,
        sparse_embedding_provider=sparse_provider,
    )
