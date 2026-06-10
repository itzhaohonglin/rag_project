import asyncio

from celery import shared_task

from backend.core.config import settings
from backend.domain.document import Document
from backend.domain.embedding import EmbeddingConfig as DomainEmbeddingConfig
from backend.domain.enums import DocumentStatus, DocumentType
from backend.ingestion.embedding.openai_embedding import OpenAIEmbeddingProvider
from backend.ingestion.processor.pipeline import IngestionPipeline
from backend.storage.file_store.local_fs import LocalFileStore
from backend.storage.relational_db.base import SessionLocal
from backend.storage.relational_db.document_repo import DocumentRepository
from backend.storage.vector_store.milvus_store import MilvusStore


@shared_task(bind=True, max_retries=3)
def process_document_task(self, document_id: str, file_path: str, filename: str):
    """Async document processing task."""
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(_process_document(document_id, file_path))
    except Exception as exc:
        self.retry(exc=exc, countdown=60)
    finally:
        loop.close()


async def _process_document(document_id: str, file_path: str):
    session = SessionLocal()
    try:
        repo = DocumentRepository(session)
        cfg = settings.embedding
        domain_cfg = DomainEmbeddingConfig(
            model_name=cfg.openai.model if cfg.provider == "openai" else "",
            dimension=cfg.dimensions,
            provider=cfg.provider,
        )
        provider = OpenAIEmbeddingProvider(
            config=domain_cfg,
            base_url=settings.llm.openai.base_url or None,
            api_key=settings.llm.openai.api_key or None,
        )
        pipeline = IngestionPipeline(embedding_provider=provider)
        vector_store = MilvusStore()
        await vector_store.connect()

        repo.update_status(document_id, DocumentStatus.PROCESSING)
        doc = repo.get(document_id)
        if not doc:
            return

        chunks = await pipeline.process(file_path, doc)
        await vector_store.insert_chunks(chunks)
        repo.update_status(document_id, DocumentStatus.READY)
    finally:
        session.close()
