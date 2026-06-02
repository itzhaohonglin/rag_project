from pathlib import Path

from backend.core.config import settings
from backend.domain.document import Document, DocumentChunk
from backend.domain.enums import DocumentStatus, DocumentType
from backend.ingestion.embedding.base import EmbeddingProvider
from backend.ingestion.loader.base import Loader
from backend.ingestion.loader.code_loader import CodeLoader, LoaderRegistry
from backend.ingestion.processor.cleaner import TextCleaner
from backend.ingestion.processor.extractor import MetadataExtractor
from backend.ingestion.splitter.base import Splitter
from backend.ingestion.splitter.code_splitter import CodeSplitter
from backend.ingestion.splitter.recursive_splitter import RecursiveSplitter
from backend.ingestion.splitter.semantic_splitter import SemanticSplitter


class IngestionPipeline:
    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        loader_registry: LoaderRegistry | None = None,
        cleaner: TextCleaner | None = None,
        extractor: MetadataExtractor | None = None,
    ):
        self.embedding_provider = embedding_provider
        self.loader_registry = loader_registry or LoaderRegistry()
        self.cleaner = cleaner or TextCleaner()
        self.extractor = extractor or MetadataExtractor()

    def _get_splitter(self, extension: str) -> Splitter:
        chunk_size = settings.storage.chunk_size
        chunk_overlap = settings.storage.chunk_overlap
        if extension in CodeLoader.CODE_EXTENSIONS:
            return CodeSplitter(extension=extension, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        return RecursiveSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    async def process(self, file_path: str | Path, document: Document) -> list[DocumentChunk]:
        p = Path(file_path)
        loader = self.loader_registry.get_loader(p)
        if not loader:
            raise ValueError(f"No loader for file: {p.suffix}")

        raw_text = await loader.load(p)
        cleaned = self.cleaner.clean(raw_text)
        metadata = self.extractor.extract(p, cleaned)

        splitter = self._get_splitter(p.suffix)
        chunks = splitter.split(document.id, cleaned, metadata)

        texts = [c.content for c in chunks]
        embeddings = await self.embedding_provider.embed(texts)
        for chunk, emb in zip(chunks, embeddings):
            chunk.embedding = emb

        return chunks
