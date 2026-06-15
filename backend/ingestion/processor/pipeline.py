from pathlib import Path

from backend.core.config import settings
from backend.domain.document import Document, DocumentChunk
from backend.domain.enums import DocumentStatus, DocumentType
from backend.ingestion.embedding.base import EmbeddingProvider
from backend.ingestion.embedding.bm25_sparse import Bm25SparseEmbedding
from backend.ingestion.loader.base import Loader
from backend.ingestion.loader.code_loader import CodeLoader, LoaderRegistry
from backend.ingestion.processor.cleaner import TextCleaner
from backend.ingestion.processor.extractor import MetadataExtractor
from backend.ingestion.splitter.base import Splitter
from backend.ingestion.splitter.code_splitter import CodeSplitter
from backend.ingestion.splitter.recursive_splitter import RecursiveSplitter
from backend.ingestion.splitter.semantic_splitter import SemanticSplitter


class IngestionPipeline:
    """文档接入流水线：加载→清洗→提元数据→分块→向量化。

    一条调用走完原始文件到可检索 chunk 的全部加工流程，
    输出已挂好稠密+稀疏双向量的 DocumentChunk 列表。
    """

    def __init__(
            self,
            embedding_provider: EmbeddingProvider,
            sparse_embedding_provider: Bm25SparseEmbedding | None = None,
            loader_registry: LoaderRegistry | None = None,
            cleaner: TextCleaner | None = None,
            extractor: MetadataExtractor | None = None,
    ):
        self.embedding_provider = embedding_provider
        self.sparse_embedding_provider = sparse_embedding_provider
        self.loader_registry = loader_registry or LoaderRegistry()
        self.cleaner = cleaner or TextCleaner()
        self.extractor = extractor or MetadataExtractor()

    @staticmethod
    def _get_splitter(extension: str) -> Splitter:
        """按文件后缀选分块策略：代码文件走 CodeSplitter，其他走 RecursiveSplitter。"""
        chunk_size = settings.storage.chunk_size
        chunk_overlap = settings.storage.chunk_overlap
        if extension in CodeLoader.CODE_EXTENSIONS:
            return CodeSplitter(extension=extension, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        return RecursiveSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    async def process(self, file_path: str | Path, document: Document) -> list[DocumentChunk]:
        """执行完整加工：加载→清洗→提元数据→分块→稠密嵌入→稀疏嵌入。"""
        p = Path(file_path)
        loader = self.loader_registry.get_loader(p)
        if not loader:
            raise ValueError(f"没有找到能处理 {p.suffix} 文件的 Loader")

        raw_text = await loader.load(p)
        cleaned = self.cleaner.clean(raw_text)
        metadata = self.extractor.extract(p, cleaned)

        chunks = self._get_splitter(p.suffix).split(document.id, cleaned, metadata)

        texts = [c.content for c in chunks]
        embeddings = await self.embedding_provider.embed(texts)
        for chunk, emb in zip(chunks, embeddings):
            chunk.embedding = emb

        if self.sparse_embedding_provider:
            sparse_embeddings = self.sparse_embedding_provider.compute_embeddings(texts)
            for chunk, spar in zip(chunks, sparse_embeddings):
                chunk.sparse_embedding = spar

        return chunks
