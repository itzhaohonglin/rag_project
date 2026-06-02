import re

from backend.domain.document import DocumentChunk
from backend.ingestion.splitter.base import Splitter


class SemanticSplitter(Splitter):
    """Split by semantic boundaries (paragraphs, sections)."""

    def split(self, document_id: str, text: str, metadata: dict | None = None) -> list[DocumentChunk]:
        sections = re.split(r"\n#{1,6}\s+|\n---+\n", text)
        chunks: list[DocumentChunk] = []
        buffer = ""
        idx = 0
        for section in sections:
            section = section.strip()
            if not section:
                continue
            if len(buffer) + len(section) < self.chunk_size:
                buffer += f"\n{section}" if buffer else section
            else:
                if buffer:
                    chunks.append(DocumentChunk(
                        document_id=document_id, content=buffer,
                        chunk_index=idx, metadata=metadata or {},
                    ))
                    idx += 1
                buffer = section
        if buffer:
            chunks.append(DocumentChunk(
                document_id=document_id, content=buffer,
                chunk_index=idx, metadata=metadata or {},
            ))
        return chunks
