from backend.domain.document import DocumentChunk
from backend.ingestion.splitter.base import Splitter


class RecursiveSplitter(Splitter):
    SEPARATORS = ["\n\n", "\n", "。", ". ", " ", ""]

    def split(self, document_id: str, text: str, metadata: dict | None = None) -> list[DocumentChunk]:
        chunks: list[DocumentChunk] = []
        current = text
        for sep in self.SEPARATORS:
            if len(current) <= self.chunk_size:
                break
            current = self._split_by_sep(current, sep)
        for i, chunk_text in enumerate(self._merge_chunks(current)):
            chunks.append(DocumentChunk(
                document_id=document_id,
                content=chunk_text.strip(),
                chunk_index=i,
                metadata=metadata or {},
            ))
        return chunks

    def _split_by_sep(self, text: str, separator: str) -> str:
        if not separator:
            return text
        parts = text.split(separator)
        lines = []
        current_line = ""
        for part in parts:
            candidate = f"{current_line}{separator}{part}" if current_line else part
            if len(candidate) <= self.chunk_size or not current_line:
                current_line = candidate
            else:
                lines.append(current_line)
                current_line = part
        if current_line:
            lines.append(current_line)
        return "\n".join(lines)

    def _merge_chunks(self, text: str) -> list[str]:
        chunks = []
        start = 0
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunks.append(text[start:end])
            start = end - self.chunk_overlap if end < len(text) else end
        return chunks
