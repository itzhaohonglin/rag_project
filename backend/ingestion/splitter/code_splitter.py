import re

from backend.domain.document import DocumentChunk
from backend.ingestion.splitter.base import Splitter


class CodeSplitter(Splitter):
    CODE_SEPARATORS = {
        ".py": [r"\n\s*(?:def |class |@|async def )"],
        ".js": [r"\n\s*(?:function |class |const |let |var )"],
        ".ts": [r"\n\s*(?:function |class |interface |type |const |export )"],
        ".java": [r"\n\s*(?:public |private |protected |class |interface )"],
        ".go": [r"\n\s*(?:func |type |struct |interface )"],
        ".rs": [r"\n\s*(?:fn |struct |enum |impl |trait )"],
    }

    def __init__(self, extension: str = ".py", chunk_size: int = 512, chunk_overlap: int = 64):
        super().__init__(chunk_size, chunk_overlap)
        self.extension = extension

    def split(self, document_id: str, text: str, metadata: dict | None = None) -> list[DocumentChunk]:
        separators = self.CODE_SEPARATORS.get(self.extension, [r"\n\n"])
        pattern = "|".join(separators)
        parts = re.split(pattern, text) if pattern else [text]
        chunks: list[DocumentChunk] = []
        buffer = ""
        idx = 0
        for part in parts:
            if len(buffer) + len(part) < self.chunk_size:
                buffer += f"\n{part}" if buffer else part
            else:
                if buffer:
                    chunks.append(DocumentChunk(
                        document_id=document_id, content=buffer,
                        chunk_index=idx, metadata={"extension": self.extension, **(metadata or {})},
                    ))
                    idx += 1
                buffer = part
        if buffer:
            chunks.append(DocumentChunk(
                document_id=document_id, content=buffer,
                chunk_index=idx, metadata={"extension": self.extension, **(metadata or {})},
            ))
        return chunks
