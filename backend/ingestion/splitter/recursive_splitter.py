from backend.domain.document import DocumentChunk
from backend.ingestion.splitter.base import Splitter


class RecursiveSplitter(Splitter):
    """递归分块器：按分隔符优先级逐级切分，保语义紧凑。

    流程：
      按分隔符数组 ["\\n\\n", "\\n", "。", ". ", " ", ""]
      从前到后逐级尝试：
        先用段落切，片段超 chunk_size 则往下级走；
        再用句子切，还超则按词切，最后按字符硬切。

      _split_by_sep 把同级片段按 chunk_size 合并成行，
      _merge_chunks 再把行按 chunk_size 切块、首尾重叠。
    """

    SEPARATORS = ["\n\n", "\n", "。", ". ", " ", ""]

    def split(self, document_id: str, text: str, metadata: dict | None = None) -> list[DocumentChunk]:
        """按分隔符优先级逐级切分到 chunk_size 以内，再合并成块。"""
        current = text
        for sep in self.SEPARATORS:
            if len(current) <= self.chunk_size:
                break
            current = self._split_by_sep(current, sep)

        chunks = [
            DocumentChunk(document_id=document_id, content=t.strip(), chunk_index=i, metadata=metadata or {})
            for i, t in enumerate(self._merge_chunks(current))
        ]
        return chunks

    def _split_by_sep(self, text: str, separator: str) -> str:
        """用分隔符切分，相邻短片段合并到不超过 chunk_size。"""
        if not separator:
            return text
        parts = text.split(separator)
        lines = []
        buf = ""
        for part in parts:
            candidate = f"{buf}{separator}{part}" if buf else part
            if len(candidate) <= self.chunk_size or not buf:
                buf = candidate
            else:
                lines.append(buf)
                buf = part
        if buf:
            lines.append(buf)
        return "\n".join(lines)

    def _merge_chunks(self, text: str) -> list[str]:
        """按 chunk_size 切块，相邻块重叠 chunk_overlap 个字符。"""
        chunks = []
        start = 0
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunks.append(text[start:end])
            start = end - self.chunk_overlap if end < len(text) else end
        return chunks
