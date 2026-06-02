import re
from pathlib import Path


class MetadataExtractor:
    def extract(self, file_path: str | Path, content: str) -> dict:
        p = Path(file_path)
        metadata = {
            "filename": p.name,
            "extension": p.suffix.lower(),
            "file_size": p.stat().st_size if p.exists() else 0,
        }
        title = self._extract_title(content)
        if title:
            metadata["title"] = title
        return metadata

    @staticmethod
    def _extract_title(content: str) -> str | None:
        lines = content.strip().split("\n")
        for line in lines[:20]:
            line = line.strip()
            if line.startswith("# ") or line.startswith("title:"):
                return line.lstrip("# title:").strip()
            if line and len(line) < 200:
                return line
        return None
