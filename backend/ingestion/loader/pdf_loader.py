from pathlib import Path

import fitz  # PyMuPDF

from backend.ingestion.loader.base import Loader


class PDFLoader(Loader):
    async def load(self, file_path: str | Path) -> str:
        doc = fitz.open(str(file_path))
        pages = []
        for page in doc:
            pages.append(page.get_text())
        doc.close()
        return "\n\n".join(pages)

    def supported_extensions(self) -> set[str]:
        return {".pdf"}
