from pathlib import Path

from backend.ingestion.loader.base import Loader


class TextLoader(Loader):
    async def load(self, file_path: str | Path) -> str:
        return Path(file_path).read_text(encoding="utf-8")

    def supported_extensions(self) -> set[str]:
        return {".txt", ".log", ".csv"}
