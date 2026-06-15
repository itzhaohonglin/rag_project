from pathlib import Path

from backend.ingestion.loader.base import Loader
from backend.ingestion.loader.text_loader import TextLoader
from backend.ingestion.loader.pdf_loader import PDFLoader
from backend.ingestion.loader.markdown_loader import MarkdownLoader
from backend.ingestion.loader.office_loader import WordLoader, ExcelLoader, PPTLoader


class CodeLoader(Loader):
    CODE_EXTENSIONS = {
        ".py", ".js", ".ts", ".jsx", ".tsx", ".java", ".go", ".rs",
        ".c", ".cpp", ".h", ".hpp", ".cs", ".rb", ".php", ".swift",
        ".kt", ".scala", ".sc", ".sh", ".bash", ".zsh", ".sql",
        ".yaml", ".yml", ".json", ".xml", ".toml", ".ini", ".cfg",
        ".html", ".css", ".scss", ".less", ".vue", ".svelte",
    }

    async def load(self, file_path: str | Path) -> str:
        return Path(file_path).read_text(encoding="utf-8")

    def supported_extensions(self) -> set[str]:
        return self.CODE_EXTENSIONS


class LoaderRegistry:
    """Auto-dispatch loaders by file extension."""

    def __init__(self):
        self._loaders: list[Loader] = [
            TextLoader(),
            PDFLoader(),
            MarkdownLoader(),
            WordLoader(),
            ExcelLoader(),
            PPTLoader(),
            CodeLoader(),
        ]

    def get_loader(self, file_path: str | Path) -> Loader | None:
        ext = Path(file_path).suffix.lower()
        for loader in self._loaders:
            if ext in loader.supported_extensions():
                return loader
        return None
