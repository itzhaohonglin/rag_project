from abc import ABC, abstractmethod
from pathlib import Path


class Loader(ABC):
    @abstractmethod
    async def load(self, file_path: str | Path) -> str:
        """Load document content, return plain text."""
        ...

    @abstractmethod
    def supported_extensions(self) -> set[str]:
        ...
