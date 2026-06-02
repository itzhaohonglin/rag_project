from abc import ABC, abstractmethod
from pathlib import Path


class FileStore(ABC):
    @abstractmethod
    async def save(self, file_path: str | Path, content: bytes) -> str:
        """Save file, return storage path."""
        ...

    @abstractmethod
    async def read(self, storage_path: str) -> bytes:
        """Read file content."""
        ...

    @abstractmethod
    async def delete(self, storage_path: str) -> bool:
        """Delete file."""
        ...

    @abstractmethod
    async def exists(self, storage_path: str) -> bool:
        """Check if file exists."""
        ...
