import uuid
from pathlib import Path

from backend.storage.file_store.base import FileStore


class LocalFileStore(FileStore):
    def __init__(self, base_dir: str | Path):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    async def save(self, file_path: str | Path, content: bytes) -> str:
        ext = Path(file_path).suffix
        storage_name = f"{uuid.uuid4().hex}{ext}"
        target = self.base_dir / storage_name
        target.write_bytes(content)
        return str(target)

    async def read(self, storage_path: str) -> bytes:
        return Path(storage_path).read_bytes()

    async def delete(self, storage_path: str) -> bool:
        p = Path(storage_path)
        if p.exists():
            p.unlink()
            return True
        return False

    async def exists(self, storage_path: str) -> bool:
        return Path(storage_path).exists()
