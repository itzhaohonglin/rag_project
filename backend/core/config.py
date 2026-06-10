from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class MilvusConfig(BaseSettings):
    host: str = "localhost"
    port: int = 19530
    collection: str = "document_chunks"

    model_config = SettingsConfigDict(extra="ignore")


class DatabaseConfig(BaseSettings):
    url: str = "postgresql+psycopg2://postgres:postgres@localhost:5432/rag_project"
    pool_size: int = 10
    max_overflow: int = 20

    model_config = SettingsConfigDict(extra="ignore")


class OpenAILLMConfig(BaseSettings):
    base_url: str = ""
    api_key: str = ""
    model: str = "gpt-4o-mini"
    temperature: float = 0.1
    max_tokens: int = 2048

    model_config = SettingsConfigDict(extra="ignore")


class LocalLLMConfig(BaseSettings):
    base_url: str = "http://localhost:8001/v1"
    model: str = ""
    temperature: float = 0.1
    max_tokens: int = 2048

    model_config = SettingsConfigDict(extra="ignore")


class LLMConfig(BaseSettings):
    provider: Literal["openai", "local"] = "openai"
    openai: OpenAILLMConfig = OpenAILLMConfig()
    local: LocalLLMConfig = LocalLLMConfig()

    model_config = SettingsConfigDict(extra="ignore")


class OpenAIEmbeddingConfig(BaseSettings):
    model: str = "text-embedding-3-small"

    model_config = SettingsConfigDict(extra="ignore")


class LocalEmbeddingConfig(BaseSettings):
    model_path: str = ""

    model_config = SettingsConfigDict(extra="ignore")


class EmbeddingConfig(BaseSettings):
    provider: Literal["openai", "local"] = "openai"
    dimensions: int = 1536
    openai: OpenAIEmbeddingConfig = OpenAIEmbeddingConfig()
    local: LocalEmbeddingConfig = LocalEmbeddingConfig()

    model_config = SettingsConfigDict(extra="ignore")


class RedisConfig(BaseSettings):
    host: str = "localhost"
    port: int = 6379
    db: int = 0

    model_config = SettingsConfigDict(extra="ignore")

    @property
    def url(self) -> str:
        return f"redis://{self.host}:{self.port}/{self.db}"


class CeleryConfig(BaseSettings):
    broker_url: str = "redis://localhost:6379/0"
    result_backend: str = "redis://localhost:6379/0"

    model_config = SettingsConfigDict(extra="ignore")


class RetrievalConfig(BaseSettings):
    top_k: int = 10
    score_threshold: float = 0.0
    rerank_enabled: bool = True
    rerank_model: str = "BAAI/bge-reranker-v2-m3"

    model_config = SettingsConfigDict(extra="ignore")


class StorageConfig(BaseSettings):
    upload_dir: str = "./data/uploads"
    chunk_size: int = 512
    chunk_overlap: int = 64

    model_config = SettingsConfigDict(extra="ignore")


class AppConfig(BaseSettings):
    name: str = "rag-project"
    version: str = "0.1.0"
    debug: bool = False
    host: str = "0.0.0.0"
    port: int = 8000

    model_config = SettingsConfigDict(extra="ignore")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="RAG_",
        env_nested_delimiter="__",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app: AppConfig = AppConfig()
    milvus: MilvusConfig = MilvusConfig()
    database: DatabaseConfig = DatabaseConfig()
    llm: LLMConfig = LLMConfig()
    embedding: EmbeddingConfig = EmbeddingConfig()
    celery: CeleryConfig = CeleryConfig()
    redis: RedisConfig = RedisConfig()
    retrieval: RetrievalConfig = RetrievalConfig()
    storage: StorageConfig = StorageConfig()

    @classmethod
    def _project_root(cls) -> Path:
        """项目根目录：config.py 在 backend/core/config.py，往上三层即根"""
        return Path(__file__).parent.parent.parent

    @classmethod
    def _env_file_path(cls) -> Path:
        """.env 绝对路径（项目根目录下）"""
        return cls._project_root() / ".env"

    @classmethod
    def load(cls, env: str = "development") -> "Settings":
        """加载配置，优先级：yaml < .env < 系统环境变量"""
        import os
        import yaml
        from pathlib import Path
        base_path = cls._project_root() / "config"

        # 收集 yaml 作为低优先级默认值
        kwargs: dict = {}
        for name in ("default.yaml", f"{env}.yaml"):
            path = base_path / name
            if path.exists():
                with open(path, encoding="utf-8") as f:
                    data = yaml.safe_load(f) or {}
                    for k, v in data.items():
                        if isinstance(v, dict):
                            kwargs.setdefault(k, {}).update(v)
                        else:
                            kwargs[k] = v

        # 读 .env，覆盖 yaml 值（处理 RAG_LLM__OPENAI__API_KEY 这类嵌套变量名）
        # 用绝对路径，不受工作目录影响（IDEA 等 IDE 中也能找到）
        env_file = cls._env_file_path()
        if env_file.exists():
            with open(env_file, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    key, _, val = line.partition("=")
                    key = key.strip()
                    val = val.strip().strip("\"'")
                    if key.startswith("RAG_"):
                        parts = key[4:].lower().split("__")
                        target = kwargs
                        for p in parts[:-1]:
                            target = target.setdefault(p, {})
                        if parts[-1]:
                            target[parts[-1]] = val

        return cls(**kwargs)


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        try:
            _settings = Settings.load()
        except ImportError:
            _settings = Settings()
    return _settings


settings = get_settings()
