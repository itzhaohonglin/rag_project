from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class MilvusConfig(BaseSettings):
    host: str = "localhost"
    port: int = 19530
    collection: str = "document_chunks"

    model_config = SettingsConfigDict(extra="ignore")


class DatabaseConfig(BaseSettings):
    url: str = "postgresql+psycopg2://postgres:postgres@localhost:5432/rag_project"
    pool_size: int = 10
    max_overflow: int = 20


class OpenAILLMConfig(BaseSettings):
    api_key: str = ""
    model: str = "gpt-4o-mini"
    temperature: float = 0.1
    max_tokens: int = 2048


class LocalLLMConfig(BaseSettings):
    base_url: str = "http://localhost:8001/v1"
    model: str = ""
    temperature: float = 0.1
    max_tokens: int = 2048


class LLMConfig(BaseSettings):
    provider: Literal["openai", "local"] = "openai"
    openai: OpenAILLMConfig = OpenAILLMConfig()
    local: LocalLLMConfig = LocalLLMConfig()


class EmbeddingConfig(BaseSettings):
    provider: Literal["openai", "local"] = "openai"
    dimensions: int = 1536


class RedisConfig(BaseSettings):
    host: str = "localhost"
    port: int = 6379
    db: int = 0

    @property
    def url(self) -> str:
        return f"redis://{self.host}:{self.port}/{self.db}"


class CeleryConfig(BaseSettings):
    broker_url: str = "redis://localhost:6379/0"
    result_backend: str = "redis://localhost:6379/0"


class RetrievalConfig(BaseSettings):
    top_k: int = 10
    score_threshold: float = 0.0
    rerank_enabled: bool = True
    rerank_model: str = "BAAI/bge-reranker-v2-m3"


class StorageConfig(BaseSettings):
    upload_dir: str = "./data/uploads"
    chunk_size: int = 512
    chunk_overlap: int = 64


class AppConfig(BaseSettings):
    name: str = "rag-project"
    version: str = "0.1.0"
    debug: bool = False
    host: str = "0.0.0.0"
    port: int = 8000


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="RAG_",
        env_nested_delimiter="__",
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
    def load(cls, env: str = "development") -> "Settings":
        import yaml
        base_path = Path(__file__).parent.parent.parent / "config"
        default_path = base_path / "default.yaml"
        env_path = base_path / f"{env}.yaml"

        settings = cls()
        if default_path.exists():
            with open(default_path) as f:
                data = yaml.safe_load(f)
                for key, value in data.items():
                    sub = getattr(settings, key, None)
                    if sub:
                        for sub_key, sub_val in value.items():
                            if hasattr(sub, sub_key):
                                setattr(sub, sub_key, sub_val)
        if env_path.exists():
            with open(env_path) as f:
                data = yaml.safe_load(f)
                for key, value in data.items():
                    sub = getattr(settings, key, None)
                    if sub:
                        for sub_key, sub_val in value.items():
                            if hasattr(sub, sub_key):
                                setattr(sub, sub_key, sub_val)
        return settings


settings = Settings.load()
