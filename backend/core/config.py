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


class CRAGConfig(BaseSettings):
    enabled: bool = False   # 默认关闭，API 传 crag=true 可覆盖

    model_config = SettingsConfigDict(extra="ignore")


class RetrievalConfig(BaseSettings):
    top_k: int = 10
    score_threshold: float = 0.0
    rerank_enabled: bool = True
    rerank_model: str = "BAAI/bge-reranker-v2-m3"
    retrieval_mode: str = "hybrid"  # dense | sparse | hybrid
    hybrid_dense_weight: float = 0.7
    hybrid_sparse_weight: float = 0.3

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
    """
    顶层配置模型，聚合所有子模块配置。

    ── 加载流程（低 → 高覆盖）────────────────────────────
      1. config/default.yaml         — 默认值
      2. config/{env}.yaml           — 环境覆盖（development/production）
      3. .env 文件（RAG_LLM__OPENAI__API_KEY 等）
      4. cls(**kwargs) 实例化        — ⬅ 至此是 load() 手动做的事
      5. BaseSettings.__init__ 隐式扫描 os.environ
         （匹配 RAG_ 前缀的变量）     — ⬅ 这是 pydantic-settings 框架自动做的

    ── 约定 ──────────────────────────────────────────────
      - 环境变量前缀：RAG_
      - 嵌套分隔符：__（双层下划线）
        例：RAG_LLM__OPENAI__API_KEY → llm.openai.api_key
      - load() 类方法整合上述来源后实例化

    ── 参数（按字段声明顺序）───────────────────────────────

    app:        应用基本信息（名称、版本、调试开关、监听地址和端口）
                 对应 config 中 app.name / app.version / app.debug / app.host / app.port

    milvus:     Milvus 向量数据库连接和集合配置
    database:   PostgreSQL 关系数据库连接池配置
    llm:        大语言模型选择（openai / local）及各 provider 参数
    embedding:  向量嵌入模型选择及维度配置
    celery:     异步任务队列的 broker / backend 地址
    redis:      Redis 缓存连接参数
    retrieval:  检索策略参数（top_k、阈值、重排序开关及模型）
    crag:       Corrective RAG 配置（默认开关）
    storage:    文件上传目录和文档分块参数
    """
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
    crag: CRAGConfig = CRAGConfig()
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
                        try:
                            for p in parts[:-1]:
                                target = target.setdefault(p, {})
                        except AttributeError:
                            # 路径中间节点不是 dict（如 YAML 标量字段被当路径走），跳过
                            continue
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
