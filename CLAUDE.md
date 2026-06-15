# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# 环境
conda env create -f environment.yaml    # 首次创建
conda activate rag-project

# 运行（开发模式）
uvicorn backend.api.main:app --reload

# 或通过脚本（含 Docker 依赖启动）
bash scripts/run.sh development

# 测试
pytest tests/                              # 全部
pytest tests/unit/ -v                      # 单元
pytest tests/integration/ -v               # 集成
pytest --cov=backend tests/                # 覆盖率
pytest tests/unit/test_xxx.py -v           # 单文件
pytest tests/unit/test_xxx.py::test_func   # 单用例

# 代码质量
ruff check backend/                        # Lint 检查
ruff format backend/                       # 自动格式化
ruff format --check backend/               # 格式检查
mypy backend/                              # 类型检查

# 数据库迁移
alembic revision --autogenerate -m "desc"
alembic upgrade head

# Docker 部署
docker compose -f docker/docker-compose.yaml up -d       # 生产
docker compose -f docker/docker-compose.dev.yaml up -d   # 开发

# 初始化数据库表
python scripts/init_db.py
```

## Architecture Overview

四层分层架构 + 跨层领域模型，RAG 流程按 pipeline 编排。

```
┌─────────────────────────────────────────────────────────────────────┐
│  API 层 (backend/api/)                                             │
│  main.py → routes/ (health, documents, retrieval, conversation)     │
│  dependencies.py (DI), middleware/ (auth, logging), schemas/        │
├─────────────────────────────────────────────────────────────────────┤
│  Core 层 (backend/core/)                                           │
│  RAGEngine → QueryRouter → QueryProcessor → Retriever → ReRanker   │
│            → ContextBuilder → LLMClient                             │
│  memory/ (ConversationMemory, LRUCache), generator/                 │
├─────────────────────────────────────────────────────────────────────┤
│  Ingestion 层 (backend/ingestion/)                                 │
│  IngestionPipeline → Loader → Cleaner → Splitter → Embedding       │
│  loader/ (text, pdf, markdown, office, code), splitter/             │
│  embedding/ (openai, local, bm25_sparse)                            │
├─────────────────────────────────────────────────────────────────────┤
│  Storage 层 (backend/storage/)                                     │
│  vector_store/ (Milvus - 稠密+稀疏双向量),                          │
│  relational_db/ (PostgreSQL - 元数据/对话),                          │
│  file_store/ (本地文件系统 - 原始文档)                               │
└─────────────────────────────────────────────────────────────────────┘
  Domain 层 (backend/domain/) - 纯 Python 数据类，贯穿所有层
```

### RAG 流程

```
查询 → QueryRouter(直答/检索)
     → QueryProcessor(重写)
     → HybridRetriever(稠密向量 + BM25 稀疏向量加权融合, 默认 weight 7:3)
     → ReRanker(Cross-Encoder 重排序)
     → ContextBuilder(组装上下文)
     → LLMClient(生成)
     → 返回 answer + citations
```

### 关键设计决策

- **配置系统** — `core/config.py` 用 `pydantic-settings`，从 `config/default.yaml` + `{env}.yaml` 加载，环境变量前缀 `RAG__` 可覆盖任意字段。`Settings.load()` 类方法统一整合 YAML/.env/环境变量
- **依赖注入** — FastAPI `Depends()` 统一管理，`dependencies.py` 中 RAGEngine/MilvusStore/EmbeddingProvider 是懒加载单例
- **响应格式** — 所有 API 返回 `{"code": 0, "data": ..., "message": "ok"}`（`errors.py:success_response()`），错误分两路：`RAGException`（业务异常，code 含错误类型）和 `APIError`（HTTP 状态码异常），全局异常处理器注册在 `main.py`
- **错误体系** — `domain/exceptions.py:RAGException` 是基类，子类有 `DocumentNotFoundError`、`DocumentProcessingError`、`EmbeddingError`、`LLMError`、`ConfigurationError`，每个有唯一 `code` 字符串
- **Milvus 双向量存储** — `storage/vector_store/schema.py`：每个 chunk 同时存稠密向量（FLOAT_VECTOR, 1536 维, COSINE 度量）和稀疏向量（SPARSE_FLOAT_VECTOR, IP 度量），IVF_FLAT + SPARSE_INVERTED_INDEX 双索引
- **文档处理** — `LoaderRegistry` 按文件扩展名自动匹配 Loader，按扩展名自动选择 Splitter（代码文件用 CodeSplitter，其他用 RecursiveSplitter），IngestionPipeline 串接 Loader → Cleaner → MetadataExtractor → Splitter → 稠密嵌入 → BM25 稀疏嵌入
- **热切换** — LLM 和 Embedding 支持 `provider: openai | local` 配置切换

### 项目目录要点

| 路径 | 内容 |
|------|------|
| `backend/api/routes/` | health, documents, retrieval, conversation 四个路由 |
| `backend/api/errors.py` | APIError 异常 + success_response/error_response 统一构造器 |
| `backend/api/dependencies.py` | 所有单例和依赖注入集中管理 |
| `backend/domain/enums.py` | DocumentStatus, DocumentType, ChunkStrategy, RetrievalMode, MessageRole |
| `backend/domain/exceptions.py` | RAGException 异常体系，含唯一 code 标识 |
| `backend/core/retriever/` | vector_retriever (稠密), sparse_retriever (BM25), hybrid_retriever (加权融合), re_ranker (Cross-Encoder) |
| `backend/storage/vector_store/schema.py` | Milvus 集合 schema：chunk_id/doc_id/content/双向量 |
| `config/` | default.yaml + development.yaml + production.yaml |
| `docker/` | Dockerfile + docker-compose.yaml（含 milvus/etcd/minio/postgres/redis） |
| `gradio_app.py` | Gradio 交互式 DEMO 前端 |
| `tests/` | unit/（纯逻辑 mock DB）、integration/（依赖真实 DB） |