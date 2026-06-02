# 企业级 RAG 项目设计文档

## 概述

基于 Miniconda 的企业级 RAG（检索增强生成）项目脚手架，支持多领域文档检索，采用经典分层架构设计。

## 技术选型

| 组件 | 技术 | 说明 |
|------|------|------|
| 语言 | Python 3.11 | |
| Web 框架 | FastAPI | API 网关层 |
| 向量数据库 | Milvus | 企业级向量检索 |
| 关系数据库 | PostgreSQL | 元数据/会话存储 |
| LLM | 混合方案 | 本地 embedding + 云端 LLM |
| ORM | SQLAlchemy + Alembic | 数据库迁移 |
| 配置管理 | pydantic-settings | 环境感知配置 |
| 任务队列 | Celery + Redis | 异步文档处理 |
| 容器化 | Docker Compose | 编排部署 |

## 架构设计：经典分层架构

```
┌──────────────────────────────────────────┐
│             API 网关层                    │  FastAPI REST API
│  routes / middleware / schemas / errors  │
├──────────────────────────────────────────┤
│             业务服务层                    │  RAG 核心逻辑
│  rag_engine / retriever / generator      │
├──────────────────────────────────────────┤
│             数据处理层                    │  文档流水线
│  loader / splitter / processor / embed   │
├──────────────────────────────────────────┤
│             数据存储层                    │  持久化
│  vector_store / relational_db / file     │
└──────────────────────────────────────────┘
     领域模型层 (贯穿所有层，纯 Python 数据类)
```

## 目录结构

```
rag_project/
├── backend/
│   ├── api/                    # API 网关层
│   │   ├── main.py             # FastAPI 入口
│   │   ├── dependencies.py     # 依赖注入
│   │   ├── middleware/         # 认证/限流/日志
│   │   ├── routes/             # 路由 (documents/retrieval/conversation/health)
│   │   ├── schemas/            # Pydantic 请求/响应模型
│   │   └── errors.py           # 统一错误处理
│   ├── core/                   # 业务服务层
│   │   ├── rag_engine.py       # RAG 核心编排
│   │   ├── retriever/          # 检索 (向量/混合/重排序)
│   │   ├── query/              # 查询理解/重写/路由
│   │   ├── generator/          # LLM 客户端/提示词管理/上下文组装
│   │   └── memory/             # 对话记忆/缓存
│   ├── domain/                 # 领域模型层
│   │   ├── document.py         # Document, DocumentChunk
│   │   ├── embedding.py        # EmbeddingVector, EmbeddingConfig
│   │   ├── retrieval.py        # Query, RetrievalResult
│   │   ├── conversation.py     # Conversation, Message, Citation
│   │   ├── enums.py            # 枚举
│   │   └── exceptions.py       # 自定义异常
│   ├── ingestion/              # 数据处理层
│   │   ├── loader/             # 文档加载器 (TXT/PDF/MD/Office/Code)
│   │   ├── splitter/           # 分块策略 (递归/语义/代码感知)
│   │   ├── processor/          # 流水线编排/清洗/元数据提取
│   │   └── embedding/          # 向量化 (OpenAI/本地模型)
│   ├── storage/                # 数据存储层
│   │   ├── vector_store/       # Milvus 实现
│   │   ├── relational_db/      # PostgreSQL + SQLAlchemy + Alembic
│   │   └── file_store/         # 本地/S3 文件存储
│   └── workers/                # Celery 后台任务
├── config/                     # YAML 配置文件
├── docs/                       # 文档
├── scripts/                    # 运维脚本
├── tests/                      # 测试 (unit/integration/e2e)
├── docker/                     # Dockerfile + docker-compose
├── environment.yaml            # Miniconda 环境定义
├── pyproject.toml              # 项目元数据和依赖
└── README.md
```

## 各层详细设计

### 1. 领域模型层 (domain/)

纯 Python 数据类，零外部依赖。定义核心实体：
- **Document**: 原始文档元数据（id, 文件名, 来源, 状态, 上传时间, 文档类型等）
- **DocumentChunk**: 分块单元（id, document_id, 文本, 向量, 元数据, 序号）
- **Query**: 用户查询（原始文本, 重写后文本, 检索参数, 过滤器）
- **RetrievalResult**: 检索结果（chunks, 相关性分, 来源引用, 耗时统计）
- **Conversation**: 对话会话（id, 消息列表, 上下文窗口, 创建时间）
- **Message**: 单条消息（role, content, citations, timestamp）
- **Citation**: 引用来源（chunk_id, document_id, 原文片段, 页码）

### 2. 数据处理层 (ingestion/)

采用 Pipeline 设计模式，处理流程：

```
Upload → Loader → Cleaner → Splitter → Extractor → Embedding → Store
```

- **抽象基类**: BaseLoader, BaseSplitter, EmbeddingProvider — 策略模式实现
- **自动路由**: 根据 MIME 类型自动匹配合适的 Loader
- **分块策略**: 递归字符分块(默认)、语义分块(LLM辅助)、代码感知分块
- **嵌入切换**: 抽象 EmbeddingProvider 支持 OpenAI/local BGE/GTE 热切换
- **异步处理**: 长耗时任务通过 Celery worker 后台执行

### 3. 业务服务层 (core/)

RAG 核心流程编排：

```
Query → QueryProcessor(重写/扩展)
       → Retriever(向量检索/混合检索)
       → ReRanker(重排序)
       → ContextBuilder(窗口管理/压缩)
       → LLM(生成)
       → Response(含引用)
```

主要组件：
- **rag_engine.py**: 流程编排器，协调各组件调用
- **Retriever**: VectorRetriever(向量检索) / HybridRetriever(向量+BM25)
- **ReRanker**: 基于 Cross-Encoder 重排序
- **QueryProcessor**: 查询重写(多轮/少轮)、查询扩展
- **QueryRouter**: 意图路由 — 直接回答 vs 检索增强
- **LLMClient**: 统一 LLM API 客户端(OpenAI/通义千问/本地 vLLM)
- **PromptManager**: 提示词模板管理(版本化)
- **ContextBuilder**: 上下文组装、Token 窗口管理、检索压缩
- **ConversationMemory**: 滑动窗口记忆管理
- **Cache**: LRU 检索结果缓存

### 4. 数据存储层 (storage/)

三层隔离设计：
- **VectorStore 抽象**: 预留多向量库扩展点，当前实现 Milvus
- **Repository 模式**: DocumentRepo / ConversationRepo 封装 SQLAlchemy
- **FileStore 抽象**: 当前本地文件系统，预留 S3/MinIO 扩展

数据流：`FileStore(原始文件) → Ingestion Pipeline → VectorStore(向量) + PostgreSQL(元数据)`

### 5. API 网关层 (api/)

FastAPI 实现，设计原则：
- **版本化路由**: `/api/v1/` 前缀
- **依赖注入**: 通过 `Depends` 注入存储层/服务层
- **统一响应**: `{"code": 0, "data": ..., "message": "ok"}` 格式
- **认证**: API Key 或 JWT 中间件
- **错误处理**: 全局 Exception Handler

API 端点设计：

| 端点 | 方法 | 说明 |
|------|------|------|
| `/api/v1/health` | GET | 健康检查(含各组件状态) |
| `/api/v1/documents` | POST | 上传文档 |
| `/api/v1/documents` | GET | 文档列表(分页/过滤) |
| `/api/v1/documents/{id}` | GET | 文档详情 |
| `/api/v1/documents/{id}` | DELETE | 删除文档 |
| `/api/v1/retrieval/query` | POST | 检索查询 |
| `/api/v1/conversations` | POST | 创建对话 |
| `/api/v1/conversations/{id}/messages` | POST | 发送消息(含RAG) |
| `/api/v1/conversations/{id}` | GET | 获取对话历史 |

## 配置管理

使用 pydantic-settings 管理分级配置：
- `default.yaml`: 默认值
- 环境变量覆盖: `RAG__MILVUS__HOST=localhost`
- 环境特定: development/production 通过 `RAG_ENV` 切换

## 错误处理

标准错误格式:
```json
{
  "code": "DOCUMENT_NOT_FOUND",
  "message": "文档不存在",
  "detail": {"document_id": "xxx"}
}
```

## 测试策略

| 层级 | 测试对象 | 工具 |
|------|----------|------|
| Unit | 分块器/重排序器/领域模型 | pytest |
| Integration | 存储层/检索流水线 | pytest + testcontainers |
| E2E | 完整 RAG 流程 | pytest + docker-compose |

## Docker 部署

```yaml
services:
  api:        # FastAPI 应用
  milvus:     # 向量数据库
  postgres:   # 关系数据库
  redis:      # 任务队列
  worker:     # Celery worker
```

## YAGNI 排除清单

以下功能不在当前范围内，未来按需引入：
- 多租户隔离
- 用户权限 RBAC
- 文档版本管理
- A/B 测试框架
- 监控告警 (Prometheus/Grafana)
- 语义缓存
