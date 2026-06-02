<p align="center">
  <h1 align="center">RAG Project</h1>
  <p align="center">企业级检索增强生成系统 · Enterprise RAG System</p>
  <p align="center">
    <img src="https://img.shields.io/badge/Python-3.11-blue?logo=python"/>
    <img src="https://img.shields.io/badge/FastAPI-0.110-teal?logo=fastapi"/>
    <img src="https://img.shields.io/badge/Milvus-2.3-00A1EA?logo=milvus"/>
    <img src="https://img.shields.io/badge/PostgreSQL-16-4169E1?logo=postgresql"/>
    <img src="https://img.shields.io/badge/Celery-5.3-green?logo=celery"/>
    <img src="https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker"/>
  </p>
</p>

---

## 项目简介

**RAG Project** 是一个基于 Miniconda 的企业级检索增强生成（Retrieval-Augmented Generation）系统脚手架，采用经典分层架构设计。支持多领域文档检索、智能问答和对话管理，开箱即用，适合作为企业知识库、智能客服、文档检索等场景的基础设施。

## 功能特性

- **多格式文档支持** — 支持 TXT、PDF、Markdown、Word、Excel、PPT、代码文件等 30+ 格式
- **分层分块策略** — 递归字符分块、语义分块、代码感知分块，适配不同文档类型
- **混合检索** — 向量检索 + 关键词打分融合，提升检索精准度
- **重排序** — 基于 Cross-Encoder 的重排序，进一步优化相关性
- **多 Provider 嵌入** — 支持 OpenAI Embedding 和本地模型（BGE/GTE 等）热切换
- **多 Provider LLM** — 支持 OpenAI、通义千问、本地 vLLM 等 OpenAI 兼容接口
- **对话管理** — 多轮对话上下文管理，滑动窗口记忆
- **缓存加速** — LRU 检索结果缓存，减少重复计算
- **异步处理** — Celery 后台异步文档解析和向量化
- **Docker 部署** — 完整 Docker Compose 编排，开发/生产双模式

## 架构设计

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

### RAG 流程

```
用户查询 → 查询路由 → 查询重写 → 向量检索 → 重排序
         → 上下文组装 → LLM 生成 → 返回结果(含引用)
```

## 技术栈

| 组件 | 技术 | 说明 |
|------|------|------|
| 运行时 | Python 3.11 + Miniconda | 环境管理 |
| Web 框架 | **FastAPI** | 高性能异步 API |
| 向量数据库 | **Milvus** | 企业级分布式向量检索 |
| 关系数据库 | **PostgreSQL 16** | 元数据/会话存储 |
| ORM | **SQLAlchemy 2.0** + Alembic | 数据库迁移 |
| LLM | **OpenAI** / 本地 vLLM | 混合方案 |
| 嵌入 | OpenAI Embedding / BGE | 热切换 |
| 任务队列 | **Celery** + Redis | 异步文档处理 |
| 容器化 | **Docker Compose** | 编排部署 |
| 配置管理 | pydantic-settings | 环境感知配置 |

## 快速开始

### 前置条件

- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) 或 Conda
- Docker & Docker Compose（生产部署）

### 1. 创建环境

```bash
conda env create -f environment.yaml
conda activate rag-project
```

### 2. 配置

编辑 `config/` 目录下的 YAML 配置文件，按需修改以下连接信息：

- `milvus.host` — Milvus 地址
- `database.url` — PostgreSQL 连接串
- `llm.openai.api_key` — OpenAI API Key
- `redis.host` — Redis 地址

支持通过环境变量覆盖：`RAG__MILVUS__HOST=localhost`

### 3. 启动服务

```bash
# 开发模式（热重载）
bash scripts/run.sh development

# 生产模式
bash scripts/run.sh production
```

### 4. 初始化数据库

```bash
python scripts/init_db.py
```

## API 文档

服务启动后访问 `http://localhost:8000/docs` 查看 Swagger 文档。

### 端点概览

| 方法 | 端点 | 说明 |
|------|------|------|
| GET | `/api/v1/health` | 健康检查 |
| POST | `/api/v1/documents` | 上传文档 |
| GET | `/api/v1/documents` | 文档列表（分页） |
| GET | `/api/v1/documents/{id}` | 文档详情 |
| DELETE | `/api/v1/documents/{id}` | 删除文档 |
| POST | `/api/v1/retrieval/query` | 检索查询 |
| POST | `/api/v1/conversations` | 创建对话 |
| POST | `/api/v1/conversations/{id}/messages` | 发送消息（含 RAG） |
| GET | `/api/v1/conversations/{id}` | 获取对话历史 |

### 响应格式

```json
{
  "code": 0,
  "data": {},
  "message": "ok"
}
```

## 项目结构

```
rag-project/
├── backend/
│   ├── api/                    # API 网关层
│   │   ├── main.py             # FastAPI 入口
│   │   ├── dependencies.py     # 依赖注入
│   │   ├── middleware/         # 认证/限流/日志
│   │   ├── routes/             # 路由
│   │   └── schemas/            # Pydantic 模型
│   ├── core/                   # 业务服务层
│   │   ├── rag_engine.py       # RAG 核心编排
│   │   ├── retriever/          # 检索器
│   │   ├── query/              # 查询处理
│   │   ├── generator/          # LLM 调用
│   │   └── memory/             # 记忆/缓存
│   ├── domain/                 # 领域模型
│   ├── ingestion/              # 数据处理层
│   │   ├── loader/             # 文档加载器
│   │   ├── splitter/           # 分块策略
│   │   ├── processor/          # 流水线
│   │   └── embedding/          # 向量化
│   ├── storage/                # 数据存储层
│   │   ├── vector_store/       # Milvus
│   │   ├── relational_db/      # PostgreSQL
│   │   └── file_store/         # 文件存储
│   └── workers/                # Celery 后台任务
├── config/                     # 配置文件
├── docker/                     # Docker 编排
├── tests/                      # 测试
├── scripts/                    # 运维脚本
├── environment.yaml            # Conda 环境
└── pyproject.toml              # 项目元数据
```

## 测试

```bash
# 运行所有测试
pytest tests/

# 运行单元测试
pytest tests/unit/ -v

# 带覆盖率
pytest --cov=backend tests/
```

## Docker 部署

```bash
# 生产部署
docker compose -f docker/docker-compose.yaml up -d

# 开发部署（热重载）
docker compose -f docker/docker-compose.dev.yaml up -d
```

包含服务：
- **api** — FastAPI 应用
- **worker** — Celery 异步任务
- **milvus** — 向量数据库
- **postgres** — 关系数据库
- **redis** — 缓存/消息队列

## 许可证

MIT License
