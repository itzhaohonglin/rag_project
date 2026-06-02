# RAG Project

企业级 RAG（检索增强生成）系统，支持多领域文档检索。

## 架构

经典四层架构：API 网关层 → 业务服务层 → 数据处理层 → 数据存储层

- **API 网关层**: FastAPI REST API
- **业务服务层**: RAG 编排、检索、生成
- **数据处理层**: 文档加载、分块、向量化
- **数据存储层**: Milvus + PostgreSQL

## 快速开始

### 1. Miniconda 环境

```bash
conda env create -f environment.yaml
conda activate rag-project
```

### 2. 配置

复制 `config/default.yaml`，按需修改连接信息。

### 3. 启动服务

```bash
# 开发模式
bash scripts/run.sh development

# 生产模式
bash scripts/run.sh production
```

### 4. API 端点

| 端点 | 说明 |
|------|------|
| `GET /api/v1/health` | 健康检查 |
| `POST /api/v1/documents` | 上传文档 |
| `GET /api/v1/documents` | 文档列表 |
| `GET /api/v1/documents/{id}` | 文档详情 |
| `DELETE /api/v1/documents/{id}` | 删除文档 |
| `POST /api/v1/retrieval/query` | 检索查询 |
| `POST /api/v1/conversations` | 创建对话 |
| `POST /api/v1/conversations/{id}/messages` | 发送消息 |

## 测试

```bash
pytest tests/
```

## 项目结构

```
backend/
├── api/          # API 网关层
├── core/         # 业务服务层
├── domain/       # 领域模型
├── ingestion/    # 数据处理层
├── storage/      # 数据存储层
└── workers/      # 后台任务
```
