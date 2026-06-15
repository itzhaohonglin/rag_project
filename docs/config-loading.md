# 配置加载机制

## 一句话

`pydantic-settings` + 自定义 `load()`，四层优先级层层覆盖。

## 优先级链（低 → 高）

```
class 默认值（代码里写死的，保证零配置能跑）
     ↓
config/default.yaml + config/{env}.yaml
     ↓
.env 文件
     ↓
系统环境变量（最高优先级）
```

## 加载流程（`Settings.load()`，config.py:147）

```
load(env="development")
    │
    ├─ 1. 从 config/default.yaml 读基础配置
    ├─ 2. 从 config/{env}.yaml 读环境特定配置（合并到第一层）
    ├─ 3. 从项目根 .env 读嵌套变量（手动解析 RAG_LLM__OPENAI__API_KEY 这种格式）
    └─ 4. 传 kwargs 给 Settings(**kwargs)
         └─ BaseSettings 自动捕获系统环境变量（同名覆盖）
```

## 映射规则

系统环境变量使用 `RAG_` 前缀 + `__` 嵌套分隔符：

| 环境变量 | 对应配置字段 |
|----------|-------------|
| `RAG_DATABASE__URL` | `settings.database.url` |
| `RAG_LLM__PROVIDER` | `settings.llm.provider` |
| `RAG_LLM__OPENAI__API_KEY` | `settings.llm.openai.api_key` |
| `RAG_MILVUS__HOST` | `settings.milvus.host` |

## 数据结构

```
Settings
 ├── app        AppConfig          (name, version, debug, host, port)
 ├── milvus     MilvusConfig       (host, port, collection)
 ├── database   DatabaseConfig     (url, pool_size, max_overflow)
 ├── llm        LLMConfig          (provider, openai={...}, local={...})
 ├── embedding  EmbeddingConfig    (provider, dimensions, openai={...}, local={...})
 ├── celery     CeleryConfig       (broker_url, result_backend)
 ├── redis      RedisConfig        (host, port, db)
 ├── retrieval  RetrievalConfig    (top_k, score_threshold, rerank_enabled, rerank_model)
 └── storage    StorageConfig      (upload_dir, chunk_size, chunk_overlap)
```

## 关键点

- class 字段的默认值（如 `host: str = "localhost"`）只是兜底，YAML 或 env 随便哪个配了都会覆盖
- `.env` 是手动解析的（`load()` 里逐行读），而不是用 `pydantic-settings` 的 `env_file` 参数——因为后者不支持 `__` 嵌套映射
- `SettingsConfigDict(env_prefix="RAG_", env_nested_delimiter="__")` 让 `BaseSettings` 自动读取系统环境变量
- `get_settings()` 以单例模式导出 `settings`，整个项目只需导入这一个实例
