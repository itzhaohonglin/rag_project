# Corrective RAG（CRAG）

## 概述

CRAG（Corrective Retrieval Augmented Generation）在标准 RAG 的检索与生成之间插入一道**相关性评估**环节。检索到的文档片段经过 LLM 打分过滤，不相关的被剔除，避免「垃圾进垃圾出」的问题。

核心思路来自论文 [Corrective Retrieval Augmented Generation](https://arxiv.org/abs/2401.15884)。

## 架构

本实现与现有流水线的关系：

```
┌─ 普通 RAG（默认）─────────────────────────────────────────────┐
│  检索 → ReRanker 重排序 → 全量 chunk 做上下文 → LLM 生成      │
└──────────────────────────────────────────────────────────────┘

┌─ CRAG 模式 ───────────────────────────────────────────────────┐
│  检索 → LLM 逐块评估相关性 ─→ 有相关块 → 过滤保留 → LLM 生成 │
│                            └→ 全不相关 → 直答（说不知道）     │
└──────────────────────────────────────────────────────────────┘
```

CRAG 使用 **LangGraph** 状态图编排三个节点：

```
retrieve ──→ evaluate ──→ generate
```

| 节点 | 组件 | 说明 |
|------|------|------|
| `retrieve` | `HybridRetriever` | 复用现有检索器，无改动 |
| `evaluate` | `RelevanceEvaluator` | LLM 批量打分，每段输出 0（不相关）/ 0.5（部分相关）/ 1（相关） |
| `generate` | `ContextBuilder` + `LLMClient` | 有相关块 → RAG 生成；没有 → 直答提示模型说不知道 |

## 新增文件

```
backend/core/crag/
├── __init__.py          # 空
├── state.py             # CRAGState TypedDict，定义图状态字段
├── evaluator.py         # RelevanceEvaluator，LLM 批量评估相关性
├── nodes.py             # 三个图节点函数
└── graph.py             # 构建 LangGraph StateGraph，依赖注入
```

## 改动文件

| 文件 | 改动 |
|------|------|
| `backend/core/config.py` | 新增 `CRAGConfig`（字段 `enabled: bool`），挂载到 `Settings.crag` |
| `backend/core/rag_engine.py` | `answer()` 新增 `crag` 参数（默认值取自 `settings.crag.enabled`），`True` 时走 LangGraph |
| `backend/api/schemas/retrieval.py` | `QueryRequest` 新增 `crag: bool = False` |
| `backend/api/routes/retrieval.py` | `answer()` 调用传递 `crag=req.crag` |
| `config/default.yaml` | 新增 `crag.enabled: false` |
| `pyproject.toml` | 新增依赖 `langgraph>=0.4.0` |

## 边界情况处理

- **chunk 列表为空**：跳过评估，直接走无上下文生成
- **LLM 评估失败**（超时/解析异常）：全标记为相关（保守 fallback，不丢弃任何文档）
- **评估返回格式异常**（非 JSON/长度不匹配）：全标记为相关
- **全不相关**：直答模式，prompt 附加说明「没有找到相关资料」

## 使用方式

### API 请求级控制

```json
POST /retrieval/query
{
  "query": "什么是 RAG？",
  "mode": "hybrid",
  "top_k": 10,
  "crag": true
}
```

### 全局开关

编辑 `config/default.yaml` 或对应环境的 yaml：

```yaml
crag:
  enabled: true
```

API 请求不传 `crag` 时，走此默认值。

## 依赖

- `langgraph>=0.4.0` — LangGraph 状态图引擎
- 其余复用现有组件：`HybridRetriever`、`LLMClient`、`ContextBuilder`、`PromptManager`
