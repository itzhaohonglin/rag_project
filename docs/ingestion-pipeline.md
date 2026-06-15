# IngestionPipeline 文档处理流水线

## 作用

`IngestionPipeline`（`backend/ingestion/processor/pipeline.py`）是文档接入的核心编排器，负责将原始文件串行经过 5 个阶段，输出带双向量的 `DocumentChunk` 列表，直接喂入 Milvus。

## 完整流程图

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    IngestionPipeline.process()                          │
│                                                                         │
│  ┌──────────┐   ┌──────────┐   ┌─────────────┐   ┌──────────┐         │
│  │ ① Loader │ → │ ② Cleaner│ → │ ③ Extractor │ → │ ④ Splitter│         │
│  │          │   │          │   │             │   │          │         │
│  │ 按后缀   │   │ 去空白   │   │ 提取文件名  │   │ 按后缀   │         │
│  │ 自动匹配 │   │ 去HTML   │   │ 文件大小    │   │ 自动选择 │         │
│  │ 读原始文 │   │ 去控制字 │   │ 提取标题    │   │ 分块策略 │         │
│  │ 本字符串 │   │ 符       │   │             │   │          │         │
│  └────┬─────┘   └──────────┘   └──────┬──────┘   └────┬─────┘         │
│       │                               │               │               │
│       └───────────────┬───────────────┘               │               │
│                       │                               │               │
│                       ▼                               ▼               │
│             纯文本字符串                      List[DocumentChunk]      │
│                                                                        │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │                        ⑤ Embedding                              │  │
│  │                                                                  │  │
│  │  遍历 chunks:                                                    │  │
│  │    for each chunk:                                               │  │
│  │      chunk.embedding        = await embedder.embed([text])[0]   │  │
│  │      chunk.sparse_embedding = bm25.compute_sparse(text)          │  │
│  │                                                                  │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                              │                                         │
│                              ▼                                         │
│                返回 list[DocumentChunk]                                 │
│                (每个 chunk 已带双向量的)                                │
└─────────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
                    ┌──────────────────┐
                    │   写入存储层      │
                    │                  │
                    │  MilvusStore     │
                    │  .insert_chunks()│
                    │                  │
                    │  同时更新:       │
                    │  PostgreSQL      │
                    │  DocumentModel   │
                    │  status=READY    │
                    └──────────────────┘
```

## 调用链路

### 链路 A：API 同步上传

```
用户 POST /api/v1/documents (multipart/form-data)
  │
  ├─ FastAPI → documents.py:upload_document()
  │   │
  │   ├─ file_store.save(content)            ← LocalFileStore
  │   │     → uuid 重命名，写 data/uploads/
  │   │     → 返回 storage_path
  │   │
  │   ├─ db_repo.save(document)              ← DocumentRepository
  │   │     → session.merge(DocumentModel)
  │   │     → session.commit()
  │   │     → 返回 Document(含 id)
  │   │
  │   ├─ db_repo.update_status(PROCESSING)
  │   │
  │   ├─ pipeline.process(storage_path, doc)  ← ⬅ 核心入口
  │   │   │
  │   │   ├── loader_registry.get_loader(ext)  → TextLoader / PDFLoader / ...
  │   │   ├── loader.load(storage_path)        → 原始字符串
  │   │   ├── cleaner.clean(text)              → 清洗后字符串
  │   │   ├── extractor.extract(path, text)    → dict(文件名, 大小, 标题)
  │   │   ├── _get_splitter(ext)               → CodeSplitter / RecursiveSplitter
  │   │   ├── splitter.split(doc_id, text, meta) → list[DocumentChunk]
  │   │   ├── embedder.embed(texts)            → 稠密向量 → chunk.embedding
  │   │   └── bm25.compute_embeddings(texts)   → 稀疏向量 → chunk.sparse_embedding
  │   │
  │   ├─ vector_store.insert_chunks(chunks)    ← MilvusStore
  │   │     → 拼 entities: [ids, doc_ids, contents, indexes, metadatas, dense_vecs, sparse_vecs]
  │   │     → collection.insert(entities)
  │   │     → collection.flush()
  │   │
  │   └─ db_repo.update_status(READY)
  │
  └─ 返回 {id, filename, status}
```

### 链路 B：Celery 异步处理

```
process_document_task.delay(document_id, file_path, filename)
  │
  ├─ Celery worker 接收
  ├─ 创建独立 EventLoop（asyncio.new_event_loop）
  │
  └─ _process_document(document_id, file_path)
      │
      ├─ 新建 EmbeddingProvider（从 settings 读配置）
      ├─ 新建 Bm25SparseEmbedding（空状态 / 从磁盘加载）
      ├─ pipeline = IngestionPipeline(...)
      │
      ├─ pipeline.process(file_path, doc)   ← 同上 5 阶段
      │
      ├─ sparse_provider.save_state()        ← 写 BM25 词频到磁盘
      ├─ vector_store.insert_chunks(chunks)  ← 写 Milvus
      └─ repo.update_status(READY)           ← 写 PostgreSQL
```

> 异步链路失败后 Celery 自动重试：`max_retries=3, countdown=60`。

## 5 阶段详细实现

### ① Loader — 文件加载

**接口**：`backend/ingestion/loader/base.py`

```python
class Loader(ABC):
    async def load(self, file_path: str | Path) -> str: ...
    def supported_extensions(self) -> set[str]: ...
```

**注册表**：`LoaderRegistry`（`code_loader.py`）

```python
class LoaderRegistry:
    def __init__(self):
        self._loaders = [
            TextLoader(), PDFLoader(), MarkdownLoader(),
            WordLoader(), ExcelLoader(), PPTLoader(), CodeLoader(),
        ]
    def get_loader(self, file_path) -> Loader | None:
        ext = Path(file_path).suffix.lower()
        for loader in self._loaders:
            if ext in loader.supported_extensions():
                return loader
        return None  # ← 匹配不到则 pipeline 抛出 ValueError
```

**Loader 展开表**：

| 类 | 文件 | 支持后缀 | 实现方式 |
|----|------|----------|----------|
| `TextLoader` | `text_loader.py` | `.txt` `.log` `.csv` `.tsv` | `Path.read_text("utf-8")` |
| `PDFLoader` | `pdf_loader.py` | `.pdf` | `PyMuPDF(fitz)` 逐页提取 |
| `MarkdownLoader` | `markdown_loader.py` | `.md` `.markdown` | 读 UTF-8 原文 |
| `WordLoader` | `office_loader.py` | `.docx` | `python-docx` 段落拼接 |
| `ExcelLoader` | `office_loader.py` | `.xlsx` | `openpyxl` 逐单元格拼接 |
| `PPTLoader` | `office_loader.py` | `.pptx` | `python-pptx` 逐幻灯片提取 |
| `CodeLoader` | `code_loader.py` | `.py .js .ts .java .go .rs .c ...`（30+） | `Path.read_text("utf-8")` |

### ② Cleaner — 文本清洗

**文件**：`backend/ingestion/processor/cleaner.py`

```python
class TextCleaner:
    def clean(self, text: str) -> str:
        text = re.sub(r"\r\n", "\n", text)     # 统一换行符
        text = re.sub(r"\x00", "", text)        # 去空字符
        text = re.sub(r"[ \t]+", " ", text)     # 合并空白
        text = re.sub(r"\n{3,}", "\n\n", text)  # 合并多余空行
        return text.strip()

    def clean_html(self, text: str) -> str:
        text = re.sub(r"<[^>]+>", "", text)     # 去 HTML 标签
        return self.clean(text)
```

### ③ Extractor — 元数据提取

**文件**：`backend/ingestion/processor/extractor.py`

```python
class MetadataExtractor:
    def extract(self, file_path: str | Path, content: str) -> dict:
        p = Path(file_path)
        metadata = {
            "filename": p.name,
            "extension": p.suffix.lower(),
            "file_size": p.stat().st_size if p.exists() else 0,
        }
        title = self._extract_title(content)
        if title:
            metadata["title"] = title
        return metadata

    @staticmethod
    def _extract_title(content: str) -> str | None:
        """取前20行，找 # 标题 或 title:，都没有则取第一行非空文本。"""
        lines = content.strip().split("\n")
        for line in lines[:20]:
            line = line.strip()
            if line.startswith("# ") or line.startswith("title:"):
                return line.lstrip("# title:").strip()
            if line and len(line) < 200:   # 避开大段乱码
                return line
        return None
```

### ④ Splitter — 分块

**接口**：`backend/ingestion/splitter/base.py`

```python
class Splitter(ABC):
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 64):
        ...
    def split(self, document_id: str, text: str, metadata=None) -> list[DocumentChunk]: ...
```

**自动选择逻辑**（在 `IngestionPipeline._get_splitter()` 中）：

```python
def _get_splitter(self, extension: str) -> Splitter:
    if extension in CodeLoader.CODE_EXTENSIONS:  # 30+ 代码后缀
        return CodeSplitter(extension, chunk_size, chunk_overlap)
    return RecursiveSplitter(chunk_size, chunk_overlap)  # 其他所有文件
```

**RecursiveSplitter**（`recursive_splitter.py`）：

```
策略：按分隔符优先级逐级切分，最后按 chunk_size 合并。
分隔符顺序：["\n\n", "\n", "。", ". ", " ", ""]
  → 先按段落(\n\n)切，片段超长则按行(\n)切，
    再按句号切，最后按词/字硬切。
  → 相邻片段用 chunk_overlap 重叠合并。
```

**CodeSplitter**（`code_splitter.py`）：

```
策略：按语言的关键字分隔符（正则）切分到函数/类级别，再按 chunk_size 合并。

语言感知的正则分隔符：
  .py  → def | class | @ | async def
  .js  → function | class | const | let | var
  .ts  → function | class | interface | type | const | export
  .java → public | private | protected | class | interface
  .go  → func | type | struct | interface
  .rs  → fn | struct | enum | impl | trait
  未匹配 → "\n\n"（段落）
```

### ⑤ Embedding — 向量化

#### 稠密向量（必选）

**接口**：`backend/ingestion/embedding/base.py`

```python
class EmbeddingProvider(ABC):
    async def embed(self, texts: list[str]) -> list[list[float]]: ...
    async def embed_query(self, text: str) -> list[float]: ...
    @property
    def dimension(self) -> int: ...
```

**两个实现**：

| 实现 | 文件 | 方式 | 默认模型 | 维度 |
|------|------|------|----------|------|
| `OpenAIEmbeddingProvider` | `openai_embedding.py` | `AsyncOpenAI().embeddings.create()` | `text-embedding-3-small` | 1536 |
| `LocalEmbeddingProvider` | `local_embedding.py` | 本地加载 BGE/GTE 等 | configurable | - |

通过 `settings.embedding.provider` 热切换（`openai` / `local`）。

#### 稀疏向量 — BM25（可选）

**文件**：`backend/ingestion/embedding/bm25_sparse.py`

```python
class Bm25SparseEmbedding:
    def compute_sparse(self, text, update_stats=True) -> dict[int, float]: ...
    def compute_embeddings(self, texts, update_stats=True) -> list[dict[int, float]]: ...
    def save_state(self): ...   # 持久化到 JSON
    def _load_state(self): ...  # 从 JSON 恢复
```

**流程**：

```
输入文本
  → _tokenize() 分词
     ├─ 中文部分 → jieba 分词（无 jieba 则字粒度回退）
     └─ 非中文部分 → 按空白/标点 split，转小写
  → Counter 统计词频
  → 计算 BM25 权重：
       idf = log((N - df + 0.5) / (df + 0.5) + 1)
       score = idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * doc_len / avgdl))
  → 返回 {term_id: score}
```

BM25 统计（DF、avg_doc_len）持续累积，`save_state()` 写入 JSON 文件跨会话持久化。

## 输出写入

### MilvusStore.insert_chunks()

**文件**：`backend/storage/vector_store/milvus_store.py`

拼实体写入 Milvus `document_chunks` 集合：

```
chunk_id         VARCHAR(64)   PRIMARY KEY
document_id      VARCHAR(64)
content          VARCHAR(8192)
chunk_index      INT64
metadata_json    VARCHAR(4096)
embedding        FLOAT_VECTOR(1536)     COSINE 度量 → IVF_FLAT 索引
sparse_embedding SPARSE_FLOAT_VECTOR    IP 度量    → SPARSE_INVERTED_INDEX 索引
```

### DocumentRepository

**文件**：`backend/storage/relational_db/document_repo.py`

```python
class DocumentRepository:
    save(document)          # merge + commit
    get(document_id)        # 查单条
    list(skip, limit)       # 分页查
    delete(document_id)     # 物理删除
    update_status(id, status)  # 更新状态 PENDING → PROCESSING → READY / FAILED
    count()                 # 总数
```

**对应模型**：`DocumentModel`（`models.py`）

| 列 | 类型 | 说明 |
|----|------|------|
| id | String(64) PK | uuid hex |
| filename | String(512) | 原始文件名 |
| source | String(1024) | 存储路径 |
| document_type | String(32) | text / pdf / markdown / word / excel / ppt / code / unknown |
| status | String(32) | pending / processing / ready / failed / deleted |
| metadata_json | Text | JSON 字符串 |
| created_at | DateTime | UTC |
| updated_at | DateTime | UTC（onupdate） |

## 设计要点

1. **可替换阶段** — Loader / Cleaner / Extractor / Splitter / EmbeddingProvider 都是抽象接口，可独立替换
2. **双路嵌入** — 同时计算稠密 + 稀疏向量，支撑混合检索（HybridRetriever 7:3 加权 RRF 融合）
3. **双路写入** — chunks 进 Milvus（向量检索），文档状态进 PostgreSQL（元数据管理）
4. **BM25 增量学习** — 每次 ingestion 更新词频统计，`save_state()` 持久化，跨会话持续改进稀疏检索质量
5. **状态追踪** — Document 状态贯穿：PENDING → PROCESSING → READY / FAILED，异常回滚标记
6. **同步 / 异步双通道** — 小文件 API 同步处理，大文件/批量走 Celery 异步队列，失败自动重试
