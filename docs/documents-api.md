# 文档管理 API 逻辑说明

## 文件位置

`backend/api/routes/documents.py`

## 接口总览

| 方法 | 路由 | 功能 |
|------|------|------|
| POST | `/api/v1/documents` | 上传文档 |
| GET | `/api/v1/documents` | 文档列表（分页） |
| GET | `/api/v1/documents/{id}` | 文档详情 |
| DELETE | `/api/v1/documents/{id}` | 删除文档 |

---

## 上传文档流程

```
用户 POST 上传文件
        │
        ▼
┌─────────────────────────────┐
│  1. file.read() 读文件内容   │
│  2. file_store.save() 存本地 │
└─────────┬───────────────────┘
          │
          ▼
┌─────────────────────────────┐
│  根据文件后缀判断文档类型    │
│  .pdf → pdf                 │
│  .docx → word               │
│  .txt → text                │
│  .md → markdown             │
│  .py/.js → code             │
│  其他 → UNKNOWN              │
└─────────┬───────────────────┘
          │
          ▼
┌─────────────────────────────┐
│  3. 数据库建记录，状态=待处理 │
└─────────┬───────────────────┘
          │
          ▼
┌─────────────────────────────┐
│  4. 状态改成「PROCESSING」   │
└─────────┬───────────────────┘
          │
          ▼
┌─────────────────────────────┐
│  5. pipeline.process()       │
│     ├── Loader 解析文件      │
│     ├── Cleaner 清洗         │
│     ├── Splitter 分块        │
│     └── Embedding 转向量     │
└─────────┬───────────────────┘
          │
          ▼
┌─────────────────────────────┐
│  6. vector_store             │
│     .insert_chunks()         │
│     向量块塞入 Milvus        │
└─────────┬───────────────────┘
          │
          ▼
┌─────────────────────────────┐
│  7. 状态改成「READY」         │
│     返回 {id, filename,      │
│           status: "ready"}   │
└─────────────────────────────┘

          出错时
          │
          ▼
┌─────────────────────────────┐
│  状态改成「FAILED」          │
│  抛 UPLOAD_FAILED 异常      │
└─────────────────────────────┘
```

### 关键代码段

```python
# 读文件 + 存本地
content = await file.read()
storage_path = await file_store.save(file.filename, content)

# 看后缀猜文档类型
ext = file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
type_map = {"txt": "text", "pdf": "pdf", "md": "markdown", "docx": "word",
            "xlsx": "excel", "pptx": "ppt", "py": "code", "js": "code"}

# 流水线处理
chunks = await pipeline.process(storage_path, document)
await vector_store.insert_chunks(chunks)
```

---

## 文档列表流程

```
GET /api/v1/documents?skip=0&limit=20
        │
        ▼
┌─────────────────────────────┐
│  doc_repo.list(skip, limit) │
│  → 查数据库，分页取文档     │
└─────────┬───────────────────┘
          │
          ▼
┌─────────────────────────────┐
│  返回 { items: [...],       │
│         total: N }          │
└─────────────────────────────┘
```

---

## 删除文档流程

```
DELETE /api/v1/documents/{id}
        │
        ▼
┌─────────────────────────────┐
│  查数据库，文档存在吗？      │
│  不存在 → 抛 404            │
└─────────┬───────────────────┘
          │ 存在
          ▼
┌─────────────────────────────┐
│  vector_store                │
│  .delete_document(id)        │
│  → 从 Milvus 删掉所有向量块  │
└─────────┬───────────────────┘
          │
          ▼
┌─────────────────────────────┐
│  数据库标 DELETED（软删）    │
└─────────────────────────────┘
```

---

## 文档状态流转

```
UPLOADING ──→ PROCESSING ──→ READY
                    │
                    └──→ FAILED

READY ──→ DELETED（删除时）
```

## 依赖的服务

| 依赖 | 用途 |
|------|------|
| PostgreSQL | 存文档元数据（id, 文件名, 类型, 状态, 时间） |
| Milvus | 存文档切块后的向量，供检索用 |
| 本地文件系统 | `data/uploads/` 目录存原始文件 |
| OpenAI / 本地模型 | 把文本块转成向量 |
