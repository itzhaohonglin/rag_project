"""PDF 加载 → 分块 → embedding → 写入 Milvus（完整链路示例）"""

import asyncio
from pathlib import Path

from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pymilvus import MilvusClient, DataType
from pymilvus.milvus_client.index import IndexParams

from backend.core.config import settings
from backend.domain.embedding import EmbeddingConfig as DomainEmbeddingConfig
from backend.ingestion.embedding.openai_embedding import OpenAIEmbeddingProvider
from scripts.examples.splitter.adaptive_splitter import create_adaptive_splitter

# 脚本所在目录，PDF 放同目录下即可
DATA_DIR = Path(__file__).parent
pdf_path = str(DATA_DIR.parent / "data" / "pdf" / "ai_llm_engeerning.pdf")

# ── 1. 加载 PDF ──────────────────────────────────────────────
loader = UnstructuredPDFLoader(file_path=pdf_path, mode="paged", strategy="fast")
documents = loader.load()

# ── 2. 自适应分块（根据文档长度动态调整 chunk 大小）───────────────
full_text = " ".join([doc.page_content for doc in documents])
splitter = create_adaptive_splitter(full_text)
chunks = splitter.split_documents(documents)
print(f"PDF 共 {len(documents)} 页，分割为 {len(chunks)} 个文本块")

# ── 3. 提取纯文本 ────────────────────────────────────────────
texts = [c.page_content for c in chunks]


# ── 4. 生成 embedding（异步） ───────────────────────────────
async def generate_embeddings():
    cfg = settings.embedding
    domain_cfg = DomainEmbeddingConfig(
        model_name=cfg.openai.model,
        dimension=cfg.dimensions,
        provider=cfg.provider,
    )
    provider = OpenAIEmbeddingProvider(
        config=domain_cfg,
        base_url=settings.llm.openai.base_url or None,
        api_key=settings.llm.openai.api_key or None,
    )
    return await provider.embed(texts)


embeddings = asyncio.run(generate_embeddings())
print(f"生成 {len(embeddings)} 条 embedding，维度: {len(embeddings[0])}")

# ── 5. 创建 Milvus 集合 ─────────────────────────────────────
client = MilvusClient(uri="http://localhost:19530")
collection_name = "rag_test_collection"

if client.has_collection(collection_name):
    client.drop_collection(collection_name)
    print(f"已删除已有集合：{collection_name}")

schema = client.create_schema(auto_id=True, enable_dynamic_field=False)
schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=1536)
schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=8192)
schema.add_field(field_name="source", datatype=DataType.VARCHAR, max_length=512)

index_params = IndexParams()
index_params.add_index(
    field_name="vector",
    index_type="IVF_FLAT",
    metric_type="COSINE",
    params={"nlist": 1024},
)
client.create_collection(
    collection_name=collection_name,
    schema=schema,
    index_params=index_params,
)
print("Milvus 集合创建完成")

# ── 6. 插入数据（text + embedding 一起写入） ────────────────
data = [
    {"text": text, "vector": emb, "source": pdf_path}
    for text, emb in zip(texts, embeddings)
]
res = client.insert(collection_name=collection_name, data=data)
print(f"成功插入 {len(res['ids'])} 条数据到 Milvus")
