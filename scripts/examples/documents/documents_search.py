"""从 Milvus 检索相似文本块（向量搜索示例）"""

import asyncio

from pymilvus import MilvusClient

from backend.core.config import settings
from backend.domain.embedding import EmbeddingConfig as DomainEmbeddingConfig
from backend.ingestion.embedding.openai_embedding import OpenAIEmbeddingProvider

COLLECTION_NAME = "rag_test_collection"


async def search(query: str, top_k: int = 5):
    # ── 1. 生成查询向量 ────────────────────────────────────
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
    query_vec = await provider.embed_query(query)
    print(f"查询向量已生成，维度: {len(query_vec)}")

    # ── 2. 连接 Milvus ─────────────────────────────────────
    client = MilvusClient(uri="http://localhost:19530")

    if not client.has_collection(COLLECTION_NAME):
        print(f"集合 {COLLECTION_NAME} 不存在，请先运行 pdf_loader.py")
        return

    # ── 3. 向量检索 ────────────────────────────────────────
    results = client.search(
        collection_name=COLLECTION_NAME,
        data=[query_vec],
        limit=top_k,
        output_fields=["text", "source"],
        search_params={"metric_type": "COSINE", "params": {"nprobe": 10}},
    )

    # ── 4. 打印结果 ────────────────────────────────────────
    for i, hit in enumerate(results[0], 1):
        content = hit['entity']['text'][:200]
        source = hit['entity']['source']
        print(f"#{i}  相似度: {hit['distance']:.4f}")
        print(f"来源: {source}")

        # Windows 终端 GBK 兼容
        try:
            print(f"内容: {content}")
        except UnicodeEncodeError:
            print(f"内容: {content.encode('gbk', errors='replace').decode('gbk')}")

        print("-" * 60)


if __name__ == "__main__":
    q = ("他擅长的java技能有哪些")
    print(f"查询: {q}\n")
    asyncio.run(search(q))
