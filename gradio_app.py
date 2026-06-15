"""Gradio 测试页面 — 测文档上传 + 检索"""

import json
import os
from pathlib import Path

import gradio as gr
import httpx

API_BASE = os.getenv("API_BASE", "http://localhost:8001/api/v1")


# ── API 调用 ──────────────────────────────────────────────

async def upload_file(file):
    if file is None:
        return "请选文件"
    async with httpx.AsyncClient(timeout=120) as client:
        with open(file.name, "rb") as f:
            resp = await client.post(f"{API_BASE}/documents", files={"file": f})
        if resp.status_code != 200:
            return f"上传失败: {resp.text}"
        data = resp.json()
        return f"✅ 上传成功\nID: {data['data']['id']}\n状态: {data['data']['status']}"


async def list_docs():
    async with httpx.AsyncClient() as client:
        resp = await client.get(f"{API_BASE}/documents")
        if resp.status_code != 200:
            return "查文档失败", ""
        data = resp.json().get("data", {})
        items = data.get("items", [])
        total = data.get("total", 0)
        if not items:
            return f"共 {total} 个文档", "（空）"
        rows = []
        for d in items:
            status = d["status"]
            emoji = {"ready": "✅", "processing": "⏳", "failed": "❌", "pending": "⏸️"}.get(status, "❓")
            rows.append(f"{emoji} {d['id'][:8]}  {d['filename']:30s}  {status}")
        return f"共 {total} 个文档", "\n".join(rows)


async def do_query(query, mode, top_k):
    if not query.strip():
        return "请输入查询", ""
    async with httpx.AsyncClient(timeout=120) as client:
        resp = await client.post(
            f"{API_BASE}/retrieval/query",
            json={"query": query, "mode": mode, "top_k": top_k},
        )
        if resp.status_code != 200:
            return f"查询失败: {resp.text}", ""
        data = resp.json().get("data", {})
        answer = data.get("answer", "")
        chunks = data.get("chunks", [])
        total_time = data.get("total_time_ms", 0)

        # 组装答案
        answer_text = f"## 回答\n{answer}\n\n⏱️ {total_time:.0f}ms\n"

        # 组装 chunks
        if chunks:
            lines = []
            for i, c in enumerate(chunks):
                score = c.get("score", 0)
                content = c.get("content", "")[:300]
                cid = c.get("chunk_id", "")[:8]
                lines.append(f"### Chunk {i+1}  (score={score:.4f}, id={cid})")
                lines.append(f"```\n{content}\n```\n")
            chunks_text = "\n".join(lines)
        else:
            chunks_text = "（无检索结果）"

        return answer_text, chunks_text


# ── Gradio 页面 ────────────────────────────────────────────

with gr.Blocks(title="RAG 测试台") as demo:
    gr.Markdown("# 🧪 RAG 测试台 — 文档上传 & 检索")

    with gr.Tab("📄 上传文档"):
        with gr.Row():
            file_input = gr.File(label="选文件", file_types=[".txt", ".pdf", ".md", ".py", ".docx"])
            upload_btn = gr.Button("上传", variant="primary")
        upload_out = gr.Textbox(label="结果", lines=3)

        upload_btn.click(upload_file, inputs=[file_input], outputs=[upload_out])

    with gr.Tab("📋 文档列表"):
        refresh_btn = gr.Button("刷新", variant="primary")
        doc_count = gr.Textbox(label="统计", lines=1)
        doc_list = gr.Textbox(label="文档列表", lines=15)

        refresh_btn.click(list_docs, outputs=[doc_count, doc_list])

    with gr.Tab("🔍 检索"):
        with gr.Row():
            with gr.Column(scale=3):
                query_input = gr.Textbox(
                    label="查询",
                    placeholder="输入你的问题...",
                    lines=3,
                )
            with gr.Column(scale=1):
                mode_radio = gr.Radio(
                    choices=["dense", "sparse", "hybrid"],
                    value="hybrid",
                    label="检索模式",
                )
                top_k_slider = gr.Slider(1, 20, value=5, step=1, label="Top-K")
        query_btn = gr.Button("查询", variant="primary", size="lg")

        with gr.Row():
            with gr.Column():
                answer_out = gr.Markdown(label="回答")
            with gr.Column():
                chunks_out = gr.Markdown(label="检索块")

        query_btn.click(
            do_query,
            inputs=[query_input, mode_radio, top_k_slider],
            outputs=[answer_out, chunks_out],
        )

    gr.Markdown("---\n> 确保后端已启动：`uvicorn backend.api.main:app --reload`")


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, theme=gr.themes.Soft())
