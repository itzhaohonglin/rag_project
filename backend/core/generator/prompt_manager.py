from typing import Any

SYSTEM_PROMPT = """你是一个智能的 RAG 问答助手。请基于提供的上下文信息回答用户问题。

## 规则
1. 只使用提供的上下文信息回答问题，不要添加自己的知识
2. 如果上下文信息不足以回答问题，请明确说明
3. 请标注引用来源 [来源:n]，其中 n 是上下文的序号
4. 使用中文回答
5. 回答应简洁且结构化"""


class PromptManager:
    def build_rag_prompt(self, query: str, contexts: list[str]) -> list[dict]:
        context_text = "\n\n".join(
            f"[来源:{i + 1}] {ctx}" for i, ctx in enumerate(contexts)
        )
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"## 上下文信息\n\n{context_text}\n\n## 问题\n\n{query}",
            },
        ]

    def build_direct_prompt(self, query: str) -> list[dict]:
        return [
            {"role": "system", "content": "你是一个有用的助手。请用中文回答。"},
            {"role": "user", "content": query},
        ]
