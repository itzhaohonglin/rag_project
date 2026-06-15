import json
import re

from backend.core.generator.llm_client import LLMClient
from backend.domain.retrieval import RetrievedChunk

RELEVANCE_SYSTEM_PROMPT = """你是一个文档相关性评估助手。你的任务是判断文档片段是否有助于回答用户的问题。

规则：
1. 如果文档片段包含与问题直接相关的信息，则为「相关」(1)
2. 如果文档片段包含部分相关信息或背景知识，则为「部分相关」(0.5)
3. 如果文档片段完全不相关，则为「不相关」(0)
4. 只依赖文档内容，不要靠自己的知识判断"""


class RelevanceEvaluator:
    """使用 LLM 评估每个检索 chunk 的相关性。"""

    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client

    async def evaluate(
        self, query: str, chunks: list[RetrievedChunk]
    ) -> list[float]:
        """批量评估，每个 chunk 返回 0~1 之间的分数。"""
        if not chunks:
            return []

        user_prompt = self._build_prompt(query, chunks)

        messages = [
            {"role": "system", "content": RELEVANCE_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        try:
            resp = await self.llm_client.generate(messages)
            return self._parse_response(resp, len(chunks))
        except Exception:
            return [1.0] * len(chunks)

    def _build_prompt(self, query: str, chunks: list[RetrievedChunk]) -> str:
        lines = [f"## 问题\n\n{query}\n\n## 文档片段"]
        for i, c in enumerate(chunks):
            snippet = c.content[:500]
            lines.append(f"\n--- 片段 {i + 1} ---\n{snippet}")
        lines.append(
            "\n\n返回 JSON 数组，每个元素是 0、0.5 或 1，例如：[1, 0, 0.5, 1]"
        )
        return "\n".join(lines)

    def _parse_response(self, text: str, expected: int) -> list[float]:
        # 尝试从 JSON 代码块提取
        m = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", text, re.DOTALL)
        if m:
            text = m.group(1)
        # 尝试直接解析 JSON 数组
        m = re.search(r"\[[\d.,\s]+\]", text)
        if m:
            try:
                scores = json.loads(m.group())
                if isinstance(scores, list) and len(scores) == expected:
                    return [float(s) for s in scores]
            except (json.JSONDecodeError, ValueError):
                pass
        # fallback: 全当相关
        return [1.0] * expected
