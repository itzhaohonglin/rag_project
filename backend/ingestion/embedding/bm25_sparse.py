import json
import math
import re
from pathlib import Path
from collections import Counter


class Bm25SparseEmbedding:
    """BM25 稀疏向量生成器。

    把文本转成 {term_id: bm25_weight} 的稀疏向量，
    给 Milvus SPARSE_FLOAT_VECTOR 用。
    """

    def __init__(self, state_path: str | Path | None = None):
        self.state_path = Path(state_path) if state_path else None

        # 词汇表：term -> term_id
        self.vocab: dict[str, int] = {}
        self.reverse_vocab: dict[int, str] = {}

        # BM25 统计
        self.df: dict[str, int] = {}       # term -> 包含该 term 的文档数
        self.total_docs: int = 0            # 总文档数
        self.avg_doc_len: float = 0.0       # 平均文档长度（词数）
        self._total_doc_len: int = 0        # 所有文档长度总和（用于计算 avg）

        self.k1: float = 1.5
        self.b: float = 0.75

        # 尝试导入 jieba（中文分词）
        self._jieba = None
        try:
            import jieba
            self._jieba = jieba
        except ImportError:
            pass

        self._load_state()

    def _tokenize(self, text: str) -> list[str]:
        """分词：英文按空白/标点 split，中文用 jieba 或字粒度回退。"""
        tokens: list[str] = []

        # 中文字符部分（用 jieba 或字 bi-gram）
        chinese_blocks = re.findall(r"[一-鿿　-〿＀-￯]+", text)
        for block in chinese_blocks:
            if self._jieba:
                tokens.extend(self._jieba.cut(block))
            else:
                # 回退：每个字当 token，去重相邻重复
                for char in block:
                    if char.strip():
                        tokens.append(char)

        # 非中文部分（按空白/标点 split）
        non_chinese = re.sub(r"[一-鿿　-〿＀-￯]", " ", text)
        for token in re.findall(r"[a-zA-Z0-9_\-]+", non_chinese):
            token = token.lower().strip("-_")
            if token:
                tokens.append(token)

        return tokens

    def _term_id(self, term: str) -> int:
        """获取 term 的 ID，不存在则分配新 ID。"""
        if term not in self.vocab:
            tid = len(self.vocab) + 1
            self.vocab[term] = tid
            self.reverse_vocab[tid] = term
            self.df[term] = 0
        return self.vocab[term]

    def compute_sparse(self, text: str, update_stats: bool = True) -> dict[int, float]:
        """计算一段文本的 BM25 稀疏向量。

        返回 {term_id: weight} 字典。
        如果 update_stats=True，会更新 DF / avg_doc_len 统计。
        """
        tokens = self._tokenize(text)
        if not tokens:
            return {}

        doc_len = len(tokens)
        term_counts = Counter(tokens)

        # 更新统计
        if update_stats:
            self.total_docs += 1
            self._total_doc_len += doc_len
            self.avg_doc_len = self._total_doc_len / self.total_docs
            for term in term_counts:
                self.df[term] = self.df.get(term, 0) + 1
                self._term_id(term)

        # BM25 top 权重
        sparse: dict[int, float] = {}
        for term, tf in term_counts.items():
            tid = self._term_id(term)
            df = self.df.get(term, 1)
            idf = math.log((self.total_docs - df + 0.5) / (df + 0.5) + 1.0)
            score = idf * (tf * (self.k1 + 1)) / (tf + self.k1 * (1 - self.b + self.b * doc_len / max(self.avg_doc_len, 1)))

            if score > 0.0:
                sparse[tid] = round(score, 6)

        return sparse

    def compute_embeddings(self, texts: list[str], update_stats: bool = True) -> list[dict[int, float]]:
        """批量计算稀疏向量。"""
        return [self.compute_sparse(t, update_stats=update_stats) for t in texts]

    @property
    def dimension(self) -> int:
        return len(self.vocab)

    def save_state(self):
        """保存词汇表和统计到 JSON。"""
        if not self.state_path:
            return
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "vocab": self.vocab,
            "df": self.df,
            "total_docs": self.total_docs,
            "avg_doc_len": self.avg_doc_len,
            "total_doc_len": self._total_doc_len,
        }
        self.state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")

    def _load_state(self):
        """从 JSON 加载状态。"""
        if not self.state_path or not self.state_path.exists():
            return
        try:
            state = json.loads(self.state_path.read_text(encoding="utf-8"))
            self.vocab = state.get("vocab", {})
            self.reverse_vocab = {v: k for k, v in self.vocab.items()}
            self.df = state.get("df", {})
            self.total_docs = state.get("total_docs", 0)
            self.avg_doc_len = state.get("avg_doc_len", 0.0)
            self._total_doc_len = state.get("total_doc_len", 0)
        except (json.JSONDecodeError, KeyError):
            pass
