"""
在RAG系统中，选择合适的分割器是决定最终效果的关键一步，好的分块策略能让系统效果提升70%。总的来说，选择哪种分割器，主要取决于你对“准”和“快”的权衡
"""
from langchain_text_splitters import RecursiveCharacterTextSplitter

"""
常见1(推荐)：RecursiveCharacterTextSplitter
尝试按段落→句子→词...的顺序递归切割
平衡性好，能最大限度保留语义完整性
绝大多数通用RAG场景的首选


关键参数说明：
chunk_size：单个块的最大长度（字符数/token数，取决于你用的计算方式）。
chunk_overlap：相邻块重叠长度，能保持边界上下文。它决定了相邻两个文本块之间共享的字符（或 token）数量，直接关系到 RAG 检索时的信息召回率和上下文连贯性
separators：递归切割的优先级列表，越靠前的分隔符优先级越高。
"""


"""
动态参数配置（推荐，更智能）:根据文档实际字符数自动计算 chunk_size 和 chunk_overlap
"""
def create_adaptive_splitter(document_text: str):
    """
    根据文档长度动态创建 RecursiveCharacterTextSplitter
    """
    doc_len = len(document_text)

    if doc_len < 1000:  # 简历等短文档
        # 整个文档作为一块，不切割（若文档非常短）
        # 但为了统一接口，仍设置较大的 chunk_size 使其不超过1块
        chunk_size = max(doc_len + 100, 2000)  # 确保不切割
        chunk_overlap = 0
    elif doc_len < 5000:  # 中等长度（如短论文、多页简历）
        chunk_size = 1000
        chunk_overlap = 10  # 10%
    else:  # 长论文
        chunk_size = 2500
        chunk_overlap = 300  # 12.5%

    # 公共分隔符（同上）
    separators = [
        "\n\n", "\n",
        "。", "！", "？",
        ". ", "! ", "? ",
        "；", ";",
        "，", ",",
        " ", ""
    ]

    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
        separators=separators
    )