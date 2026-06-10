"""使用PyPDFLoader加载PDF。"""

from pathlib import Path

from langchain_community.document_loaders import UnstructuredPDFLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 脚本所在目录，PDF 放同目录下即可
DATA_DIR = Path(__file__).parent

pdf_path = str(DATA_DIR.parent / "data" / "pdf" / "ai_llm_engeerning.pdf")


"""
方式一：使用PyPDFLoader 
核心特点与适用场景： 快速提取PDF文本，简单易用，适合普通扫描件或基于pypdf的转换

"""
# loader = PyPDFLoader(file_path=pdf_path)
# documents = loader.load()
# print(f"加载了 {len(documents)} 页")
"""

方式二：UnstructuredPDFLoader
能精准提取PDF中的表格和复杂布局；具备智能文档解析能力；性能高，功能全面
"""
loader = UnstructuredPDFLoader(file_path=pdf_path,
                               mode="paged",
                               strategy="fast"
                               )
documents = loader.load()

print(f"加载了 {len(documents)} 页")


print(f"加载[0]的内容:{documents[0].page_content}")
print(f"加载[0]的源数据:{documents[0].metadata}")

