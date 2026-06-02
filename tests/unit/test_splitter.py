from backend.ingestion.splitter.recursive_splitter import RecursiveSplitter
from backend.ingestion.splitter.code_splitter import CodeSplitter


class TestRecursiveSplitter:
    def test_split_small_text(self):
        splitter = RecursiveSplitter(chunk_size=1000, chunk_overlap=0)
        chunks = splitter.split("doc1", "short text")
        assert len(chunks) == 1
        assert chunks[0].content == "short text"

    def test_split_large_text(self, sample_text):
        splitter = RecursiveSplitter(chunk_size=20, chunk_overlap=5)
        chunks = splitter.split("doc1", sample_text)
        assert len(chunks) >= 2
        assert all(c.document_id == "doc1" for c in chunks)
        assert all(c.chunk_index == i for i, c in enumerate(chunks))


class TestCodeSplitter:
    def test_split_python_code(self, sample_code):
        splitter = CodeSplitter(extension=".py", chunk_size=200, chunk_overlap=20)
        chunks = splitter.split("doc1", sample_code)
        assert len(chunks) >= 1
        for c in chunks:
            assert c.metadata.get("extension") == ".py"
