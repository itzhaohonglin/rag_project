from backend.domain.document import Document, DocumentChunk
from backend.domain.enums import DocumentStatus, DocumentType
from backend.domain.retrieval import Query, RetrievalResult, RetrievedChunk


class TestDocument:
    def test_create_document(self):
        doc = Document(filename="test.txt", source="/path/to/test.txt",
                        document_type=DocumentType.TEXT)
        assert doc.filename == "test.txt"
        assert doc.status == DocumentStatus.PENDING
        assert doc.id is not None

    def test_document_to_dict(self):
        doc = Document(filename="test.txt", source="/path/to/test.txt")
        data = doc.to_dict()
        assert data["filename"] == "test.txt"
        assert data["status"] == "pending"

    def test_document_from_dict(self):
        original = Document(filename="test.txt", source="/path/to/test.txt")
        data = original.to_dict()
        restored = Document.from_dict(data)
        assert restored.id == original.id
        assert restored.filename == original.filename


class TestDocumentChunk:
    def test_create_chunk(self):
        chunk = DocumentChunk(
            document_id="doc1", content="hello world", chunk_index=0
        )
        assert chunk.document_id == "doc1"
        assert chunk.content == "hello world"
        assert chunk.id is not None


class TestRetrieval:
    def test_query_defaults(self):
        q = Query(text="test query")
        assert q.text == "test query"
        assert q.top_k == 10

    def test_retrieval_result(self):
        chunk = RetrievedChunk(
            chunk_id="c1", document_id="d1",
            content="test", score=0.95,
        )
        result = RetrievalResult(query=Query(text="q"), chunks=[chunk])
        assert result.total_chunks == 1
        data = result.to_dict()
        assert data["total_chunks"] == 1
