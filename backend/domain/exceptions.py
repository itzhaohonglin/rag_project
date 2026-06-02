class RAGException(Exception):
    """Base exception for all RAG project errors."""
    def __init__(self, message: str, code: str, detail: dict | None = None):
        self.message = message
        self.code = code
        self.detail = detail or {}
        super().__init__(self.message)


class DocumentNotFoundError(RAGException):
    def __init__(self, document_id: str):
        super().__init__(
            message=f"Document not found: {document_id}",
            code="DOCUMENT_NOT_FOUND",
            detail={"document_id": document_id},
        )


class DocumentProcessingError(RAGException):
    def __init__(self, document_id: str, reason: str):
        super().__init__(
            message=f"Document processing failed: {reason}",
            code="DOCUMENT_PROCESSING_ERROR",
            detail={"document_id": document_id, "reason": reason},
        )


class EmbeddingError(RAGException):
    def __init__(self, reason: str):
        super().__init__(
            message=f"Embedding generation failed: {reason}",
            code="EMBEDDING_ERROR",
            detail={"reason": reason},
        )


class LLMError(RAGException):
    def __init__(self, reason: str):
        super().__init__(
            message=f"LLM call failed: {reason}",
            code="LLM_ERROR",
            detail={"reason": reason},
        )


class ConfigurationError(RAGException):
    def __init__(self, key: str, reason: str):
        super().__init__(
            message=f"Configuration error for {key}: {reason}",
            code="CONFIGURATION_ERROR",
            detail={"key": key, "reason": reason},
        )
