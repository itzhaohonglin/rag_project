from fastapi import APIRouter, Depends, File, UploadFile
from sqlalchemy.orm import Session

from backend.api.errors import APIError, success_response
from backend.api.schemas.document import DocumentListResponse, DocumentResponse, DocumentUploadResponse
from backend.core.config import settings
from backend.domain.document import Document
from backend.domain.enums import DocumentStatus, DocumentType
from backend.domain.exceptions import DocumentNotFoundError
from backend.ingestion.embedding.openai_embedding import OpenAIEmbeddingProvider
from backend.ingestion.processor.pipeline import IngestionPipeline
from backend.storage.file_store.local_fs import LocalFileStore
from backend.storage.relational_db.base import get_session
from backend.storage.relational_db.document_repo import DocumentRepository
from backend.storage.vector_store.milvus_store import MilvusStore

router = APIRouter()


@router.post("", response_model=dict)
async def upload_document(
    file: UploadFile = File(...),
    doc_repo: DocumentRepository = Depends(),
    file_store: LocalFileStore = Depends(),
    vector_store: MilvusStore = Depends(),
    pipeline: IngestionPipeline = Depends(),
):
    content = await file.read()
    storage_path = await file_store.save(file.filename, content)

    ext = file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
    doc_type = DocumentType.UNKNOWN
    type_map = {"txt": "text", "pdf": "pdf", "md": "markdown", "docx": "word",
                "xlsx": "excel", "pptx": "ppt", "py": "code", "js": "code"}
    if ext in type_map:
        doc_type = DocumentType(type_map[ext])

    document = Document(filename=file.filename, source=storage_path, document_type=doc_type)
    document = doc_repo.save(document)

    try:
        document.status = DocumentStatus.PROCESSING
        doc_repo.update_status(document.id, DocumentStatus.PROCESSING)

        chunks = await pipeline.process(storage_path, document)
        await vector_store.insert_chunks(chunks)

        doc_repo.update_status(document.id, DocumentStatus.READY)
    except Exception as e:
        doc_repo.update_status(document.id, DocumentStatus.FAILED)
        raise APIError(code="UPLOAD_FAILED", message=str(e), status_code=500)

    return success_response(DocumentUploadResponse(
        id=document.id, filename=document.filename, status=document.status.value,
    ).model_dump())


@router.get("", response_model=dict)
async def list_documents(
    skip: int = 0,
    limit: int = 20,
    doc_repo: DocumentRepository = Depends(),
):
    docs = doc_repo.list(skip=skip, limit=limit)
    total = doc_repo.count()
    return success_response(DocumentListResponse(
        items=[DocumentResponse(**d.to_dict()) for d in docs], total=total,
    ).model_dump())


@router.get("/{document_id}", response_model=dict)
async def get_document(
    document_id: str,
    doc_repo: DocumentRepository = Depends(),
):
    doc = doc_repo.get(document_id)
    if not doc:
        raise DocumentNotFoundError(document_id)
    return success_response(DocumentResponse(**doc.to_dict()).model_dump())


@router.delete("/{document_id}", response_model=dict)
async def delete_document(
    document_id: str,
    doc_repo: DocumentRepository = Depends(),
    vector_store: MilvusStore = Depends(),
):
    doc = doc_repo.get(document_id)
    if not doc:
        raise DocumentNotFoundError(document_id)
    await vector_store.delete_document(document_id)
    doc_repo.update_status(document_id, DocumentStatus.DELETED)
    return success_response(message="Document deleted")
