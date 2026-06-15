from fastapi import APIRouter, Depends, File, UploadFile

from backend.api.dependencies import get_document_repo, get_file_store, get_ingestion_pipeline, get_vector_store
from backend.api.errors import APIError, success_response
from backend.api.schemas.document import DocumentListResponse, DocumentResponse, DocumentUploadResponse
from backend.core.config import settings
from backend.domain.document import Document
from backend.domain.enums import DocumentStatus, DocumentType
from backend.domain.exceptions import DocumentNotFoundError
from backend.ingestion.processor.pipeline import IngestionPipeline
from backend.storage.file_store.local_fs import LocalFileStore
from backend.storage.relational_db.document_repo import DocumentRepository
from backend.storage.vector_store.milvus_store import MilvusStore

router = APIRouter()


@router.post("", response_model=dict)
async def upload_document(
    file: UploadFile = File(...),   # ← FastAPI 自动收上传的文件
    db_repo: DocumentRepository = Depends(get_document_repo), ## 数据库操作
    file_store: LocalFileStore = Depends(get_file_store), # 文件存本地
    vector_store: MilvusStore = Depends(get_vector_store), # 向量库
    pipeline: IngestionPipeline = Depends(get_ingestion_pipeline), # 解析流水线
):
    # 读文件 + 存本地
    content = await file.read()
    storage_path = await file_store.save(file.filename, content)

    # 看后缀猜文档类型：.pdf→pdf，.docx→word，.py→code……
    ext = file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
    doc_type = DocumentType.UNKNOWN
    type_map = {"txt": "text", "pdf": "pdf", "md": "markdown", "docx": "word",
                "xlsx": "excel", "pptx": "ppt", "py": "code", "js": "code"}
    if ext in type_map:
        doc_type = DocumentType(type_map[ext])

    # 数据库里建一条记录
    document = Document(filename=file.filename, source=storage_path, document_type=doc_type)
    document = db_repo.save(document)

    # 流水线处理：解析→分块→向量化→塞 Milvus
    try:
        db_repo.update_status(document.id, DocumentStatus.PROCESSING)
        chunks = await pipeline.process(storage_path, document)
        await vector_store.insert_chunks(chunks)
        db_repo.update_status(document.id, DocumentStatus.READY)
    except Exception as e:
        db_repo.update_status(document.id, DocumentStatus.FAILED)
        raise APIError(code="UPLOAD_FAILED", message=str(e), status_code=500)

    return success_response(DocumentUploadResponse(
        id=document.id, filename=document.filename, status=document.status.value,
    ).model_dump())


@router.get("", response_model=dict)
async def list_documents(
    skip: int = 0,
    limit: int = 20,
    doc_repo: DocumentRepository = Depends(get_document_repo),
):
    """分页查文档列表"""
    docs = doc_repo.list(skip=skip, limit=limit)
    total = doc_repo.count()
    return success_response(DocumentListResponse(
        items=[DocumentResponse(**d.to_dict()) for d in docs], total=total,
    ).model_dump())


@router.get("/{document_id}", response_model=dict)
async def get_document(
    document_id: str,
    doc_repo: DocumentRepository = Depends(get_document_repo),
):
    """查单个文档详情"""
    doc = doc_repo.get(document_id)
    if not doc:
        raise DocumentNotFoundError(document_id)
    return success_response(DocumentResponse(**doc.to_dict()).model_dump())


@router.delete("/{document_id}", response_model=dict)
async def delete_document(
    document_id: str,
    doc_repo: DocumentRepository = Depends(get_document_repo),
    vector_store: MilvusStore = Depends(get_vector_store),
):
    """删文档：清向量 + 标删除状态"""
    doc = doc_repo.get(document_id)
    if not doc:
        raise DocumentNotFoundError(document_id)
    await vector_store.delete_document(document_id)
    doc_repo.update_status(document_id, DocumentStatus.DELETED)
    return success_response(message="Document deleted")
