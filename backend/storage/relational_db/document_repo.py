from datetime import datetime

from sqlalchemy.orm import Session

from backend.domain.document import Document
from backend.domain.enums import DocumentStatus
from backend.storage.relational_db.models import DocumentModel


class DocumentRepository:
    def __init__(self, session: Session):
        self.session = session

    def save(self, document: Document) -> Document:
        model = DocumentModel(
            id=document.id,
            filename=document.filename,
            source=document.source,
            document_type=document.document_type.value,
            status=document.status.value,
            metadata_json=str(document.metadata),
            created_at=document.created_at,
            updated_at=document.updated_at,
        )
        self.session.merge(model)
        self.session.commit()
        return document

    def get(self, document_id: str) -> Document | None:
        model = self.session.query(DocumentModel).filter_by(id=document_id).first()
        if not model:
            return None
        return self._to_domain(model)

    def list(self, skip: int = 0, limit: int = 20) -> list[Document]:
        models = (
            self.session.query(DocumentModel)
            .order_by(DocumentModel.created_at.desc())
            .offset(skip)
            .limit(limit)
            .all()
        )
        return [self._to_domain(m) for m in models]

    def delete(self, document_id: str) -> bool:
        model = self.session.query(DocumentModel).filter_by(id=document_id).first()
        if not model:
            return False
        self.session.delete(model)
        self.session.commit()
        return True

    def update_status(self, document_id: str, status: DocumentStatus) -> Document | None:
        model = self.session.query(DocumentModel).filter_by(id=document_id).first()
        if not model:
            return None
        model.status = status.value
        model.updated_at = datetime.utcnow()
        self.session.commit()
        return self._to_domain(model)

    def count(self) -> int:
        return self.session.query(DocumentModel).count()

    @staticmethod
    def _to_domain(model: DocumentModel) -> Document:
        from backend.domain.enums import DocumentType
        return Document(
            document_id=model.id,
            filename=model.filename,
            source=model.source,
            document_type=DocumentType(model.document_type),
            status=DocumentStatus(model.status),
            metadata=model.metadata_json or {},
            created_at=model.created_at,
            updated_at=model.updated_at,
        )
