import json
from datetime import datetime

from sqlalchemy.orm import Session

from backend.domain.conversation import Citation, Conversation, Message
from backend.domain.enums import MessageRole
from backend.storage.relational_db.models import ConversationModel, MessageModel


class ConversationRepository:
    def __init__(self, session: Session):
        self.session = session

    def save(self, conversation: Conversation) -> Conversation:
        model = ConversationModel(
            id=conversation.id,
            metadata_json=json.dumps(conversation.metadata, ensure_ascii=False),
            created_at=conversation.created_at,
            updated_at=conversation.updated_at,
        )
        self.session.merge(model)
        for msg in conversation.messages:
            msg_model = MessageModel(
                conversation_id=conversation.id,
                role=msg.role.value,
                content=msg.content,
                citations_json=json.dumps(
                    [{"chunk_id": c.chunk_id, "document_id": c.document_id,
                      "content": c.content, "score": c.score} for c in msg.citations],
                    ensure_ascii=False,
                ),
                timestamp=msg.timestamp,
            )
            self.session.add(msg_model)
        self.session.commit()
        return conversation

    def get(self, conversation_id: str) -> Conversation | None:
        model = self.session.query(ConversationModel).filter_by(id=conversation_id).first()
        if not model:
            return None
        return self._to_domain(model)

    def list(self, skip: int = 0, limit: int = 20) -> list[Conversation]:
        models = (
            self.session.query(ConversationModel)
            .order_by(ConversationModel.updated_at.desc())
            .offset(skip)
            .limit(limit)
            .all()
        )
        return [self._to_domain(m) for m in models]

    def delete(self, conversation_id: str) -> bool:
        model = self.session.query(ConversationModel).filter_by(id=conversation_id).first()
        if not model:
            return False
        self.session.delete(model)
        self.session.commit()
        return True

    def _to_domain(self, model: ConversationModel) -> Conversation:
        messages = []
        for msg_model in model.messages:
            citations_data = json.loads(msg_model.citations_json or "[]")
            citations = [Citation(**c) for c in citations_data]
            messages.append(Message(
                role=MessageRole(msg_model.role),
                content=msg_model.content,
                citations=citations,
                timestamp=msg_model.timestamp,
            ))
        return Conversation(
            id=model.id,
            messages=messages,
            created_at=model.created_at,
            updated_at=model.updated_at,
            metadata=json.loads(model.metadata_json or "{}"),
        )
