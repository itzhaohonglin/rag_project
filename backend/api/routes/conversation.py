from fastapi import APIRouter, Depends

from backend.api.dependencies import get_conversation_repo, get_rag_engine
from backend.api.errors import success_response
from backend.api.schemas.conversation import (
    ConversationListResponse,
    ConversationResponse,
    SendMessageRequest,
    SendMessageResponse,
)
from backend.core.rag_engine import RAGEngine
from backend.domain.conversation import Conversation, Message
from backend.domain.enums import MessageRole
from backend.domain.exceptions import DocumentNotFoundError
from backend.storage.relational_db.conversation_repo import ConversationRepository

router = APIRouter()


@router.post("", response_model=dict)
async def create_conversation(
    conv_repo: ConversationRepository = Depends(get_conversation_repo),
):
    conversation = Conversation()
    conv_repo.save(conversation)
    return success_response(ConversationResponse(
        id=conversation.id, created_at=conversation.created_at,
        updated_at=conversation.updated_at, metadata=conversation.metadata,
    ).model_dump())


@router.get("", response_model=dict)
async def list_conversations(
    skip: int = 0,
    limit: int = 20,
    conv_repo: ConversationRepository = Depends(get_conversation_repo),
):
    convs = conv_repo.list(skip=skip, limit=limit)
    return success_response(ConversationListResponse(
        items=[ConversationResponse(**c.to_dict()) for c in convs],
        total=len(convs),
    ).model_dump())


@router.get("/{conversation_id}", response_model=dict)
async def get_conversation(
    conversation_id: str,
    conv_repo: ConversationRepository = Depends(get_conversation_repo),
):
    conv = conv_repo.get(conversation_id)
    if not conv:
        raise DocumentNotFoundError(conversation_id)
    return success_response(ConversationResponse(**conv.to_dict()).model_dump())


@router.post("/{conversation_id}/messages", response_model=dict)
async def send_message(
    conversation_id: str,
    req: SendMessageRequest,
    conv_repo: ConversationRepository = Depends(get_conversation_repo),
    engine: RAGEngine = Depends(get_rag_engine),
):
    conv = conv_repo.get(conversation_id)
    if not conv:
        raise DocumentNotFoundError(conversation_id)

    user_msg = Message(role=MessageRole.USER, content=req.content)
    conv.add_message(user_msg)

    answer, result, citations = await engine.answer(req.content, conversation=conv)

    assistant_msg = Message(
        role=MessageRole.ASSISTANT, content=answer, citations=citations,
    )
    conv.add_message(assistant_msg)
    conv_repo.save(conv)

    return success_response(SendMessageResponse(
        reply=answer,
        citations=[{"chunk_id": c.chunk_id, "document_id": c.document_id,
                     "content": c.content, "score": c.score} for c in citations],
    ).model_dump())
