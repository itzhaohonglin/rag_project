from datetime import datetime
from pydantic import BaseModel


class MessageResponse(BaseModel):
    role: str
    content: str
    citations: list[dict] = []
    timestamp: datetime


class ConversationResponse(BaseModel):
    id: str
    messages: list[MessageResponse] = []
    created_at: datetime
    updated_at: datetime
    metadata: dict = {}


class ConversationListResponse(BaseModel):
    items: list[ConversationResponse]
    total: int


class SendMessageRequest(BaseModel):
    content: str
    stream: bool = False


class SendMessageResponse(BaseModel):
    reply: str
    citations: list[dict] = []
