from backend.domain.conversation import Conversation, Message
from backend.domain.enums import MessageRole


class ConversationMemory:
    """Sliding window conversation memory."""

    MAX_HISTORY = 20

    def __init__(self, max_history: int | None = None):
        self.max_history = max_history or self.MAX_HISTORY

    def trim(self, conversation: Conversation) -> Conversation:
        if len(conversation.messages) > self.max_history:
            overflow = len(conversation.messages) - self.max_history
            overflow = overflow if overflow % 2 == 0 else overflow + 1  # keep pairs
            conversation.messages = conversation.messages[overflow:]
        return conversation

    def to_llm_messages(self, conversation: Conversation) -> list[dict]:
        return [
            {"role": m.role.value, "content": m.content}
            for m in conversation.messages
            if m.role != MessageRole.SYSTEM
        ]
