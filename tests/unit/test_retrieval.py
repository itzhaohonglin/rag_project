from backend.domain.retrieval import Query, RetrievalResult, RetrievedChunk
from backend.core.memory.conversation_memory import ConversationMemory
from backend.domain.conversation import Conversation, Message
from backend.domain.enums import MessageRole


class TestConversationMemory:
    def test_trim_overflow(self):
        mem = ConversationMemory(max_history=4)
        conv = Conversation()
        for i in range(6):
            conv.add_message(Message(role=MessageRole.USER, content=f"msg{i}"))
            conv.add_message(Message(role=MessageRole.ASSISTANT, content=f"resp{i}"))
        assert len(conv.messages) == 12
        mem.trim(conv)
        assert len(conv.messages) <= 4

    def test_to_llm_messages(self):
        mem = ConversationMemory()
        conv = Conversation()
        conv.add_message(Message(role=MessageRole.USER, content="hello"))
        msgs = mem.to_llm_messages(conv)
        assert len(msgs) == 1
        assert msgs[0]["role"] == "user"
