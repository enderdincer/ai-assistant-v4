"""Memory-related message schemas."""

from dataclasses import dataclass, field
from typing import Any

from ai_assistant.shared.messages.base import BaseMessage


@dataclass
class MemoryQueryMessage(BaseMessage):
    """Message requesting context from memory service.

    Published to: all/memory/query

    Attributes:
        request_id: Unique ID for correlating response
        session_id: Current session ID
        query: Query text to find relevant context
        max_facts: Maximum number of facts to return
        max_similar: Maximum number of similar conversations
    """

    request_id: str = ""
    session_id: str = ""
    query: str = ""
    max_facts: int = 5
    max_similar: int = 3

    @classmethod
    def create(
        cls,
        query: str,
        session_id: str = "",
        max_facts: int = 5,
        max_similar: int = 3,
    ) -> "MemoryQueryMessage":
        """Create a memory query message.

        Args:
            query: Query text
            session_id: Session ID
            max_facts: Max facts to return
            max_similar: Max similar conversations

        Returns:
            MemoryQueryMessage instance
        """
        import uuid

        return cls(
            request_id=str(uuid.uuid4()),
            session_id=session_id,
            query=query,
            max_facts=max_facts,
            max_similar=max_similar,
            source="assistant-service",
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MemoryQueryMessage":
        """Create from dictionary."""
        return cls(
            message_id=data.get("message_id", ""),
            timestamp=data.get("timestamp", 0.0),
            source=data.get("source", ""),
            request_id=data.get("request_id", ""),
            session_id=data.get("session_id", ""),
            query=data.get("query", ""),
            max_facts=data.get("max_facts", 5),
            max_similar=data.get("max_similar", 3),
        )


@dataclass
class MemoryResponseMessage(BaseMessage):
    """Message containing memory context response.

    Published to: all/memory/response

    Attributes:
        request_id: Correlates with original query
        relevant_facts: List of relevant facts
        recent_messages: Recent conversation messages
        similar_conversations: Similar past conversations
    """

    request_id: str = ""
    relevant_facts: list[dict[str, Any]] = field(default_factory=list)
    recent_messages: list[dict[str, Any]] = field(default_factory=list)
    similar_conversations: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MemoryResponseMessage":
        """Create from dictionary."""
        return cls(
            message_id=data.get("message_id", ""),
            timestamp=data.get("timestamp", 0.0),
            source=data.get("source", ""),
            request_id=data.get("request_id", ""),
            relevant_facts=data.get("relevant_facts", []),
            recent_messages=data.get("recent_messages", []),
            similar_conversations=data.get("similar_conversations", []),
        )


@dataclass
class MemoryStoreMessage(BaseMessage):
    """Message requesting to store a conversation message.

    Published to: all/memory/store

    Attributes:
        session_id: Session ID for the conversation
        role: Message role (user/assistant)
        content: Message content
        metadata: Optional metadata dictionary
    """

    session_id: str = ""
    role: str = ""
    content: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        session_id: str,
        role: str,
        content: str,
        metadata: dict[str, Any] | None = None,
        source: str = "",
    ) -> "MemoryStoreMessage":
        """Create a memory store message.

        Args:
            session_id: Session ID
            role: Message role (user/assistant)
            content: Message content
            metadata: Optional metadata
            source: Source service name

        Returns:
            MemoryStoreMessage instance
        """
        return cls(
            session_id=session_id,
            role=role,
            content=content,
            metadata=metadata or {},
            source=source or "unknown",
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MemoryStoreMessage":
        """Create from dictionary."""
        return cls(
            message_id=data.get("message_id", ""),
            timestamp=data.get("timestamp", 0.0),
            source=data.get("source", ""),
            session_id=data.get("session_id", ""),
            role=data.get("role", ""),
            content=data.get("content", ""),
            metadata=data.get("metadata", {}),
        )


@dataclass
class FactMessage(BaseMessage):
    """Message containing extracted facts.

    Published to: all/memory/facts

    Attributes:
        facts: List of extracted facts
        source_event_id: ID of the event these facts were extracted from
    """

    facts: list[dict[str, Any]] = field(default_factory=list)
    source_event_id: str = ""

    @classmethod
    def create(
        cls,
        facts: list[dict[str, Any]],
        source_event_id: str = "",
    ) -> "FactMessage":
        """Create a fact message.

        Each fact in the list should have:
        - subject: Who/what the fact is about
        - attribute: What aspect
        - value: The fact value
        - confidence: 0-1 confidence score
        - category: Fact category (e.g., "personal", "preference")

        Args:
            facts: List of fact dictionaries
            source_event_id: Source event ID

        Returns:
            FactMessage instance
        """
        return cls(
            facts=facts,
            source_event_id=source_event_id,
            source="extraction-service",
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FactMessage":
        """Create from dictionary."""
        return cls(
            message_id=data.get("message_id", ""),
            timestamp=data.get("timestamp", 0.0),
            source=data.get("source", ""),
            facts=data.get("facts", []),
            source_event_id=data.get("source_event_id", ""),
        )
