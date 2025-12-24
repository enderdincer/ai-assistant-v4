from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any
from uuid import uuid4


class FactCategory(StrEnum):
    SYSTEM = "system"
    USER = "user"
    LEARNED = "learned"


@dataclass(frozen=True)
class Fact:
    category: FactCategory
    subject: str
    attribute: str
    value: str
    id: str = field(default_factory=lambda: str(uuid4()))
    confidence: float = 1.0
    source: str = "system"
    is_immutable: bool = False
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    @staticmethod
    def create_system_fact(
        subject: str,
        attribute: str,
        value: str,
    ) -> "Fact":
        return Fact(
            category=FactCategory.SYSTEM,
            subject=subject,
            attribute=attribute,
            value=value,
            source="system",
            is_immutable=True,
            confidence=1.0,
        )

    @staticmethod
    def create_learned_fact(
        subject: str,
        attribute: str,
        value: str,
        confidence: float = 0.8,
        source: str = "conversation",
    ) -> "Fact":
        return Fact(
            category=FactCategory.LEARNED,
            subject=subject,
            attribute=attribute,
            value=value,
            source=source,
            is_immutable=False,
            confidence=confidence,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "category": self.category.value,
            "subject": self.subject,
            "attribute": self.attribute,
            "value": self.value,
            "confidence": self.confidence,
            "source": self.source,
            "is_immutable": self.is_immutable,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }

    def to_text(self) -> str:
        return f"{self.subject} {self.attribute}: {self.value}"


@dataclass(frozen=True)
class EventLogEntry:
    event_type: str
    session_id: str
    data: dict[str, Any]
    id: str = field(default_factory=lambda: str(uuid4()))
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "event_type": self.event_type,
            "session_id": self.session_id,
            "data": self.data,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass(frozen=True)
class Message:
    session_id: str
    role: str
    content: str
    id: str = field(default_factory=lambda: str(uuid4()))
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "session_id": self.session_id,
            "role": self.role,
            "content": self.content,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class MemoryContext:
    relevant_facts: list[Fact] = field(default_factory=list)
    recent_messages: list[Message] = field(default_factory=list)
    similar_conversations: list[Message] = field(default_factory=list)

    def to_system_prompt_addition(self) -> str:
        parts: list[str] = []

        if self.relevant_facts:
            facts_text = "\n".join(f"- {f.to_text()}" for f in self.relevant_facts)
            parts.append(f"Known facts:\n{facts_text}")

        if self.similar_conversations:
            similar_text = "\n".join(
                f"- [{m.role}]: {m.content[:100]}..."
                if len(m.content) > 100
                else f"- [{m.role}]: {m.content}"
                for m in self.similar_conversations[:5]
            )
            parts.append(f"Relevant past conversations:\n{similar_text}")

        return "\n\n".join(parts)

    def is_empty(self) -> bool:
        return (
            not self.relevant_facts and not self.recent_messages and not self.similar_conversations
        )


@dataclass(frozen=True)
class ConversationSummary:
    session_id: str
    summary: str
    message_count: int
    first_message_id: str
    last_message_id: str
    first_timestamp: datetime
    last_timestamp: datetime
    id: str = field(default_factory=lambda: str(uuid4()))
    created_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "session_id": self.session_id,
            "summary": self.summary,
            "message_count": self.message_count,
            "first_message_id": self.first_message_id,
            "last_message_id": self.last_message_id,
            "first_timestamp": self.first_timestamp.isoformat(),
            "last_timestamp": self.last_timestamp.isoformat(),
            "created_at": self.created_at.isoformat(),
        }
