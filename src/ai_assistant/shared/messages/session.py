"""Session-related message schemas."""

from dataclasses import dataclass
from typing import Any, Optional

from ai_assistant.shared.messages.base import BaseMessage


@dataclass
class SessionChangedMessage(BaseMessage):
    """Message containing the current active session information.

    Published to: all/events/current-session

    This message is published by the memory service whenever the active
    session is set. Since it's idempotent, services can always get the
    latest session state by listening to this topic.

    Attributes:
        session_id: The current active session ID
        previous_session_id: The previous session ID (empty if first session)
        message_count: Number of messages in the session
        created_at: When the session was created (ISO format)
    """

    session_id: str = ""
    previous_session_id: str = ""
    message_count: int = 0
    created_at: str = ""

    @classmethod
    def create(
        cls,
        session_id: str,
        previous_session_id: str = "",
        message_count: int = 0,
        created_at: str = "",
    ) -> "SessionChangedMessage":
        """Create a session changed message.

        Args:
            session_id: The new active session ID
            previous_session_id: The previous session ID
            message_count: Number of messages in the session
            created_at: When the session was created

        Returns:
            SessionChangedMessage instance
        """
        return cls(
            session_id=session_id,
            previous_session_id=previous_session_id,
            message_count=message_count,
            created_at=created_at,
            source="memory-service",
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SessionChangedMessage":
        """Create from dictionary."""
        return cls(
            message_id=data.get("message_id", ""),
            timestamp=data.get("timestamp", 0.0),
            source=data.get("source", ""),
            session_id=data.get("session_id", ""),
            previous_session_id=data.get("previous_session_id", ""),
            message_count=data.get("message_count", 0),
            created_at=data.get("created_at", ""),
        )


@dataclass
class CurrentSessionMessage(BaseMessage):
    """Message containing current session information.

    Published to: all/events/session-changed (on service startup queries)

    This message can be used as a response when services query
    the memory service for the current active session.

    Attributes:
        session_id: The current active session ID
        message_count: Number of messages in the session
        created_at: When the session was created (ISO format)
        last_activity: When the last message was added (ISO format)
    """

    session_id: str = ""
    message_count: int = 0
    created_at: str = ""
    last_activity: str = ""

    @classmethod
    def create(
        cls,
        session_id: str,
        message_count: int = 0,
        created_at: str = "",
        last_activity: str = "",
    ) -> "CurrentSessionMessage":
        """Create a current session message.

        Args:
            session_id: The current active session ID
            message_count: Number of messages in the session
            created_at: When the session was created
            last_activity: When the last message was added

        Returns:
            CurrentSessionMessage instance
        """
        return cls(
            session_id=session_id,
            message_count=message_count,
            created_at=created_at,
            last_activity=last_activity,
            source="memory-service",
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CurrentSessionMessage":
        """Create from dictionary."""
        return cls(
            message_id=data.get("message_id", ""),
            timestamp=data.get("timestamp", 0.0),
            source=data.get("source", ""),
            session_id=data.get("session_id", ""),
            message_count=data.get("message_count", 0),
            created_at=data.get("created_at", ""),
            last_activity=data.get("last_activity", ""),
        )
