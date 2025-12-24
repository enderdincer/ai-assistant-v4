"""Assistant-related message schemas."""

from dataclasses import dataclass
from typing import Any, Optional

from ai_assistant.shared.messages.base import BaseMessage


@dataclass
class TextInputMessage(BaseMessage):
    """Message containing user text input.

    Published to: all/events/text-input

    Used by text interaction service to send user messages
    to the assistant service.

    Attributes:
        text: User's text message
        session_id: Session identifier for conversation continuity
        client_id: ID of the client that sent this message
    """

    text: str = ""
    session_id: str = ""
    client_id: str = ""

    @classmethod
    def create(
        cls,
        text: str,
        session_id: str = "",
        client_id: str = "",
    ) -> "TextInputMessage":
        """Create a text input message.

        Args:
            text: User's text message
            session_id: Session ID for conversation
            client_id: Client identifier

        Returns:
            TextInputMessage instance
        """
        return cls(
            text=text,
            session_id=session_id,
            client_id=client_id,
            source=f"text-client-{client_id}" if client_id else "text-client",
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TextInputMessage":
        """Create from dictionary."""
        return cls(
            message_id=data.get("message_id", ""),
            timestamp=data.get("timestamp", 0.0),
            source=data.get("source", ""),
            text=data.get("text", ""),
            session_id=data.get("session_id", ""),
            client_id=data.get("client_id", ""),
        )


@dataclass
class AssistantResponseMessage(BaseMessage):
    """Message containing assistant's response.

    Published to: all/events/assistant-response

    Attributes:
        text: Assistant's response text
        session_id: Session identifier
        input_text: Original user input that triggered this response
        model_name: LLM model used to generate response
        tokens_used: Number of tokens used (if available)
    """

    text: str = ""
    session_id: str = ""
    input_text: str = ""
    model_name: str = ""
    tokens_used: int = 0

    @classmethod
    def create(
        cls,
        text: str,
        session_id: str = "",
        input_text: str = "",
        model_name: str = "",
        tokens_used: int = 0,
    ) -> "AssistantResponseMessage":
        """Create an assistant response message.

        Args:
            text: Response text
            session_id: Session ID
            input_text: Original user input
            model_name: LLM model name
            tokens_used: Token count

        Returns:
            AssistantResponseMessage instance
        """
        return cls(
            text=text,
            session_id=session_id,
            input_text=input_text,
            model_name=model_name,
            tokens_used=tokens_used,
            source="assistant-service",
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AssistantResponseMessage":
        """Create from dictionary."""
        return cls(
            message_id=data.get("message_id", ""),
            timestamp=data.get("timestamp", 0.0),
            source=data.get("source", ""),
            text=data.get("text", ""),
            session_id=data.get("session_id", ""),
            input_text=data.get("input_text", ""),
            model_name=data.get("model_name", ""),
            tokens_used=data.get("tokens_used", 0),
        )
