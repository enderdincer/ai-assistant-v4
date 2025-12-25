"""Transcription-related message schemas."""

from dataclasses import dataclass
from typing import Any

from ai_assistant.shared.messages.base import BaseMessage


@dataclass
class TranscriptionMessage(BaseMessage):
    """Message containing speech-to-text transcription results.

    Published to: all/events/audio-transcribed

    Attributes:
        text: Transcribed text
        language: Language code (e.g., "en")
        confidence: Confidence score 0-1
        audio_duration: Duration of the audio in seconds
        audio_source: ID of the machine/source that captured the audio
        model_name: Name of the STT model used
        session_id: Session ID for conversation tracking
    """

    text: str = ""
    language: str = "en"
    confidence: float = 1.0
    audio_duration: float = 0.0
    audio_source: str = ""
    model_name: str = ""
    session_id: str = ""

    @classmethod
    def create(
        cls,
        text: str,
        audio_source: str,
        session_id: str = "",
        language: str = "en",
        confidence: float = 1.0,
        audio_duration: float = 0.0,
        model_name: str = "",
    ) -> "TranscriptionMessage":
        """Create a transcription message.

        Args:
            text: Transcribed text
            audio_source: ID of the audio source machine
            session_id: Session ID for conversation tracking
            language: Language code
            confidence: Confidence score
            audio_duration: Audio duration in seconds
            model_name: STT model name

        Returns:
            TranscriptionMessage instance
        """
        return cls(
            text=text,
            language=language,
            confidence=confidence,
            audio_duration=audio_duration,
            audio_source=audio_source,
            model_name=model_name,
            session_id=session_id,
            source="transcription-service",
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TranscriptionMessage":
        """Create from dictionary."""
        return cls(
            message_id=data.get("message_id", ""),
            timestamp=data.get("timestamp", 0.0),
            source=data.get("source", ""),
            text=data.get("text", ""),
            language=data.get("language", "en"),
            confidence=data.get("confidence", 1.0),
            audio_duration=data.get("audio_duration", 0.0),
            audio_source=data.get("audio_source", ""),
            model_name=data.get("model_name", ""),
            session_id=data.get("session_id", ""),
        )
