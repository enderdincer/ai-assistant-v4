"""Shared message schemas for inter-service communication."""

from ai_assistant.shared.messages.audio import AudioSampleMessage
from ai_assistant.shared.messages.transcription import TranscriptionMessage
from ai_assistant.shared.messages.speech import (
    SpeechRequestMessage,
    SpeakerActivityMessage,
    SpeechControlMessage,
    SpeechControlAction,
)
from ai_assistant.shared.messages.assistant import (
    TextInputMessage,
    AssistantResponseMessage,
)
from ai_assistant.shared.messages.memory import (
    MemoryQueryMessage,
    MemoryResponseMessage,
    MemoryStoreMessage,
    FactMessage,
)
from ai_assistant.shared.messages.session import (
    SessionChangedMessage,
    CurrentSessionMessage,
)
from ai_assistant.shared.messages.base import BaseMessage

__all__ = [
    "BaseMessage",
    "AudioSampleMessage",
    "TranscriptionMessage",
    "SpeechRequestMessage",
    "SpeakerActivityMessage",
    "SpeechControlMessage",
    "SpeechControlAction",
    "TextInputMessage",
    "AssistantResponseMessage",
    "MemoryQueryMessage",
    "MemoryResponseMessage",
    "MemoryStoreMessage",
    "FactMessage",
    "SessionChangedMessage",
    "CurrentSessionMessage",
]
