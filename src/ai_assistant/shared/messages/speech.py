"""Speech-related message schemas (TTS)."""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional

from ai_assistant.shared.messages.base import BaseMessage


class SpeakerState(str, Enum):
    """Speaker activity states."""

    SPEAKING_STARTED = "speaking_started"
    SPEAKING_ENDED = "speaking_ended"


class SpeechControlAction(str, Enum):
    """Speech control actions."""

    SKIP_CURRENT = "skip_current"  # Stop current speech, play next in queue
    SKIP_ALL = "skip_all"  # Stop current speech and clear queue
    CLEAR_QUEUE = "clear_queue"  # Clear queue but don't stop current
    PAUSE = "pause"  # Pause playback (future)
    RESUME = "resume"  # Resume playback (future)


@dataclass
class SpeechRequestMessage(BaseMessage):
    """Message requesting text-to-speech synthesis and playback.

    Published to: all/actions/speech

    Attributes:
        text: Text to synthesize and speak
        voice: Voice name (e.g., "af_bella", "am_adam")
        speed: Speech speed multiplier (1.0 = normal)
        request_id: Unique ID for tracking this request
        priority: Priority level (higher = more urgent)
    """

    text: str = ""
    voice: str = "af_bella"
    speed: float = 1.0
    request_id: str = ""
    priority: int = 0

    @classmethod
    def create(
        cls,
        text: str,
        voice: str = "af_bella",
        speed: float = 1.0,
        priority: int = 0,
        source: str = "",
    ) -> "SpeechRequestMessage":
        """Create a speech request message.

        Args:
            text: Text to speak
            voice: Voice name
            speed: Speech speed multiplier
            priority: Priority level
            source: Source service name

        Returns:
            SpeechRequestMessage instance
        """
        import uuid

        return cls(
            text=text,
            voice=voice,
            speed=speed,
            request_id=str(uuid.uuid4()),
            priority=priority,
            source=source,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SpeechRequestMessage":
        """Create from dictionary."""
        return cls(
            message_id=data.get("message_id", ""),
            timestamp=data.get("timestamp", 0.0),
            source=data.get("source", ""),
            text=data.get("text", ""),
            voice=data.get("voice", "af_bella"),
            speed=data.get("speed", 1.0),
            request_id=data.get("request_id", ""),
            priority=data.get("priority", 0),
        )


@dataclass
class SpeakerActivityMessage(BaseMessage):
    """Message indicating TTS playback state changes.

    Published to: all/events/speaker-activity

    Used for echo prevention - other services can pause
    transcription while TTS is playing.

    Attributes:
        state: Current speaker state (started/ended)
        text: Text being spoken (for echo filtering)
        duration_ms: Estimated/actual duration in milliseconds
        request_id: ID of the speech request this relates to
    """

    state: SpeakerState = SpeakerState.SPEAKING_ENDED
    text: str = ""
    duration_ms: float = 0.0
    request_id: str = ""

    def _to_dict(self) -> dict[str, Any]:
        """Convert to dict with enum serialization."""
        data = super()._to_dict()
        data["state"] = self.state.value
        return data

    @classmethod
    def create_started(
        cls,
        text: str,
        duration_ms: float,
        request_id: str = "",
    ) -> "SpeakerActivityMessage":
        """Create a speaking started message.

        Args:
            text: Text being spoken
            duration_ms: Estimated duration in milliseconds
            request_id: Speech request ID

        Returns:
            SpeakerActivityMessage instance
        """
        return cls(
            state=SpeakerState.SPEAKING_STARTED,
            text=text,
            duration_ms=duration_ms,
            request_id=request_id,
            source="speech-service",
        )

    @classmethod
    def create_ended(
        cls,
        text: str,
        duration_ms: float,
        request_id: str = "",
    ) -> "SpeakerActivityMessage":
        """Create a speaking ended message.

        Args:
            text: Text that was spoken
            duration_ms: Actual duration in milliseconds
            request_id: Speech request ID

        Returns:
            SpeakerActivityMessage instance
        """
        return cls(
            state=SpeakerState.SPEAKING_ENDED,
            text=text,
            duration_ms=duration_ms,
            request_id=request_id,
            source="speech-service",
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SpeakerActivityMessage":
        """Create from dictionary."""
        state_str = data.get("state", "speaking_ended")
        state = SpeakerState(state_str) if isinstance(state_str, str) else state_str

        return cls(
            message_id=data.get("message_id", ""),
            timestamp=data.get("timestamp", 0.0),
            source=data.get("source", ""),
            state=state,
            text=data.get("text", ""),
            duration_ms=data.get("duration_ms", 0.0),
            request_id=data.get("request_id", ""),
        )


@dataclass
class SpeechControlMessage(BaseMessage):
    """Message for controlling speech playback.

    Published to: all/actions/speech-control

    Allows clients to skip, pause, or clear the speech queue.

    Attributes:
        action: Control action to perform
        request_id: Optional specific request ID to target
    """

    action: SpeechControlAction = SpeechControlAction.SKIP_CURRENT
    request_id: str = ""  # Optional: target specific request

    def _to_dict(self) -> dict[str, Any]:
        """Convert to dict with enum serialization."""
        data = super()._to_dict()
        data["action"] = self.action.value
        return data

    @classmethod
    def create(
        cls,
        action: SpeechControlAction,
        request_id: str = "",
        source: str = "",
    ) -> "SpeechControlMessage":
        """Create a speech control message.

        Args:
            action: Control action to perform
            request_id: Optional specific request ID to target
            source: Source service name

        Returns:
            SpeechControlMessage instance
        """
        return cls(
            action=action,
            request_id=request_id,
            source=source,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SpeechControlMessage":
        """Create from dictionary."""
        action_str = data.get("action", "skip_current")
        action = SpeechControlAction(action_str) if isinstance(action_str, str) else action_str

        return cls(
            message_id=data.get("message_id", ""),
            timestamp=data.get("timestamp", 0.0),
            source=data.get("source", ""),
            action=action,
            request_id=data.get("request_id", ""),
        )
