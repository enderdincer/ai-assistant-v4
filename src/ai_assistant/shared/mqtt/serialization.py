"""Message serialization and deserialization for MQTT transport.

Handles conversion between Event objects and MQTT message payloads.
Supports both JSON (for simple data) and MessagePack (for binary data like frames).
"""

import base64
import json
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Optional, Protocol, runtime_checkable

import numpy as np

from ai_assistant.shared.interfaces import EventPriority, IEvent
from ai_assistant.shared.events import Event


@runtime_checkable
class IMessageSerializer(Protocol):
    """Protocol for message serializers."""

    def serialize(self, event: IEvent) -> bytes:
        """Serialize an event to bytes for MQTT transport."""
        ...

    def deserialize(self, payload: bytes, topic: str) -> IEvent:
        """Deserialize bytes from MQTT to an event."""
        ...


class NumpyJSONEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy arrays and other special types."""

    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.ndarray):
            # For small arrays, include data inline as base64
            # For large arrays (>1MB), just include metadata
            if obj.nbytes > 1_000_000:
                return {
                    "__numpy__": True,
                    "__large__": True,
                    "shape": list(obj.shape),
                    "dtype": str(obj.dtype),
                    "nbytes": obj.nbytes,
                }
            return {
                "__numpy__": True,
                "data": base64.b64encode(obj.tobytes()).decode("ascii"),
                "shape": list(obj.shape),
                "dtype": str(obj.dtype),
            }
        if isinstance(obj, datetime):
            return {"__datetime__": True, "value": obj.isoformat()}
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, bytes):
            return {"__bytes__": True, "data": base64.b64encode(obj).decode("ascii")}
        return super().default(obj)


def _json_object_hook(obj: dict[str, Any]) -> Any:
    """JSON decoder hook for special types."""
    if "__numpy__" in obj:
        if obj.get("__large__"):
            # Return metadata only for large arrays
            return {
                "type": "numpy_array_ref",
                "shape": tuple(obj["shape"]),
                "dtype": obj["dtype"],
                "nbytes": obj["nbytes"],
            }
        data = base64.b64decode(obj["data"])
        dtype = np.dtype(obj["dtype"])
        shape = tuple(obj["shape"])
        return np.frombuffer(data, dtype=dtype).reshape(shape)
    if "__datetime__" in obj:
        return datetime.fromisoformat(obj["value"])
    if "__bytes__" in obj:
        return base64.b64decode(obj["data"])
    return obj


class JSONSerializer:
    """JSON-based serializer for MQTT messages.

    Suitable for most events. Handles numpy arrays via base64 encoding.
    """

    def serialize(self, event: IEvent) -> bytes:
        """Serialize an event to JSON bytes.

        Args:
            event: Event to serialize

        Returns:
            JSON-encoded bytes
        """
        payload = {
            "event_type": event.event_type,
            "source": event.source,
            "priority": event.priority.value,
            "timestamp": event.timestamp.isoformat(),
            "data": event.data,
        }
        return json.dumps(payload, cls=NumpyJSONEncoder).encode("utf-8")

    def deserialize(self, payload: bytes, topic: str) -> IEvent:
        """Deserialize JSON bytes to an event.

        Args:
            payload: JSON-encoded bytes
            topic: MQTT topic (used for context if event_type missing)

        Returns:
            Reconstructed Event object
        """
        data = json.loads(payload.decode("utf-8"), object_hook=_json_object_hook)

        return Event(
            event_type=data["event_type"],
            source=data["source"],
            priority=EventPriority(data["priority"]),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            data=data["data"],
        )


class ActionMessage:
    """Represents an action message for the actions/ topic hierarchy.

    Action messages have a different structure than events - they're
    requests to perform an action rather than notifications of what happened.
    """

    def __init__(
        self,
        action: str,
        params: dict[str, Any],
        request_id: Optional[str] = None,
        priority: int = 1,
    ):
        """Initialize an action message.

        Args:
            action: Action type (e.g., 'speak', 'display')
            params: Action parameters
            request_id: Optional unique request ID for tracking
            priority: Priority level (0-3)
        """
        self.action = action
        self.params = params
        self.request_id = request_id
        self.priority = priority
        self.timestamp = datetime.now()

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "action": self.action,
            "params": self.params,
            "request_id": self.request_id,
            "priority": self.priority,
            "timestamp": self.timestamp.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ActionMessage":
        """Create from dictionary."""
        msg = cls(
            action=data["action"],
            params=data["params"],
            request_id=data.get("request_id"),
            priority=data.get("priority", 1),
        )
        if "timestamp" in data:
            msg.timestamp = datetime.fromisoformat(data["timestamp"])
        return msg

    def serialize(self) -> bytes:
        """Serialize to JSON bytes."""
        return json.dumps(self.to_dict(), cls=NumpyJSONEncoder).encode("utf-8")

    @classmethod
    def deserialize(cls, payload: bytes) -> "ActionMessage":
        """Deserialize from JSON bytes."""
        data = json.loads(payload.decode("utf-8"), object_hook=_json_object_hook)
        return cls.from_dict(data)


# Pre-built action message factories
class ActionMessages:
    """Factory methods for common action messages."""

    @staticmethod
    def speech(
        text: str,
        voice: str = "af_heart",
        speed: float = 1.0,
        request_id: Optional[str] = None,
    ) -> ActionMessage:
        """Create a speech/TTS action message.

        Args:
            text: Text to speak
            voice: Voice ID to use
            speed: Speech speed multiplier
            request_id: Optional request ID

        Returns:
            ActionMessage for TTS
        """
        return ActionMessage(
            action="speak",
            params={
                "text": text,
                "voice": voice,
                "speed": speed,
            },
            request_id=request_id,
            priority=2,  # High priority for user-facing actions
        )

    @staticmethod
    def notification(
        message: str,
        level: str = "info",
        title: Optional[str] = None,
        request_id: Optional[str] = None,
    ) -> ActionMessage:
        """Create a notification action message.

        Args:
            message: Notification message
            level: Notification level (info, warning, error)
            title: Optional notification title
            request_id: Optional request ID

        Returns:
            ActionMessage for notification
        """
        return ActionMessage(
            action="notify",
            params={
                "message": message,
                "level": level,
                "title": title,
            },
            request_id=request_id,
        )
