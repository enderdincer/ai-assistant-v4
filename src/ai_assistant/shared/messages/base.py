"""Base message class for all MQTT messages."""

import json
import time
import uuid
from dataclasses import dataclass, field, asdict
from typing import Any, TypeVar, Type
import base64

import numpy as np


T = TypeVar("T", bound="BaseMessage")


@dataclass
class BaseMessage:
    """Base class for all inter-service messages.

    All messages include:
    - message_id: Unique identifier for this message
    - timestamp: Unix timestamp when message was created
    - source: Service/machine that created this message
    """

    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=time.time)
    source: str = ""

    def to_json(self) -> str:
        """Serialize message to JSON string."""
        return json.dumps(self._to_dict(), cls=MessageEncoder)

    def to_bytes(self) -> bytes:
        """Serialize message to bytes for MQTT."""
        return self.to_json().encode("utf-8")

    def _to_dict(self) -> dict[str, Any]:
        """Convert message to dictionary for serialization."""
        return asdict(self)

    @classmethod
    def from_json(cls: Type[T], json_str: str) -> T:
        """Deserialize message from JSON string."""
        data = json.loads(json_str, cls=MessageDecoder)
        return cls.from_dict(data)

    @classmethod
    def from_bytes(cls: Type[T], data: bytes) -> T:
        """Deserialize message from bytes."""
        return cls.from_json(data.decode("utf-8"))

    @classmethod
    def from_dict(cls: Type[T], data: dict[str, Any]) -> T:
        """Create message from dictionary.

        Subclasses may override this for custom deserialization.
        """
        # Filter to only include fields that exist in the dataclass
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_data = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered_data)


class MessageEncoder(json.JSONEncoder):
    """JSON encoder for message objects with numpy support."""

    def default(self, o: Any) -> Any:
        """Encode special types."""
        if isinstance(o, np.ndarray):
            # Encode numpy arrays as base64
            return {
                "__numpy__": True,
                "dtype": str(o.dtype),
                "shape": o.shape,
                "data": base64.b64encode(o.tobytes()).decode("ascii"),
            }
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, bytes):
            return {
                "__bytes__": True,
                "data": base64.b64encode(o).decode("ascii"),
            }
        return super().default(o)


class MessageDecoder(json.JSONDecoder):
    """JSON decoder for message objects with numpy support."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(object_hook=self._object_hook, *args, **kwargs)

    def _object_hook(self, obj: dict[str, Any]) -> Any:
        """Decode special types."""
        if "__numpy__" in obj:
            data = base64.b64decode(obj["data"])
            dtype = np.dtype(obj["dtype"])
            array = np.frombuffer(data, dtype=dtype)
            return array.reshape(obj["shape"])
        if "__bytes__" in obj:
            return base64.b64decode(obj["data"])
        return obj
