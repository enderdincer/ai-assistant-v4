"""Event interface for the pub-sub system."""

from typing import Any, Protocol, runtime_checkable
from datetime import datetime
from enum import IntEnum


class EventPriority(IntEnum):
    """Priority levels for events in the system."""

    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3


@runtime_checkable
class IEvent(Protocol):
    """Interface for events in the system.

    Events are the primary communication mechanism between components.
    """

    @property
    def event_type(self) -> str:
        """Get the type identifier of this event.

        Returns:
            str: Event type (e.g., 'camera.frame', 'audio.sample')
        """
        ...

    @property
    def source(self) -> str:
        """Get the source identifier that produced this event.

        Returns:
            str: Source identifier (e.g., 'camera_0', 'microphone_1')
        """
        ...

    @property
    def priority(self) -> EventPriority:
        """Get the priority of this event.

        Returns:
            EventPriority: Priority level
        """
        ...

    @property
    def timestamp(self) -> datetime:
        """Get the timestamp when this event was created.

        Returns:
            datetime: Creation timestamp
        """
        ...

    @property
    def data(self) -> Any:
        """Get the event payload data.

        Returns:
            Any: Event-specific data payload
        """
        ...

    def to_dict(self) -> dict[str, Any]:
        """Convert event to dictionary representation.

        Returns:
            dict: Dictionary containing event data
        """
        ...
