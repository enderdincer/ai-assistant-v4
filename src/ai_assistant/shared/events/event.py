"""Concrete implementation of the Event class."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from ai_assistant.shared.interfaces import IEvent, EventPriority


@dataclass(frozen=True)
class Event:
    """Concrete implementation of IEvent.

    This is an immutable event that carries data through the system.
    """

    event_type: str
    source: str
    data: Any
    priority: EventPriority = EventPriority.NORMAL
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        """Convert event to dictionary representation.

        Returns:
            dict: Dictionary containing event data
        """
        return {
            "event_type": self.event_type,
            "source": self.source,
            "priority": self.priority.name,
            "timestamp": self.timestamp.isoformat(),
            "data": self.data,
        }

    def __lt__(self, other: object) -> bool:
        """Compare events for priority queue ordering.

        Higher priority events come first. If priorities are equal,
        earlier timestamps come first.

        Args:
            other: Another event to compare with

        Returns:
            bool: True if this event has higher priority (should come first)
        """
        if not isinstance(other, Event):
            return NotImplemented

        # Higher priority values come first (reverse order)
        if self.priority != other.priority:
            return self.priority > other.priority

        # Earlier timestamps come first (normal order)
        return self.timestamp < other.timestamp

    def __repr__(self) -> str:
        """String representation of the event.

        Returns:
            str: Human-readable representation
        """
        return (
            f"Event(type={self.event_type}, source={self.source}, "
            f"priority={self.priority.name}, timestamp={self.timestamp.isoformat()})"
        )
