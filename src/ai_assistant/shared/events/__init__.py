"""Event system implementation."""

from ai_assistant.shared.events.event import Event
from ai_assistant.shared.events.specialized import (
    CameraFrameEvent,
    TextInputEvent,
    AudioSampleEvent,
)
from ai_assistant.shared.events.priority_queue import EventPriorityQueue
from ai_assistant.shared.events.event_bus import EventBus

__all__ = [
    # Base event
    "Event",
    # Specialized events
    "CameraFrameEvent",
    "TextInputEvent",
    "AudioSampleEvent",
    # Queue and bus
    "EventPriorityQueue",
    "EventBus",
]
