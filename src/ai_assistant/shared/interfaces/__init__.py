"""Core protocol interfaces for the AI assistant system."""

from ai_assistant.shared.interfaces.lifecycle import ILifecycle
from ai_assistant.shared.interfaces.event import IEvent, EventPriority
from ai_assistant.shared.interfaces.pubsub import (
    IPublisher,
    ISubscriber,
    EventHandler,
)
from ai_assistant.shared.interfaces.event_bus import IEventBus
from ai_assistant.shared.interfaces.input_source import IInputSource
from ai_assistant.shared.interfaces.processor import IProcessor

__all__ = [
    # Lifecycle
    "ILifecycle",
    # Events
    "IEvent",
    "EventPriority",
    # Pub-Sub
    "IPublisher",
    "ISubscriber",
    "EventHandler",
    # Event Bus
    "IEventBus",
    # Input Sources
    "IInputSource",
    # Processors
    "IProcessor",
]
