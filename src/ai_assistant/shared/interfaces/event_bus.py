"""Event bus interface for centralized event routing."""

from typing import Protocol, runtime_checkable
from ai_assistant.shared.interfaces.lifecycle import ILifecycle
from ai_assistant.shared.interfaces.pubsub import IPublisher, ISubscriber


@runtime_checkable
class IEventBus(ILifecycle, IPublisher, ISubscriber, Protocol):
    """Interface for the central event bus.

    The event bus is responsible for routing events from publishers
    to subscribers based on event types. It combines lifecycle management
    with pub-sub capabilities.
    """

    def get_queue_size(self) -> int:
        """Get the current number of events in the queue.

        Returns:
            int: Number of pending events
        """
        ...

    def clear(self) -> None:
        """Clear all pending events from the queue.

        This should only be used for testing or emergency shutdown.
        """
        ...
