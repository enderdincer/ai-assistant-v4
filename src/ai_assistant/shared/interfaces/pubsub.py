"""Publisher-Subscriber interfaces for event-driven communication."""

from typing import Callable, Protocol, runtime_checkable
from ai_assistant.shared.interfaces.event import IEvent


# Type alias for event handler callbacks
EventHandler = Callable[[IEvent], None]


@runtime_checkable
class IPublisher(Protocol):
    """Interface for components that publish events."""

    def publish(self, event: IEvent) -> None:
        """Publish an event to the event bus.

        Args:
            event: The event to publish

        Raises:
            RuntimeError: If publisher is not connected to an event bus
        """
        ...


@runtime_checkable
class ISubscriber(Protocol):
    """Interface for components that subscribe to events."""

    def subscribe(self, event_type: str, handler: EventHandler) -> None:
        """Subscribe to events of a specific type.

        Args:
            event_type: The type of events to subscribe to (e.g., 'camera.frame')
            handler: Callback function to handle matching events

        Raises:
            ValueError: If event_type is invalid
        """
        ...

    def unsubscribe(self, event_type: str, handler: EventHandler) -> None:
        """Unsubscribe from events of a specific type.

        Args:
            event_type: The type of events to unsubscribe from
            handler: The handler to remove

        Raises:
            ValueError: If handler is not subscribed to event_type
        """
        ...
