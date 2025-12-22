"""Central event bus implementation."""

import threading
import queue
from typing import DefaultDict, Optional
from collections import defaultdict
from ai_assistant.shared.interfaces import IEvent, EventHandler
from ai_assistant.shared.events.priority_queue import EventPriorityQueue
from ai_assistant.shared.logging import get_logger

logger = get_logger(__name__)


class EventBus:
    """Central event bus for pub-sub communication.

    The event bus routes events from publishers to subscribers based on
    event types. It uses a priority queue for event ordering and processes
    events in a dedicated thread.
    """

    def __init__(self, max_queue_size: int = 0) -> None:
        """Initialize the event bus.

        Args:
            max_queue_size: Maximum queue size (0 = unlimited)
        """
        self._queue = EventPriorityQueue(maxsize=max_queue_size)
        self._subscribers: DefaultDict[str, list[EventHandler]] = defaultdict(list)
        self._lock = threading.RLock()
        self._running = False
        self._worker_thread: Optional[threading.Thread] = None

    def initialize(self) -> None:
        """Initialize the event bus."""
        logger.info("Event bus initialized")

    def start(self) -> None:
        """Start the event bus processing."""
        with self._lock:
            if self._running:
                raise RuntimeError("Event bus is already running")

            self._running = True
            self._worker_thread = threading.Thread(
                target=self._process_events,
                name="EventBus-Worker",
                daemon=False,
            )
            self._worker_thread.start()
            logger.info("Event bus started")

    def stop(self) -> None:
        """Stop the event bus gracefully."""
        with self._lock:
            if not self._running:
                return

            self._running = False

        # Wait for worker thread to finish
        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=5.0)
            if self._worker_thread.is_alive():
                logger.warning("Event bus worker thread did not stop cleanly")

        logger.info("Event bus stopped")

    def is_running(self) -> bool:
        """Check if the event bus is running.

        Returns:
            bool: True if running
        """
        return self._running

    def publish(self, event: IEvent) -> None:
        """Publish an event to the bus.

        Args:
            event: The event to publish

        Raises:
            RuntimeError: If event bus is not running
        """
        if not self._running:
            raise RuntimeError("Event bus is not running")

        try:
            self._queue.put(event, block=False)
        except queue.Full:
            logger.error(f"Event queue full, dropping event: {event.event_type}")

    def subscribe(self, event_type: str, handler: EventHandler) -> None:
        """Subscribe to events of a specific type.

        Args:
            event_type: The event type to subscribe to
            handler: Callback function to handle events
        """
        with self._lock:
            if handler not in self._subscribers[event_type]:
                self._subscribers[event_type].append(handler)
                logger.debug(
                    f"Subscribed to {event_type} "
                    f"(total subscribers: {len(self._subscribers[event_type])})"
                )

    def unsubscribe(self, event_type: str, handler: EventHandler) -> None:
        """Unsubscribe from events of a specific type.

        Args:
            event_type: The event type to unsubscribe from
            handler: The handler to remove
        """
        with self._lock:
            if handler in self._subscribers[event_type]:
                self._subscribers[event_type].remove(handler)
                logger.debug(
                    f"Unsubscribed from {event_type} "
                    f"(remaining subscribers: {len(self._subscribers[event_type])})"
                )

    def get_queue_size(self) -> int:
        """Get the current queue size.

        Returns:
            int: Number of pending events
        """
        return self._queue.qsize()

    def clear(self) -> None:
        """Clear all pending events."""
        self._queue.clear()

    def _process_events(self) -> None:
        """Process events from the queue (runs in worker thread)."""
        logger.info("Event bus worker thread started")

        while self._running:
            try:
                # Get next event with timeout to check _running flag periodically
                event = self._queue.get(block=True, timeout=0.1)
                self._dispatch_event(event)
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error processing event: {e}", exc_info=True)

        logger.info("Event bus worker thread stopped")

    def _dispatch_event(self, event: IEvent) -> None:
        """Dispatch an event to all subscribers.

        Args:
            event: The event to dispatch
        """
        with self._lock:
            handlers = self._subscribers.get(event.event_type, []).copy()

        if not handlers:
            logger.debug(f"No subscribers for event type: {event.event_type}")
            return

        logger.debug(f"Dispatching {event.event_type} to {len(handlers)} subscriber(s)")

        for handler in handlers:
            try:
                handler(event)
            except Exception as e:
                logger.error(f"Error in event handler for {event.event_type}: {e}", exc_info=True)
