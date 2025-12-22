"""Thread-safe priority queue for events."""

import queue
import threading
from typing import Any, Optional
from ai_assistant.shared.interfaces import IEvent
from ai_assistant.shared.logging import get_logger

logger = get_logger(__name__)


class EventPriorityQueue:
    """Thread-safe priority queue for events.

    Events are ordered by priority (higher priority first), then by timestamp
    (earlier timestamps first).
    """

    def __init__(self, maxsize: int = 0) -> None:
        """Initialize the priority queue.

        Args:
            maxsize: Maximum queue size (0 = unlimited)
        """
        self._queue: queue.PriorityQueue[Any] = queue.PriorityQueue(maxsize=maxsize)
        self._lock = threading.Lock()
        self._total_events_received = 0
        self._total_events_processed = 0

    def put(self, event: IEvent, block: bool = True, timeout: Optional[float] = None) -> None:
        """Put an event into the queue.

        Args:
            event: The event to enqueue
            block: Whether to block if queue is full
            timeout: Timeout in seconds (None = wait forever)

        Raises:
            queue.Full: If queue is full and block=False or timeout expires
        """
        try:
            self._queue.put(event, block=block, timeout=timeout)
            with self._lock:
                self._total_events_received += 1
            logger.debug(
                f"Event enqueued: {event.event_type} from {event.source} "
                f"(priority={event.priority.name}, queue_size={self.qsize()})"
            )
        except queue.Full:
            logger.warning(f"Queue full, dropping event: {event.event_type} from {event.source}")
            raise

    def get(self, block: bool = True, timeout: Optional[float] = None) -> IEvent:
        """Get an event from the queue.

        Args:
            block: Whether to block if queue is empty
            timeout: Timeout in seconds (None = wait forever)

        Returns:
            IEvent: The highest priority event

        Raises:
            queue.Empty: If queue is empty and block=False or timeout expires
        """
        event: IEvent = self._queue.get(block=block, timeout=timeout)
        with self._lock:
            self._total_events_processed += 1
        logger.debug(
            f"Event dequeued: {event.event_type} from {event.source} "
            f"(priority={event.priority.name}, queue_size={self.qsize()})"
        )
        return event

    def qsize(self) -> int:
        """Get the approximate size of the queue.

        Returns:
            int: Number of events in queue
        """
        return self._queue.qsize()

    def empty(self) -> bool:
        """Check if the queue is empty.

        Returns:
            bool: True if queue is empty
        """
        return self._queue.empty()

    def full(self) -> bool:
        """Check if the queue is full.

        Returns:
            bool: True if queue is full
        """
        return self._queue.full()

    def clear(self) -> None:
        """Clear all events from the queue."""
        with self._lock:
            while not self._queue.empty():
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    break
        logger.info("Queue cleared")

    def get_stats(self) -> dict[str, int]:
        """Get queue statistics.

        Returns:
            dict: Statistics including total events received/processed
        """
        with self._lock:
            return {
                "current_size": self.qsize(),
                "total_received": self._total_events_received,
                "total_processed": self._total_events_processed,
                "pending": self._total_events_received - self._total_events_processed,
            }
