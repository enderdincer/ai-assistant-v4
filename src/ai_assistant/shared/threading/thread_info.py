"""Thread information tracking."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
import threading


@dataclass
class ThreadInfo:
    """Information about a managed thread."""

    name: str
    thread_id: Optional[int]
    component_name: str
    started_at: datetime
    is_alive: bool = True
    error: Optional[Exception] = None
    stopped_at: Optional[datetime] = None

    @classmethod
    def from_thread(cls, thread: threading.Thread, component_name: str) -> "ThreadInfo":
        """Create ThreadInfo from a threading.Thread.

        Args:
            thread: The thread to track
            component_name: Name of the component using this thread

        Returns:
            ThreadInfo: Thread information instance
        """
        return cls(
            name=thread.name,
            thread_id=thread.ident,
            component_name=component_name,
            started_at=datetime.now(),
            is_alive=thread.is_alive(),
        )

    def update_status(self, thread: threading.Thread) -> None:
        """Update the status from the thread.

        Args:
            thread: The thread to check
        """
        self.is_alive = thread.is_alive()
        self.thread_id = thread.ident
        if not self.is_alive and self.stopped_at is None:
            self.stopped_at = datetime.now()

    def mark_error(self, error: Exception) -> None:
        """Mark the thread as having encountered an error.

        Args:
            error: The exception that occurred
        """
        self.error = error
        self.is_alive = False
        self.stopped_at = datetime.now()
