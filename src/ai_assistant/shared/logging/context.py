"""Thread-local logging context for enriched logging."""

import logging
import threading
from typing import Any, Optional
from contextlib import contextmanager
from collections.abc import Iterator


class LogContext:
    """Thread-local storage for logging context.

    This allows adding contextual information to all log messages
    within a specific thread or code block.
    """

    def __init__(self) -> None:
        """Initialize the log context."""
        self._local = threading.local()

    def set(self, key: str, value: Any) -> None:
        """Set a context value for the current thread.

        Args:
            key: Context key
            value: Context value
        """
        if not hasattr(self._local, "context"):
            self._local.context = {}
        self._local.context[key] = value

    def get(self, key: str) -> Optional[Any]:
        """Get a context value for the current thread.

        Args:
            key: Context key

        Returns:
            Optional[Any]: Context value or None if not set
        """
        if not hasattr(self._local, "context"):
            return None
        return self._local.context.get(key)

    def clear(self) -> None:
        """Clear all context for the current thread."""
        if hasattr(self._local, "context"):
            self._local.context.clear()

    def get_all(self) -> dict[str, Any]:
        """Get all context for the current thread.

        Returns:
            dict: All context key-value pairs
        """
        if not hasattr(self._local, "context"):
            return {}
        return dict(self._local.context)


# Global log context instance
_log_context = LogContext()


@contextmanager
def log_context(**kwargs: Any) -> Iterator[None]:
    """Context manager for adding temporary logging context.

    Args:
        **kwargs: Context key-value pairs to add

    Example:
        with log_context(source_id="camera_0", frame_num=42):
            logger.info("Processing frame")  # Will include source_id and frame_num
    """
    # Save current context
    old_context = _log_context.get_all().copy()

    # Set new context
    for key, value in kwargs.items():
        _log_context.set(key, value)

    try:
        yield
    finally:
        # Restore old context
        _log_context.clear()
        for key, value in old_context.items():
            _log_context.set(key, value)


def get_log_context() -> dict[str, Any]:
    """Get the current logging context.

    Returns:
        dict: Current context key-value pairs
    """
    return _log_context.get_all()
