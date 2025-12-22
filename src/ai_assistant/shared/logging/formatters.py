"""Custom log formatters for thread-aware logging."""

import logging
import threading
from datetime import datetime


class ThreadAwareFormatter(logging.Formatter):
    """Log formatter that includes thread information.

    Format: [timestamp] [level] [thread_name:thread_id] [module.function:line] message
    """

    def __init__(self) -> None:
        """Initialize the formatter with a specific format string."""
        super().__init__(
            fmt="[%(asctime)s] [%(levelname)-8s] [%(thread_name)s:%(thread_id)s] "
            "[%(name)s.%(funcName)s:%(lineno)d] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    def format(self, record: logging.LogRecord) -> str:
        """Format the log record with thread information.

        Args:
            record: The log record to format

        Returns:
            str: Formatted log message
        """
        # Add thread information to the record
        thread = threading.current_thread()
        setattr(record, "thread_name", thread.name)
        setattr(record, "thread_id", thread.ident or 0)

        return super().format(record)


class ColoredThreadAwareFormatter(ThreadAwareFormatter):
    """Thread-aware formatter with color support for console output."""

    # ANSI color codes
    COLORS = {
        "DEBUG": "\033[36m",  # Cyan
        "INFO": "\033[32m",  # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[35m",  # Magenta
        "RESET": "\033[0m",  # Reset
    }

    def format(self, record: logging.LogRecord) -> str:
        """Format the log record with colors.

        Args:
            record: The log record to format

        Returns:
            str: Colored formatted log message
        """
        # Get the base formatted message
        formatted = super().format(record)

        # Add color to the level name
        level_color = self.COLORS.get(record.levelname, self.COLORS["RESET"])
        reset_color = self.COLORS["RESET"]

        # Replace the level name with colored version
        formatted = formatted.replace(
            f"[{record.levelname}", f"[{level_color}{record.levelname}{reset_color}"
        )

        return formatted
