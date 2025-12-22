"""Log level definitions."""

from enum import IntEnum


class LogLevel(IntEnum):
    """Log levels for the application."""

    DEBUG = 10
    INFO = 20
    WARNING = 30
    ERROR = 40
    CRITICAL = 50
