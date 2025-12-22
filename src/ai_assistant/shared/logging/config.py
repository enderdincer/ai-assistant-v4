"""Logging configuration and setup."""

import logging
import sys
from pathlib import Path
from typing import Optional
from ai_assistant.shared.logging.formatters import (
    ThreadAwareFormatter,
    ColoredThreadAwareFormatter,
)
from ai_assistant.shared.logging.levels import LogLevel


class LogConfig:
    """Configuration for the logging system."""

    def __init__(
        self,
        level: LogLevel = LogLevel.INFO,
        log_file: Optional[Path] = None,
        console_output: bool = True,
        colored_console: bool = True,
    ) -> None:
        """Initialize log configuration.

        Args:
            level: Minimum log level to capture
            log_file: Optional path to log file
            console_output: Whether to output to console
            colored_console: Whether to use colored output on console
        """
        self.level = level
        self.log_file = log_file
        self.console_output = console_output
        self.colored_console = colored_console


def setup_logging(config: LogConfig) -> None:
    """Setup the logging system with the given configuration.

    Args:
        config: Logging configuration
    """
    # Get the root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(config.level)

    # Remove existing handlers
    root_logger.handlers.clear()

    # Add console handler if requested
    if config.console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(config.level)

        if config.colored_console:
            console_handler.setFormatter(ColoredThreadAwareFormatter())
        else:
            console_handler.setFormatter(ThreadAwareFormatter())

        root_logger.addHandler(console_handler)

    # Add file handler if requested
    if config.log_file:
        # Ensure log directory exists
        config.log_file.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(config.log_file)
        file_handler.setLevel(config.level)
        file_handler.setFormatter(ThreadAwareFormatter())

        root_logger.addHandler(file_handler)


def get_logger(name: str) -> logging.Logger:
    """Get a logger with the given name.

    Args:
        name: Name for the logger (typically __name__)

    Returns:
        logging.Logger: Configured logger instance
    """
    return logging.getLogger(name)
