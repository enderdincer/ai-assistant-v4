"""Configuration for the perception system."""

from dataclasses import dataclass, field
from typing import Dict, Any, List
from pathlib import Path
from ai_assistant.shared.logging import LogLevel


@dataclass
class PerceptionConfig:
    """Configuration for the perception system.

    This defines all settings for the perception system including
    logging, threading, event bus, and default input sources.
    """

    # Logging configuration
    log_level: LogLevel = LogLevel.INFO
    log_file: Path | None = None
    colored_console: bool = True

    # Event bus configuration
    max_queue_size: int = 1000

    # Thread pool configuration
    max_worker_threads: int = 4

    # Input source configurations
    input_sources: List[Dict[str, Any]] = field(default_factory=list)

    @classmethod
    def default(cls) -> "PerceptionConfig":
        """Create default configuration.

        Returns:
            PerceptionConfig: Default configuration
        """
        return cls(
            log_level=LogLevel.INFO,
            log_file=None,
            colored_console=True,
            max_queue_size=1000,
            max_worker_threads=4,
            input_sources=[],
        )

    @classmethod
    def with_camera(cls, device_id: int = 0, fps: int = 30, **kwargs: Any) -> "PerceptionConfig":
        """Create configuration with a camera input.

        Args:
            device_id: Camera device ID
            fps: Target frames per second
            **kwargs: Additional configuration options

        Returns:
            PerceptionConfig: Configuration with camera
        """
        config = cls.default()
        config.input_sources.append(
            {
                "type": "camera",
                "source_id": f"camera_{device_id}",
                "config": {"device_id": device_id, "fps": fps, **kwargs},
            }
        )
        return config
