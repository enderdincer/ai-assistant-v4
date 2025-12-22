"""Configuration for the perception system."""

from dataclasses import dataclass, field
from typing import Dict, Any, List
from pathlib import Path
from ai_assistant.shared.logging import LogLevel


@dataclass
class PerceptionConfig:
    """Configuration for the perception system.

    This defines all settings for the perception system including
    logging, threading, event bus, input sources, and processors.
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

    # Processor configurations
    processors: List[Dict[str, Any]] = field(default_factory=list)

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
            processors=[],
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

    @classmethod
    def with_audio_and_stt(
        cls, audio_source_id: str = "microphone", stt_processor_id: str = "stt", **kwargs: Any
    ) -> "PerceptionConfig":
        """Create configuration with audio input and STT processor.

        Args:
            audio_source_id: Audio source identifier
            stt_processor_id: STT processor identifier
            **kwargs: Additional configuration options for STT processor

        Returns:
            PerceptionConfig: Configuration with audio and STT
        """
        config = cls.default()

        # Add audio input source
        config.input_sources.append(
            {
                "type": "audio",
                "source_id": audio_source_id,
                "config": {},
            }
        )

        # Add STT processor
        config.processors.append(
            {
                "type": "stt",
                "processor_id": stt_processor_id,
                "config": kwargs,
            }
        )

        return config

    def add_stt_processor(self, processor_id: str, **kwargs: Any) -> "PerceptionConfig":
        """Add an STT processor configuration.

        Args:
            processor_id: Processor identifier
            **kwargs: Processor configuration options

        Returns:
            PerceptionConfig: Self for method chaining
        """
        self.processors.append(
            {
                "type": "stt",
                "processor_id": processor_id,
                "config": kwargs,
            }
        )
        return self
