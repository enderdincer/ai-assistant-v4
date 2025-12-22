"""Audio input source implementation."""

import time
from typing import Any, Optional
import numpy as np
import numpy.typing as npt
from ai_assistant.shared.interfaces import IEventBus, EventPriority
from ai_assistant.shared.events import AudioSampleEvent
from ai_assistant.shared.logging import get_logger
from ai_assistant.perception.input_sources.base import BaseInputSource

logger = get_logger(__name__)


class AudioInputSource(BaseInputSource):
    """Audio input source that captures audio samples.

    Note: This is a stub implementation that generates dummy audio.
    In a real implementation, you would use pyaudio or sounddevice.

    Configuration:
        - sample_rate: Sample rate in Hz (default: 16000)
        - chunk_size: Number of samples per chunk (default: 1024)
        - channels: Number of audio channels (default: 1)
        - priority: Event priority (default: NORMAL)
    """

    def __init__(
        self,
        source_id: str,
        event_bus: IEventBus,
        config: Optional[dict[str, Any]] = None,
    ) -> None:
        """Initialize audio input source.

        Args:
            source_id: Unique identifier for this audio source
            event_bus: Event bus for publishing audio events
            config: Audio configuration
        """
        super().__init__(source_id, "audio", event_bus, config)

        self._sample_rate = self._config.get("sample_rate", 16000)
        self._chunk_size = self._config.get("chunk_size", 1024)
        self._channels = self._config.get("channels", 1)
        self._priority = self._config.get("priority", EventPriority.NORMAL)

        self._sample_count = 0
        self._chunk_delay = self._chunk_size / self._sample_rate

    def _initialize_source(self) -> None:
        """Initialize the audio input source."""
        logger.info(
            f"Audio source {self._source_id} initialized: "
            f"{self._sample_rate}Hz, {self._channels}ch, chunk_size={self._chunk_size}"
        )
        logger.warning(
            "This is a stub implementation generating dummy audio. "
            "Replace with real audio capture for production."
        )

    def _cleanup_source(self) -> None:
        """Clean up the audio input source."""
        logger.info(f"Audio source {self._source_id} cleaned up")

    def _capture_and_publish(self) -> None:
        """Capture audio samples and publish them as an event."""
        start_time = time.time()

        # Generate dummy audio (silence)
        # In a real implementation, this would capture from a microphone
        samples = np.zeros(self._chunk_size, dtype=np.float32)

        self._sample_count += 1

        # Create and publish event
        event = AudioSampleEvent.create(
            source=self._source_id,
            samples=samples,
            sample_rate=self._sample_rate,
            priority=self._priority,
        )

        try:
            self._event_bus.publish(event)
            logger.debug(
                f"Published audio chunk {self._sample_count} from {self._source_id} "
                f"({len(samples)} samples)"
            )
        except Exception as e:
            logger.error(f"Failed to publish audio chunk: {e}")

        # Maintain timing
        elapsed = time.time() - start_time
        sleep_time = max(0, self._chunk_delay - elapsed)
        if sleep_time > 0:
            time.sleep(sleep_time)
