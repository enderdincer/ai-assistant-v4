"""Audio input source implementation."""

import time
import threading
import queue
from typing import Any, Optional
import numpy as np
import numpy.typing as npt
from ai_assistant.shared.interfaces import IEventBus, EventPriority
from ai_assistant.shared.events import AudioSampleEvent
from ai_assistant.shared.logging import get_logger
from ai_assistant.perception.input_sources.base import BaseInputSource

logger = get_logger(__name__)


class AudioInputSource(BaseInputSource):
    """Audio input source that captures audio from the microphone.

    Uses sounddevice for real-time audio capture.

    Configuration:
        - sample_rate: Sample rate in Hz (default: 16000)
        - chunk_size: Number of samples per chunk (default: 1024)
        - channels: Number of audio channels (default: 1)
        - priority: Event priority (default: NORMAL)
        - device: Audio input device index or name (default: None = system default)
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
        self._device = self._config.get("device", None)

        self._sample_count = 0
        self._chunk_delay = self._chunk_size / self._sample_rate

        # Audio stream and buffer
        self._stream = None
        self._audio_queue: queue.Queue = queue.Queue()

    def _initialize_source(self) -> None:
        """Initialize the audio input source with real microphone."""
        try:
            import sounddevice as sd

            # Get device info
            if self._device is None:
                device_info = sd.query_devices(kind="input")
                device_name = device_info["name"]
            else:
                device_info = sd.query_devices(self._device)
                device_name = device_info["name"]

            logger.info(f"Audio source {self._source_id} initializing with device: {device_name}")
            logger.info(
                f"  Sample rate: {self._sample_rate}Hz, "
                f"Channels: {self._channels}, "
                f"Chunk size: {self._chunk_size}"
            )

            # Create input stream
            self._stream = sd.InputStream(
                samplerate=self._sample_rate,
                blocksize=self._chunk_size,
                device=self._device,
                channels=self._channels,
                dtype=np.float32,
                callback=self._audio_callback,
            )

            logger.info(f"Audio source {self._source_id} initialized successfully")

        except ImportError:
            logger.error("sounddevice not installed. Install with: pip install sounddevice")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize audio source: {e}")
            raise

    def _audio_callback(
        self,
        indata: npt.NDArray[np.float32],
        frames: int,
        time_info: Any,
        status: Any,
    ) -> None:
        """Callback for audio stream - called from audio thread."""
        if status:
            logger.warning(f"Audio callback status: {status}")

        # Copy data to queue (callback must be fast, no blocking)
        self._audio_queue.put(indata.copy())

    def _cleanup_source(self) -> None:
        """Clean up the audio input source."""
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception as e:
                logger.warning(f"Error closing audio stream: {e}")
            self._stream = None

        # Clear the queue
        while not self._audio_queue.empty():
            try:
                self._audio_queue.get_nowait()
            except queue.Empty:
                break

        logger.info(f"Audio source {self._source_id} cleaned up")

    def _capture_and_publish(self) -> None:
        """Capture audio samples and publish them as an event."""
        # Start stream if not already running
        if self._stream is not None and not self._stream.active:
            self._stream.start()

        try:
            # Get audio from queue with timeout
            samples = self._audio_queue.get(timeout=1.0)

            # Flatten if multi-channel to mono
            if samples.ndim > 1:
                samples = samples.mean(axis=1)

            samples = samples.flatten().astype(np.float32)

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

        except queue.Empty:
            # No audio available, just continue
            pass
