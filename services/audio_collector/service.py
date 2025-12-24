"""Audio Collector Service implementation.

Captures audio from the local microphone and publishes raw audio chunks
to MQTT for processing by other services (e.g., transcription).
"""

import os
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import numpy.typing as npt

from ai_assistant.shared.logging import get_logger, LogLevel
from ai_assistant.shared.services import BaseService, ServiceConfig
from ai_assistant.shared.messages import AudioSampleMessage
from ai_assistant.shared.mqtt.topics import Topics

logger = get_logger(__name__)


@dataclass
class AudioCollectorConfig(ServiceConfig):
    """Configuration for Audio Collector Service.

    Attributes:
        sample_rate: Audio sample rate in Hz
        chunk_size: Number of samples per chunk
        channels: Number of audio channels
        device: Audio input device index or name
    """

    sample_rate: int = 16000
    chunk_size: int = 1024
    channels: int = 1
    device: Optional[int] = None

    @classmethod
    def from_env(cls) -> "AudioCollectorConfig":
        """Create configuration from environment variables."""
        return cls(
            service_name="audio-collector",
            machine_id=os.getenv("MACHINE_ID", ""),
            sample_rate=int(os.getenv("AUDIO_SAMPLE_RATE", "16000")),
            chunk_size=int(os.getenv("AUDIO_CHUNK_SIZE", "1024")),
            channels=int(os.getenv("AUDIO_CHANNELS", "1")),
            device=int(os.getenv("AUDIO_DEVICE", "")) if os.getenv("AUDIO_DEVICE") else None,
            log_level=LogLevel.DEBUG
            if os.getenv("DEBUG", "").lower() in ("1", "true", "yes")
            else LogLevel.INFO,
        )


class AudioCollectorService(BaseService):
    """Service that captures audio from microphone and publishes to MQTT.

    This service:
    1. Initializes audio capture from the system microphone
    2. Captures audio in chunks continuously
    3. Publishes each chunk to all/raw/audio/{machine_id}

    The audio is published as raw float32 samples with metadata.
    """

    def __init__(self, config: AudioCollectorConfig) -> None:
        """Initialize the audio collector service.

        Args:
            config: Service configuration
        """
        super().__init__(config)
        self._audio_config = config

        # Audio capture state
        self._stream: Any = None
        self._audio_queue: queue.Queue[npt.NDArray[np.float32]] = queue.Queue()
        self._capture_thread: Optional[threading.Thread] = None
        self._chunk_index = 0

        # MQTT topic for publishing
        self._audio_topic = Topics.RAW_AUDIO.with_param(self.machine_id)

    def _setup(self) -> None:
        """Set up the audio capture stream."""
        try:
            import sounddevice as sd

            # Get device info
            if self._audio_config.device is None:
                device_info = sd.query_devices(kind="input")
            else:
                device_info = sd.query_devices(self._audio_config.device)

            device_name = (
                device_info.get("name", "Unknown") if isinstance(device_info, dict) else "Unknown"
            )

            self._logger.info(f"Initializing audio capture from: {device_name}")
            self._logger.info(
                f"  Sample rate: {self._audio_config.sample_rate}Hz, "
                f"Channels: {self._audio_config.channels}, "
                f"Chunk size: {self._audio_config.chunk_size}"
            )

            # Create input stream
            self._stream = sd.InputStream(
                samplerate=self._audio_config.sample_rate,
                blocksize=self._audio_config.chunk_size,
                device=self._audio_config.device,
                channels=self._audio_config.channels,
                dtype=np.float32,
                callback=self._audio_callback,
            )

            self._logger.info("Audio capture initialized successfully")

        except ImportError:
            raise RuntimeError("sounddevice is required. Install with: pip install sounddevice")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize audio capture: {e}")

    def _cleanup(self) -> None:
        """Clean up audio resources."""
        # Stop capture thread
        if self._capture_thread and self._capture_thread.is_alive():
            self._capture_thread.join(timeout=2.0)

        # Stop audio stream
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception as e:
                self._logger.warning(f"Error closing audio stream: {e}")
            self._stream = None

        # Clear queue
        while not self._audio_queue.empty():
            try:
                self._audio_queue.get_nowait()
            except queue.Empty:
                break

        self._logger.info("Audio capture cleaned up")

    def start(self) -> None:
        """Start the audio collector service."""
        super().start()

        # Start audio stream
        if self._stream is not None:
            self._stream.start()

        # Start capture/publish thread
        self._capture_thread = threading.Thread(
            target=self._capture_loop,
            name="AudioCollector-Capture",
            daemon=True,
        )
        self._capture_thread.start()

        self._logger.info(f"Publishing audio to: {self._audio_topic}")

    def _audio_callback(
        self,
        indata: npt.NDArray[np.float32],
        frames: int,
        time_info: Any,
        status: Any,
    ) -> None:
        """Callback for audio stream - called from audio thread."""
        if status:
            self._logger.warning(f"Audio callback status: {status}")

        # Copy data and add to queue (callback must be fast)
        self._audio_queue.put(indata.copy())

    def _capture_loop(self) -> None:
        """Background thread that publishes audio chunks to MQTT."""
        self._logger.info("Audio capture loop started")

        while self._running:
            try:
                # Get audio from queue with timeout
                samples = self._audio_queue.get(timeout=1.0)

                # Flatten to mono if needed
                if samples.ndim > 1:
                    samples = samples.mean(axis=1)
                samples = samples.flatten().astype(np.float32)

                # Create message
                message = AudioSampleMessage.create(
                    samples=samples,
                    sample_rate=self._audio_config.sample_rate,
                    machine_id=self.machine_id,
                    chunk_index=self._chunk_index,
                    channels=self._audio_config.channels,
                )

                # Publish to MQTT
                payload = message.to_bytes()
                success = self._publish(self._audio_topic, payload)

                if success:
                    self._chunk_index += 1
                    if self._chunk_index % 100 == 0:
                        self._logger.debug(
                            f"Published chunk {self._chunk_index} to {self._audio_topic}"
                        )
                else:
                    self._logger.warning(f"Failed to publish audio chunk {self._chunk_index}")

            except queue.Empty:
                # No audio available, continue
                continue
            except Exception as e:
                self._logger.error(f"Error in capture loop: {e}")
                time.sleep(0.1)

        self._logger.info("Audio capture loop stopped")
