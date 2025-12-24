"""Audio input source implementation."""

import time
import threading
import queue
from typing import Any, Callable, Optional
import numpy as np
import numpy.typing as npt
from ai_assistant.shared.interfaces import IEventBus, EventPriority
from ai_assistant.shared.events import AudioSampleEvent
from ai_assistant.shared.logging import get_logger
from ai_assistant.perception.input_sources.base import BaseInputSource

logger = get_logger(__name__)


class AudioInputSource(BaseInputSource):
    """Audio input source that captures audio from the microphone.

    Uses sounddevice for real-time audio capture with interrupt detection.

    Configuration:
        - sample_rate: Sample rate in Hz (default: 16000)
        - chunk_size: Number of samples per chunk (default: 1024)
        - channels: Number of audio channels (default: 1)
        - priority: Event priority (default: NORMAL)
        - device: Audio input device index or name (default: None = system default)
        - interrupt_threshold: Energy multiplier for interrupt detection (default: 3.0)
        - baseline_window: Number of chunks for baseline energy calculation (default: 50)
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

        # Interrupt detection
        self._interrupt_threshold = self._config.get("interrupt_threshold", 2.0)  # Lowered from 3.0
        self._baseline_window = self._config.get("baseline_window", 50)
        self._tts_warmup_chunks = self._config.get(
            "tts_warmup_chunks", 10
        )  # Chunks to wait before detecting
        self._interrupt_cooldown_chunks = self._config.get(
            "interrupt_cooldown_chunks", 20
        )  # Cooldown after interrupt
        self._interrupt_callback: Optional[Callable[[], None]] = None
        self._tts_playing = False
        self._baseline_energy = 0.0
        self._tts_baseline_energy = 0.0  # Baseline during TTS playback
        self._energy_history: list[float] = []
        self._tts_energy_history: list[float] = []  # Energy history during TTS
        self._tts_chunks_received = 0  # Count chunks since TTS started
        self._interrupt_cooldown = 0  # Cooldown counter
        self._interrupt_lock = threading.Lock()

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

    def set_interrupt_callback(self, callback: Optional[Callable[[], None]]) -> None:
        """Set callback to be called when interrupt is detected.

        Args:
            callback: Function to call when loud audio (interrupt) is detected
        """
        with self._interrupt_lock:
            self._interrupt_callback = callback
            logger.debug(f"Interrupt callback {'set' if callback else 'cleared'}")

    def set_tts_state(self, is_playing: bool) -> None:
        """Notify audio source when TTS is playing.

        When TTS is playing, the audio source monitors for energy spikes
        that indicate user interruption.

        Args:
            is_playing: True if TTS is currently playing
        """
        with self._interrupt_lock:
            was_playing = self._tts_playing
            self._tts_playing = is_playing

            if is_playing and not was_playing:
                # TTS just started - reset TTS-specific tracking
                self._tts_chunks_received = 0
                self._tts_energy_history.clear()
                self._tts_baseline_energy = 0.0
                self._interrupt_cooldown = 0
                logger.debug("TTS started - warming up interrupt detection")
            elif not is_playing and was_playing:
                # TTS stopped
                logger.debug("TTS stopped - interrupt detection disabled")

    def _compute_energy(self, samples: npt.NDArray[np.float32]) -> float:
        """Compute RMS energy of audio samples.

        Args:
            samples: Audio samples

        Returns:
            RMS energy value
        """
        return float(np.sqrt(np.mean(samples**2)))

    def _update_baseline_energy(self, energy: float) -> None:
        """Update baseline energy calculation.

        Maintains a rolling average of recent audio energy levels
        to establish a baseline for interrupt detection.

        Args:
            energy: Current chunk energy
        """
        self._energy_history.append(energy)

        # Keep only recent history
        if len(self._energy_history) > self._baseline_window:
            self._energy_history.pop(0)

        # Calculate baseline (mean of history)
        if self._energy_history:
            self._baseline_energy = sum(self._energy_history) / len(self._energy_history)

    def _check_interrupt(self, samples: npt.NDArray[np.float32]) -> bool:
        """Check if audio samples indicate user interruption.

        Detects sudden energy spikes ABOVE the TTS baseline that suggest
        the user is speaking over the TTS output.

        Args:
            samples: Audio samples to check

        Returns:
            True if interrupt detected
        """
        with self._interrupt_lock:
            # Only check during TTS playback
            if not self._tts_playing or self._interrupt_callback is None:
                return False

            # Check cooldown
            if self._interrupt_cooldown > 0:
                self._interrupt_cooldown -= 1
                return False

            # Compute current energy
            energy = self._compute_energy(samples)

            # Track TTS chunks
            self._tts_chunks_received += 1

            # Build TTS baseline during warmup period
            if self._tts_chunks_received <= self._tts_warmup_chunks:
                self._tts_energy_history.append(energy)
                if len(self._tts_energy_history) >= self._tts_warmup_chunks:
                    self._tts_baseline_energy = sum(self._tts_energy_history) / len(
                        self._tts_energy_history
                    )
                    logger.debug(f"TTS baseline established: {self._tts_baseline_energy:.4f}")
                return False

            # Continue updating TTS baseline with rolling average
            self._tts_energy_history.append(energy)
            if len(self._tts_energy_history) > self._baseline_window:
                self._tts_energy_history.pop(0)

            # Use TTS baseline if available, otherwise fall back to silence baseline
            baseline = (
                self._tts_baseline_energy
                if self._tts_baseline_energy > 0
                else self._baseline_energy
            )

            if baseline <= 0:
                return False

            # Check for energy spike above TTS baseline
            ratio = energy / baseline
            if ratio > self._interrupt_threshold:
                logger.info(
                    f"Interrupt detected! Energy: {energy:.4f}, "
                    f"TTS Baseline: {baseline:.4f}, "
                    f"Ratio: {ratio:.2f}x (threshold: {self._interrupt_threshold}x)"
                )
                # Set cooldown to prevent rapid re-triggers
                self._interrupt_cooldown = self._interrupt_cooldown_chunks
                return True

            return False

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

        # Copy data (callback must be fast, no blocking)
        samples_copy = indata.copy()
        self._audio_queue.put(samples_copy)

        # Check for interrupt (non-blocking)
        try:
            # Flatten if needed for energy calculation
            if samples_copy.ndim > 1:
                samples_flat = samples_copy.mean(axis=1).flatten()
            else:
                samples_flat = samples_copy.flatten()

            # Update baseline when not playing TTS
            if not self._tts_playing:
                energy = self._compute_energy(samples_flat)
                self._update_baseline_energy(energy)

            # Check for interrupt during TTS
            elif self._check_interrupt(samples_flat):
                # Call interrupt callback in separate thread to avoid blocking
                if self._interrupt_callback:
                    threading.Thread(
                        target=self._interrupt_callback, name="Interrupt-Handler", daemon=True
                    ).start()
        except Exception as e:
            logger.warning(f"Error in interrupt detection: {e}")

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
