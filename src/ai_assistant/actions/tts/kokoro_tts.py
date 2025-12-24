"""Kokoro TTS implementation using ONNX Runtime.

This module provides a Text-to-Speech interface using the Kokoro 82M model
via ONNX Runtime for efficient inference with audio playback support.
"""

import os
import threading
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np

from ai_assistant.shared.logging import get_logger

logger = get_logger(__name__)


@dataclass
class KokoroTTSConfig:
    """Configuration for Kokoro TTS.

    Attributes:
        model_path: Path to the ONNX model file (kokoro-v1.0.onnx)
        voices_path: Path to the voices binary file (voices-v1.0.bin)
        voice: Voice name to use (e.g., 'af_bella', 'af_sarah', 'am_adam', etc.)
        speed: Speech speed multiplier (1.0 is normal speed)
        lang: Language code for phonemization (e.g., 'en-us', 'ja', etc.)
        sample_rate: Audio sample rate (default 24000 Hz for Kokoro)
        device: Device to run inference on ('cpu' or 'cuda')
        num_threads: Number of threads for CPU inference (None = auto)
    """

    model_path: Union[str, Path]
    voices_path: Union[str, Path]
    voice: str = "af_bella"
    speed: float = 1.0
    lang: str = "en-us"
    sample_rate: int = 24000
    device: str = "cpu"
    num_threads: Optional[int] = None
    extra_options: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        self.model_path = Path(self.model_path)
        self.voices_path = Path(self.voices_path)

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        if not self.voices_path.exists():
            raise FileNotFoundError(f"Voices file not found: {self.voices_path}")

        if self.speed <= 0:
            raise ValueError(f"Speed must be positive, got {self.speed}")

        if self.sample_rate <= 0:
            raise ValueError(f"Sample rate must be positive, got {self.sample_rate}")


class KokoroTTS:
    """Kokoro TTS engine using ONNX Runtime.

    This class provides a simple interface for text-to-speech synthesis
    using the Kokoro 82M model. It handles model loading, voice selection,
    and audio generation.

    Example:
        >>> config = KokoroTTSConfig(
        ...     model_path="kokoro-v1.0.onnx",
        ...     voices_path="voices-v1.0.bin",
        ...     voice="af_bella"
        ... )
        >>> tts = KokoroTTS(config)
        >>> tts.initialize()
        >>> audio = tts.synthesize("Hello, world!")
        >>> tts.save_to_file(audio, "output.wav")
    """

    def __init__(self, config: KokoroTTSConfig) -> None:
        """Initialize Kokoro TTS engine.

        Args:
            config: Configuration for the TTS engine
        """
        self._config = config
        self._kokoro: Optional[Any] = None
        self._initialized = False
        self._logger = get_logger(f"{__name__}.KokoroTTS")

        # Playback control
        self._stop_event = threading.Event()
        self._playback_thread: Optional[threading.Thread] = None

        # Suppress ONNX Runtime warnings
        warnings.filterwarnings("ignore", category=UserWarning, module="onnxruntime")

    def initialize(self) -> None:
        """Initialize the TTS engine by loading the model and voices.

        Raises:
            RuntimeError: If initialization fails
            ImportError: If required dependencies are not installed
        """
        if self._initialized:
            self._logger.warning("TTS engine is already initialized")
            return

        try:
            # Import here to provide better error messages
            try:
                import kokoro_onnx
            except ImportError as e:
                raise ImportError(
                    "kokoro-onnx is required for Kokoro TTS. "
                    "Install it with: pip install kokoro-onnx"
                ) from e

            self._logger.info("Initializing Kokoro TTS engine...")
            self._logger.info(f"Model: {self._config.model_path}")
            self._logger.info(f"Voices: {self._config.voices_path}")

            # Initialize Kokoro with model and voices
            self._kokoro = kokoro_onnx.Kokoro(
                model_path=str(self._config.model_path),
                voices_path=str(self._config.voices_path),
            )

            self._initialized = True
            self._logger.info("Kokoro TTS engine initialized successfully")

        except Exception as e:
            self._logger.error(f"Failed to initialize TTS engine: {e}")
            raise RuntimeError(f"TTS initialization failed: {e}") from e

    def synthesize(self, text: str, voice: Optional[str] = None) -> np.ndarray:
        """Synthesize speech from text.

        Args:
            text: Text to synthesize
            voice: Voice name to use (None = use config default)

        Returns:
            np.ndarray: Audio samples as float32 array

        Raises:
            RuntimeError: If engine is not initialized
            ValueError: If voice is invalid
        """
        if not self._initialized or self._kokoro is None:
            raise RuntimeError("TTS engine is not initialized. Call initialize() first.")

        if not text.strip():
            self._logger.warning("Empty text provided, returning silence")
            return np.array([], dtype=np.float32)

        voice_name = voice or self._config.voice

        try:
            self._logger.debug(f"Synthesizing with voice '{voice_name}': {text[:50]}...")

            # Generate speech using Kokoro
            audio, _ = self._kokoro.create(
                text,
                voice=voice_name,
                speed=self._config.speed,
                lang=self._config.lang,
            )

            self._logger.debug(f"Generated audio: {len(audio)} samples")

            return audio

        except Exception as e:
            self._logger.error(f"Failed to synthesize speech: {e}")
            raise RuntimeError(f"Speech synthesis failed: {e}") from e

    def estimate_duration(self, text: str) -> float:
        """Estimate speech duration in milliseconds.

        Uses a simple heuristic based on character count and speech speed.
        Average speaking rate is ~150 words per minute, or ~12.5 characters per second.

        Args:
            text: Text to estimate duration for

        Returns:
            Estimated duration in milliseconds
        """
        if not text.strip():
            return 0.0

        # Average characters per second at normal speed
        chars_per_second = 12.5

        # Adjust for speed setting
        adjusted_chars_per_second = chars_per_second * self._config.speed

        # Calculate duration
        duration_seconds = len(text) / adjusted_chars_per_second

        # Add small padding for pauses (10%)
        duration_seconds *= 1.1

        return duration_seconds * 1000  # Convert to milliseconds

    def stop(self) -> None:
        """Stop current playback immediately.

        This method can be called to interrupt ongoing speech synthesis.
        """
        self._stop_event.set()
        self._logger.debug("Stop signal sent to TTS")

        try:
            import sounddevice as sd

            sd.stop()
            self._logger.debug("Audio playback stopped")
        except ImportError:
            pass
        except Exception as e:
            self._logger.warning(f"Error stopping audio: {e}")

    def play(self, audio: np.ndarray, blocking: bool = True) -> None:
        """Play audio through the default audio device.

        Args:
            audio: Audio samples to play
            blocking: If True, wait for playback to complete (default: True)

        Raises:
            ImportError: If sounddevice is not installed
            RuntimeError: If playback fails
        """
        try:
            import sounddevice as sd
        except ImportError as e:
            raise ImportError(
                "sounddevice is required for audio playback. "
                "Install it with: pip install sounddevice"
            ) from e

        try:
            # Clear stop event before starting
            self._stop_event.clear()

            self._logger.debug(
                f"Playing audio: {len(audio)} samples at {self._config.sample_rate}Hz"
            )

            # Ensure audio is float32 and contiguous
            if audio.dtype != np.float32:
                audio = audio.astype(np.float32)
            audio = np.ascontiguousarray(audio)

            # Use a larger blocksize to prevent buffer underruns and popping
            # 2048 samples at 24kHz = ~85ms blocks, which should be smooth
            blocksize = 2048

            # Create an output stream with explicit settings
            with sd.OutputStream(
                samplerate=self._config.sample_rate,
                channels=1,
                dtype="float32",
                blocksize=blocksize,
                latency="high",
            ) as stream:
                # Write audio in chunks
                offset = 0
                while offset < len(audio):
                    if self._stop_event.is_set():
                        self._logger.debug("Playback interrupted")
                        break

                    # Get next chunk
                    chunk = audio[offset : offset + blocksize]

                    # Pad last chunk if needed
                    if len(chunk) < blocksize:
                        chunk = np.pad(chunk, (0, blocksize - len(chunk)), mode="constant")

                    # Write to stream (blocking write)
                    stream.write(chunk)
                    offset += blocksize

            self._logger.debug("Audio playback completed")

        except Exception as e:
            self._logger.error(f"Failed to play audio: {e}")
            raise RuntimeError(f"Audio playback failed: {e}") from e

    def speak(
        self, text: str, voice: Optional[str] = None, save_to: Optional[Union[str, Path]] = None
    ) -> np.ndarray:
        """Synthesize and play speech from text (convenience method).

        This is a high-level method that combines synthesize() and play().

        Args:
            text: Text to speak
            voice: Voice name to use (None = use config default)
            save_to: Optional path to save audio file

        Returns:
            np.ndarray: Audio samples

        Raises:
            RuntimeError: If synthesis or playback fails
        """
        # Synthesize
        audio = self.synthesize(text, voice=voice)

        # Save if requested
        if save_to:
            self.save_to_file(audio, save_to)

        # Play
        self.play(audio, blocking=True)

        return audio

    def save_to_file(
        self, audio: np.ndarray, output_path: Union[str, Path], format: str = "wav"
    ) -> None:
        """Save audio to a file.

        Args:
            audio: Audio samples to save
            output_path: Path to output file
            format: Audio format ('wav', 'flac', etc.)

        Raises:
            ImportError: If soundfile is not installed
            RuntimeError: If saving fails
        """
        try:
            import soundfile as sf
        except ImportError as e:
            raise ImportError(
                "soundfile is required to save audio files. Install it with: pip install soundfile"
            ) from e

        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            self._logger.info(f"Saving audio to {output_path}")

            sf.write(
                str(output_path), audio, self._config.sample_rate, format=format, subtype="PCM_16"
            )

            self._logger.info(f"Audio saved successfully: {output_path}")

        except Exception as e:
            self._logger.error(f"Failed to save audio: {e}")
            raise RuntimeError(f"Failed to save audio: {e}") from e

    def get_available_voices(self) -> list[str]:
        """Get list of available voices.

        Returns:
            list[str]: List of voice names

        Raises:
            RuntimeError: If engine is not initialized
        """
        if not self._initialized or self._kokoro is None:
            raise RuntimeError("TTS engine is not initialized. Call initialize() first.")

        # Get voices from the kokoro instance
        return list(self._kokoro.get_voices())

    def cleanup(self) -> None:
        """Clean up resources."""
        if self._initialized:
            self._logger.info("Cleaning up TTS engine")
            self.stop()  # Stop any ongoing playback
            self._kokoro = None
            self._initialized = False

    @property
    def is_initialized(self) -> bool:
        """Check if the engine is initialized."""
        return self._initialized

    @property
    def config(self) -> KokoroTTSConfig:
        """Get the current configuration."""
        return self._config

    def __enter__(self) -> "KokoroTTS":
        """Context manager entry."""
        self.initialize()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit."""
        self.cleanup()
