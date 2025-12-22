"""Audio processing utilities for sensory processors."""

import os
import urllib.request
from enum import Enum
from typing import Optional

import numpy as np
import numpy.typing as npt


class SpeechState(Enum):
    """State of speech detection."""

    SILENCE = "silence"  # No speech detected
    SPEECH_STARTED = "speech_started"  # Speech just started
    SPEECH_ONGOING = "speech_ongoing"  # Speech is continuing
    SPEECH_ENDED = "speech_ended"  # Speech just ended (silence after speech)


class VoiceActivityDetector:
    """Voice Activity Detector using Silero VAD model via sherpa-onnx.

    This class wraps the sherpa-onnx VAD implementation to provide
    speech segment detection. It tracks the state of speech and
    accumulates audio during speech segments.

    Usage:
        vad = VoiceActivityDetector()
        vad.initialize()

        for audio_chunk in audio_stream:
            state, speech_audio = vad.process(audio_chunk)
            if state == SpeechState.SPEECH_ENDED:
                # speech_audio contains the complete utterance
                transcribe(speech_audio)
    """

    # Default VAD model URL
    VAD_MODEL_URL = (
        "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx"
    )

    def __init__(
        self,
        sample_rate: int = 16000,
        threshold: float = 0.5,
        min_silence_duration: float = 0.3,
        min_speech_duration: float = 0.15,
        max_speech_duration: float = 45.0,
        speech_pad_ms: int = 200,
        model_path: Optional[str] = None,
    ) -> None:
        """Initialize VAD.

        Args:
            sample_rate: Audio sample rate in Hz (must be 16000 for Silero VAD)
            threshold: Speech detection threshold (0.0-1.0, higher = less sensitive)
            min_silence_duration: Minimum silence duration to end speech (seconds)
            min_speech_duration: Minimum speech duration to be considered valid (seconds)
            max_speech_duration: Maximum speech duration before forced segmentation (seconds)
            speech_pad_ms: Padding to add before/after speech segments (milliseconds)
            model_path: Path to VAD model file (downloads if not provided)
        """
        self._sample_rate = sample_rate
        self._threshold = threshold
        self._min_silence_duration = min_silence_duration
        self._min_speech_duration = min_speech_duration
        self._max_speech_duration = max_speech_duration
        self._speech_pad_ms = speech_pad_ms
        self._model_path = model_path

        # State tracking
        self._vad_model = None
        self._is_speaking = False
        self._speech_buffer: list[npt.NDArray[np.float32]] = []
        self._silence_samples = 0
        self._speech_samples = 0

        # Pre-speech buffer for padding (keeps audio BEFORE speech is detected)
        self._pre_speech_samples = int(speech_pad_ms * sample_rate / 1000)
        self._pre_speech_buffer: list[npt.NDArray[np.float32]] = []

        # Buffer for samples that couldn't be processed (less than window_size)
        self._pending_samples: Optional[npt.NDArray[np.float32]] = None

    def initialize(self) -> None:
        """Initialize the VAD model.

        Downloads the model if not present and creates the VAD instance.

        Raises:
            RuntimeError: If model loading fails
        """
        try:
            import sherpa_onnx
        except ImportError as e:
            raise RuntimeError(
                "sherpa-onnx is required for VAD. Install with: pip install sherpa-onnx"
            ) from e

        # Determine model path
        if self._model_path is None:
            cache_dir = os.path.expanduser("~/.cache/sherpa-onnx-models")
            os.makedirs(cache_dir, exist_ok=True)
            self._model_path = os.path.join(cache_dir, "silero_vad.onnx")

        # Download model if not present
        if not os.path.exists(self._model_path):
            urllib.request.urlretrieve(self.VAD_MODEL_URL, self._model_path)

        # Create VAD configuration
        silero_config = sherpa_onnx.SileroVadModelConfig(
            model=self._model_path,
            threshold=self._threshold,
            min_silence_duration=self._min_silence_duration,
            min_speech_duration=self._min_speech_duration,
            max_speech_duration=self._max_speech_duration,
        )

        vad_config = sherpa_onnx.VadModelConfig()
        vad_config.silero_vad = silero_config
        vad_config.sample_rate = self._sample_rate
        vad_config.num_threads = 1
        vad_config.provider = "cpu"

        if not vad_config.validate():
            raise RuntimeError("Invalid VAD configuration")

        # Create VAD model
        self._vad_model = sherpa_onnx.VadModel.create(vad_config)

    def process(
        self, samples: npt.NDArray[np.float32]
    ) -> tuple[SpeechState, Optional[npt.NDArray[np.float32]]]:
        """Process audio samples and detect speech.

        Args:
            samples: Audio samples to process (float32, mono)

        Returns:
            Tuple of:
                - SpeechState indicating current state
                - Complete speech audio if SPEECH_ENDED, None otherwise
        """
        if self._vad_model is None:
            raise RuntimeError("VAD not initialized. Call initialize() first.")

        # VAD requires specific window size (512 samples for 16kHz)
        window_size = self._vad_model.window_size()

        # Prepend any pending samples from previous call
        if self._pending_samples is not None and len(self._pending_samples) > 0:
            samples = np.concatenate([self._pending_samples, samples])
            self._pending_samples = None

        # Initialize state for this call
        state = SpeechState.SILENCE if not self._is_speaking else SpeechState.SPEECH_ONGOING
        speech_audio = None

        # Process samples in window_size chunks
        offset = 0
        while offset + window_size <= len(samples):
            window = samples[offset : offset + window_size]
            is_speech = self._vad_model.is_speech(window)

            if is_speech:
                self._silence_samples = 0
                self._speech_samples += window_size

                if not self._is_speaking:
                    # Speech just started!
                    self._is_speaking = True
                    state = SpeechState.SPEECH_STARTED

                    # Add pre-speech padding from PREVIOUS chunks (not current chunk)
                    # This ensures we capture audio just before speech was detected
                    if self._pre_speech_buffer:
                        self._speech_buffer.extend(self._pre_speech_buffer)
                        self._pre_speech_buffer.clear()

                    # Also add any windows from the CURRENT chunk that came before this one
                    # These are windows we already processed but didn't add to speech buffer
                    # because we weren't speaking yet
                    if offset > 0:
                        pre_windows = samples[0:offset]
                        self._speech_buffer.append(pre_windows.copy())
                        self._speech_samples += len(pre_windows)

                # Add current window to speech buffer
                self._speech_buffer.append(window.copy())

            else:  # Silence
                if self._is_speaking:
                    self._silence_samples += window_size
                    # Still add to buffer (could be mid-sentence pause)
                    self._speech_buffer.append(window.copy())

                    # Check if silence duration exceeded threshold
                    silence_duration = self._silence_samples / self._sample_rate
                    if silence_duration >= self._min_silence_duration:
                        # Speech ended
                        state = SpeechState.SPEECH_ENDED
                        speech_audio = self._get_speech_buffer()
                        self._reset_state()

            # Check max speech duration
            if self._is_speaking:
                speech_duration = self._speech_samples / self._sample_rate
                if speech_duration >= self._max_speech_duration:
                    # Force end speech
                    state = SpeechState.SPEECH_ENDED
                    speech_audio = self._get_speech_buffer()
                    self._reset_state()

            offset += window_size

        # Handle remaining samples (less than window_size)
        remaining_start = offset
        if remaining_start < len(samples):
            remaining = samples[remaining_start:]
            if self._is_speaking:
                # If speaking, add to speech buffer
                self._speech_buffer.append(remaining.copy())
                self._speech_samples += len(remaining)
            else:
                # If not speaking, save for next call to ensure we don't lose samples
                self._pending_samples = remaining.copy()

        # Update pre-speech buffer AFTER processing (for next call)
        # Only do this if we're not currently speaking
        if not self._is_speaking and speech_audio is None:
            # Add the processed portion of this chunk to pre-speech buffer
            processed_samples = samples[:remaining_start] if remaining_start > 0 else samples
            if len(processed_samples) > 0:
                self._pre_speech_buffer.append(processed_samples.copy())
                # Keep only enough for padding
                total_pre_samples = sum(len(s) for s in self._pre_speech_buffer)
                while (
                    total_pre_samples > self._pre_speech_samples
                    and len(self._pre_speech_buffer) > 1
                ):
                    removed = self._pre_speech_buffer.pop(0)
                    total_pre_samples -= len(removed)

        return state, speech_audio

    def _get_speech_buffer(self) -> npt.NDArray[np.float32]:
        """Get accumulated speech audio."""
        if not self._speech_buffer:
            return np.array([], dtype=np.float32)
        return np.concatenate(self._speech_buffer)

    def _reset_state(self) -> None:
        """Reset VAD state for next utterance."""
        self._is_speaking = False
        self._speech_buffer.clear()
        self._silence_samples = 0
        self._speech_samples = 0
        self._pre_speech_buffer.clear()
        self._pending_samples = None
        if self._vad_model:
            self._vad_model.reset()

    def reset(self) -> None:
        """Public method to reset VAD state."""
        self._reset_state()

    @property
    def is_speaking(self) -> bool:
        """Check if currently in speech state."""
        return self._is_speaking

    @property
    def speech_duration(self) -> float:
        """Get current speech segment duration in seconds."""
        return self._speech_samples / self._sample_rate


class AudioBuffer:
    """Buffer for accumulating audio chunks.

    This class manages a circular buffer for audio data, automatically
    discarding old samples when the buffer reaches its maximum size.
    Useful for accumulating audio chunks until enough data is available
    for processing.
    """

    def __init__(self, max_duration: float, sample_rate: int) -> None:
        """Initialize audio buffer.

        Args:
            max_duration: Maximum buffer duration in seconds
            sample_rate: Audio sample rate in Hz

        Raises:
            ValueError: If max_duration or sample_rate are invalid
        """
        if max_duration <= 0:
            raise ValueError("max_duration must be positive")
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive")

        self._max_samples = int(max_duration * sample_rate)
        self._buffer = np.array([], dtype=np.float32)
        self._sample_rate = sample_rate

    def add_chunk(self, samples: npt.NDArray[np.float32]) -> None:
        """Add audio chunk to buffer.

        If adding the chunk exceeds max_duration, older samples are
        automatically discarded to maintain the buffer size.

        Args:
            samples: Audio samples to add (1D array)
        """
        if len(samples) == 0:
            return

        # Concatenate new samples
        self._buffer = np.concatenate([self._buffer, samples])

        # Keep only most recent samples if buffer is too large
        if len(self._buffer) > self._max_samples:
            self._buffer = self._buffer[-self._max_samples :]

    def get_buffer(self) -> npt.NDArray[np.float32]:
        """Get current buffer contents.

        Returns:
            Copy of buffer contents as numpy array
        """
        return self._buffer.copy()

    def clear(self) -> None:
        """Clear buffer, removing all accumulated samples."""
        self._buffer = np.array([], dtype=np.float32)

    def is_full(self) -> bool:
        """Check if buffer has reached maximum capacity.

        Returns:
            True if buffer is full, False otherwise
        """
        return len(self._buffer) >= self._max_samples

    @property
    def duration(self) -> float:
        """Get current buffer duration in seconds.

        Returns:
            Duration of buffered audio in seconds
        """
        return len(self._buffer) / self._sample_rate

    @property
    def sample_count(self) -> int:
        """Get number of samples in buffer.

        Returns:
            Number of samples currently buffered
        """
        return len(self._buffer)

    @property
    def is_empty(self) -> bool:
        """Check if buffer is empty.

        Returns:
            True if buffer contains no samples
        """
        return len(self._buffer) == 0


def resample_audio(
    samples: npt.NDArray[np.float32],
    original_rate: int,
    target_rate: int,
) -> npt.NDArray[np.float32]:
    """Resample audio to target sample rate.

    Uses linear interpolation for resampling. For production use,
    consider using scipy.signal.resample for higher quality.

    Args:
        samples: Input audio samples (1D array)
        original_rate: Original sample rate in Hz
        target_rate: Target sample rate in Hz

    Returns:
        Resampled audio array

    Raises:
        ValueError: If sample rates are invalid or samples is empty
    """
    if original_rate <= 0:
        raise ValueError("original_rate must be positive")
    if target_rate <= 0:
        raise ValueError("target_rate must be positive")
    if len(samples) == 0:
        return np.array([], dtype=np.float32)

    # No resampling needed
    if original_rate == target_rate:
        return samples.astype(np.float32)

    # Calculate target length
    duration = len(samples) / original_rate
    target_length = int(duration * target_rate)

    # Perform linear interpolation
    resampled = np.interp(
        np.linspace(0, len(samples) - 1, target_length),
        np.arange(len(samples)),
        samples,
    )

    return resampled.astype(np.float32)


def convert_to_mono(samples: npt.NDArray[np.float32], channels: int = 2) -> npt.NDArray[np.float32]:
    """Convert multi-channel audio to mono by averaging channels.

    Args:
        samples: Audio samples (1D for mono, 2D for multi-channel)
        channels: Number of channels (default: 2 for stereo)

    Returns:
        Mono audio array
    """
    if len(samples.shape) == 1:
        # Already mono
        return samples

    if len(samples.shape) == 2:
        # Average across channels
        return np.mean(samples, axis=1).astype(np.float32)

    raise ValueError(f"Unsupported audio shape: {samples.shape}")


def normalize_audio(
    samples: npt.NDArray[np.float32], target_level: float = 0.9
) -> npt.NDArray[np.float32]:
    """Normalize audio to target level.

    Scales audio so the maximum absolute value equals target_level.

    Args:
        samples: Input audio samples
        target_level: Target maximum absolute value (default: 0.9)

    Returns:
        Normalized audio array
    """
    if len(samples) == 0:
        return samples

    max_val = np.abs(samples).max()

    if max_val == 0:
        # Silent audio, return as-is
        return samples

    # Scale to target level
    scale_factor = target_level / max_val
    return (samples * scale_factor).astype(np.float32)


def float_to_int16(samples: npt.NDArray[np.float32]) -> npt.NDArray[np.int16]:
    """Convert float32 audio samples to int16.

    Assumes float samples are in range [-1.0, 1.0].

    Args:
        samples: Float audio samples in range [-1.0, 1.0]

    Returns:
        Int16 audio samples
    """
    # Clip to valid range
    samples = np.clip(samples, -1.0, 1.0)

    # Scale to int16 range
    return (samples * 32767).astype(np.int16)


def int16_to_float(samples: npt.NDArray[np.int16]) -> npt.NDArray[np.float32]:
    """Convert int16 audio samples to float32.

    Converts to range [-1.0, 1.0].

    Args:
        samples: Int16 audio samples

    Returns:
        Float32 audio samples in range [-1.0, 1.0]
    """
    return (samples.astype(np.float32) / 32767.0).astype(np.float32)


def compute_energy(samples: npt.NDArray[np.float32]) -> float:
    """Compute energy (RMS) of audio samples.

    Useful for simple voice activity detection.

    Args:
        samples: Audio samples

    Returns:
        RMS energy value
    """
    if len(samples) == 0:
        return 0.0

    return float(np.sqrt(np.mean(samples**2)))


def is_speech(
    samples: npt.NDArray[np.float32],
    threshold: float = 0.01,
) -> bool:
    """Simple speech detection based on audio energy.

    Returns True if the energy of the audio exceeds the threshold,
    indicating potential speech activity.

    Args:
        samples: Audio samples to check
        threshold: Energy threshold for speech detection

    Returns:
        True if audio energy exceeds threshold
    """
    energy = compute_energy(samples)
    return energy >= threshold


def compute_duration(samples: npt.NDArray[np.float32], sample_rate: int) -> float:
    """Calculate audio duration in seconds.

    Args:
        samples: Audio samples
        sample_rate: Sample rate in Hz

    Returns:
        Duration in seconds

    Raises:
        ValueError: If sample_rate is invalid
    """
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")

    return len(samples) / sample_rate


def trim_silence(
    samples: npt.NDArray[np.float32],
    threshold: float = 0.01,
) -> npt.NDArray[np.float32]:
    """Remove leading and trailing silence from audio.

    Args:
        samples: Audio samples
        threshold: Energy threshold to consider as silence

    Returns:
        Audio with silence trimmed
    """
    if len(samples) == 0:
        return samples

    # Find first and last non-silent samples
    abs_samples = np.abs(samples)
    non_silent = abs_samples > threshold

    if not np.any(non_silent):
        # All silence, return empty array
        return np.array([], dtype=np.float32)

    # Find indices of first and last non-silent samples
    indices = np.where(non_silent)[0]
    start_idx = indices[0]
    end_idx = indices[-1] + 1

    return samples[start_idx:end_idx]


def split_on_silence(
    samples: npt.NDArray[np.float32],
    sample_rate: int,
    min_silence_duration: float = 0.5,
    silence_threshold: float = 0.01,
) -> list[npt.NDArray[np.float32]]:
    """Split audio into chunks based on silence.

    Splits audio at points where silence exceeds min_silence_duration.

    Args:
        samples: Audio samples
        sample_rate: Sample rate in Hz
        min_silence_duration: Minimum silence duration to split on (seconds)
        silence_threshold: Energy threshold for silence detection

    Returns:
        List of audio chunks
    """
    if len(samples) == 0:
        return []

    # Calculate window size for silence detection
    window_size = int(min_silence_duration * sample_rate)

    # Find silence regions
    chunks = []
    chunk_start = 0
    i = 0

    while i < len(samples):
        window_end = min(i + window_size, len(samples))
        window = samples[i:window_end]

        # Check if window is silence
        if not is_speech(window, silence_threshold):
            # Found silence, save chunk if it's not empty
            if i > chunk_start:
                chunks.append(samples[chunk_start:i])

            # Skip silence
            i = window_end
            chunk_start = i
        else:
            i += 1

    # Add final chunk if exists
    if chunk_start < len(samples):
        chunks.append(samples[chunk_start:])

    return chunks
