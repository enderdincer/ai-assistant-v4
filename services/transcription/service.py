"""Transcription Service implementation.

Listens for raw audio on MQTT and produces transcriptions.
Supports multiple audio sources with per-source VAD state.
"""

import os
import threading
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import numpy.typing as npt

from ai_assistant.shared.logging import get_logger, LogLevel
from ai_assistant.shared.services import BaseService, ServiceConfig
from ai_assistant.shared.messages import (
    AudioSampleMessage,
    TranscriptionMessage,
    SpeakerActivityMessage,
)
from ai_assistant.shared.messages.speech import SpeakerState
from ai_assistant.shared.mqtt.topics import Topics, extract_machine_id
from ai_assistant.perception.processors.audio_utils import (
    VoiceActivityDetector,
    SpeechState,
    resample_audio,
)

logger = get_logger(__name__)


@dataclass
class TranscriptionServiceConfig(ServiceConfig):
    """Configuration for Transcription Service.

    Attributes:
        model_name: Sherpa-ONNX STT model identifier
        sample_rate: Target sample rate for STT
        language: Language code
        vad_threshold: Speech detection threshold
        vad_min_silence: Silence duration to end speech
        vad_min_speech: Minimum speech duration
        vad_max_speech: Maximum speech before forced split
    """

    model_name: str = "csukuangfj/sherpa-onnx-nemo-parakeet-tdt-0.6b-v2"
    sample_rate: int = 16000
    language: str = "en"
    vad_threshold: float = 0.5
    vad_min_silence: float = 0.3
    vad_min_speech: float = 0.15
    vad_max_speech: float = 45.0

    @classmethod
    def from_env(cls) -> "TranscriptionServiceConfig":
        """Create configuration from environment variables."""
        return cls(
            service_name="transcription-service",
            model_name=os.getenv(
                "STT_MODEL_NAME",
                "csukuangfj/sherpa-onnx-nemo-parakeet-tdt-0.6b-v2",
            ),
            sample_rate=int(os.getenv("STT_SAMPLE_RATE", "16000")),
            language=os.getenv("STT_LANGUAGE", "en"),
            vad_threshold=float(os.getenv("VAD_THRESHOLD", "0.5")),
            vad_min_silence=float(os.getenv("VAD_MIN_SILENCE", "0.3")),
            vad_min_speech=float(os.getenv("VAD_MIN_SPEECH", "0.15")),
            vad_max_speech=float(os.getenv("VAD_MAX_SPEECH", "45.0")),
            log_level=LogLevel.DEBUG
            if os.getenv("DEBUG", "").lower() in ("1", "true", "yes")
            else LogLevel.INFO,
        )


class SourceState:
    """Per-source state for VAD and audio processing."""

    def __init__(self, source_id: str, config: TranscriptionServiceConfig) -> None:
        """Initialize state for an audio source.

        Args:
            source_id: Source identifier (machine_id)
            config: Transcription config
        """
        self.source_id = source_id
        self.vad = VoiceActivityDetector(
            sample_rate=config.sample_rate,
            threshold=config.vad_threshold,
            min_silence_duration=config.vad_min_silence,
            min_speech_duration=config.vad_min_speech,
            max_speech_duration=config.vad_max_speech,
        )
        self.vad.initialize()
        self.chunk_count = 0
        self.muted = False  # Muted during TTS playback


class TranscriptionService(BaseService):
    """Service that transcribes audio from multiple sources.

    This service:
    1. Subscribes to all/raw/audio/# for all audio sources
    2. Maintains per-source VAD state
    3. Performs STT when speech ends
    4. Publishes transcriptions to all/events/audio-transcribed
    5. Listens to speaker activity to pause during TTS
    """

    def __init__(self, config: TranscriptionServiceConfig) -> None:
        """Initialize the transcription service.

        Args:
            config: Service configuration
        """
        super().__init__(config)
        self._trans_config = config

        # STT model
        self._recognizer: Any = None

        # Per-source state
        self._sources: Dict[str, SourceState] = {}
        self._sources_lock = threading.Lock()

        # Global mute during TTS
        self._global_mute = False

        # Topics
        self._audio_pattern = Topics.RAW_AUDIO.subscription_pattern
        self._transcription_topic = Topics.EVENT_AUDIO_TRANSCRIBED.topic
        self._speaker_activity_topic = Topics.EVENT_SPEAKER_ACTIVITY.topic

    def _setup(self) -> None:
        """Set up STT model and subscribe to audio."""
        # Load STT model
        self._load_stt_model()

        # Subscribe to audio streams
        self._subscribe(self._audio_pattern, self._on_audio)

        # Subscribe to speaker activity for muting
        self._subscribe(self._speaker_activity_topic, self._on_speaker_activity)

    def _load_stt_model(self) -> None:
        """Load the STT model."""
        self._logger.info(f"Loading STT model: {self._trans_config.model_name}")

        try:
            import sherpa_onnx
            import urllib.request
            import tarfile

            # Model download URL and cache directory
            model_cache_dir = os.path.expanduser("~/.cache/sherpa-onnx-models")
            os.makedirs(model_cache_dir, exist_ok=True)

            # Model info
            model_info = {
                "url": "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8.tar.bz2",
                "dir_name": "sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8",
                "encoder": "encoder.int8.onnx",
                "decoder": "decoder.int8.onnx",
                "joiner": "joiner.int8.onnx",
                "tokens": "tokens.txt",
            }

            model_dir = os.path.join(model_cache_dir, model_info["dir_name"])

            # Download and extract if not present
            if not os.path.exists(model_dir):
                self._logger.info(f"Downloading model from {model_info['url']}...")
                archive_path = os.path.join(model_cache_dir, f"{model_info['dir_name']}.tar.bz2")

                urllib.request.urlretrieve(model_info["url"], archive_path)
                self._logger.info("Extracting model...")

                with tarfile.open(archive_path, "r:bz2") as tar:
                    tar.extractall(model_cache_dir)

                os.remove(archive_path)
                self._logger.info(f"Model extracted to {model_dir}")
            else:
                self._logger.info(f"Using cached model from {model_dir}")

            # Build paths
            encoder = os.path.join(model_dir, model_info["encoder"])
            decoder = os.path.join(model_dir, model_info["decoder"])
            joiner = os.path.join(model_dir, model_info["joiner"])
            tokens = os.path.join(model_dir, model_info["tokens"])

            # Create recognizer
            self._recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
                encoder=encoder,
                decoder=decoder,
                joiner=joiner,
                tokens=tokens,
                num_threads=4,
                sample_rate=self._trans_config.sample_rate,
                feature_dim=80,
                decoding_method="greedy_search",
                provider="cpu",
                model_type="nemo_transducer",
            )

            self._logger.info("STT model loaded successfully")

        except ImportError as e:
            raise RuntimeError(
                f"sherpa-onnx is required: {e}. Install with: pip install sherpa-onnx"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load STT model: {e}")

    def _cleanup(self) -> None:
        """Clean up resources."""
        # Clean up per-source VAD
        with self._sources_lock:
            for source in self._sources.values():
                source.vad.reset()
            self._sources.clear()

        # Release model
        if self._recognizer:
            del self._recognizer
            self._recognizer = None

        self._logger.info("Transcription service cleaned up")

    def _get_or_create_source(self, source_id: str) -> SourceState:
        """Get or create state for an audio source.

        Args:
            source_id: Source identifier

        Returns:
            SourceState for this source
        """
        with self._sources_lock:
            if source_id not in self._sources:
                self._logger.info(f"New audio source: {source_id}")
                self._sources[source_id] = SourceState(source_id, self._trans_config)
            return self._sources[source_id]

    def _on_speaker_activity(self, topic: str, payload: bytes) -> None:
        """Handle speaker activity events for muting.

        Args:
            topic: MQTT topic
            payload: Message payload
        """
        try:
            message = SpeakerActivityMessage.from_bytes(payload)

            if message.state == SpeakerState.SPEAKING_STARTED:
                self._global_mute = True
                self._logger.debug("Muting transcription (TTS started)")

                # Reset all VAD states
                with self._sources_lock:
                    for source in self._sources.values():
                        source.vad.reset()

            elif message.state == SpeakerState.SPEAKING_ENDED:
                self._global_mute = False
                self._logger.debug("Unmuting transcription (TTS ended)")

        except Exception as e:
            self._logger.error(f"Error handling speaker activity: {e}")

    def _on_audio(self, topic: str, payload: bytes) -> None:
        """Handle incoming audio samples.

        Args:
            topic: MQTT topic (e.g., all/raw/audio/machine-1)
            payload: Message payload
        """
        # Check global mute
        if self._global_mute:
            return

        try:
            # Parse message
            message = AudioSampleMessage.from_bytes(payload)

            # Get source ID from topic or message
            source_id = extract_machine_id(topic) or message.machine_id
            if not source_id:
                self._logger.warning("Audio message without source ID")
                return

            # Get source state
            source = self._get_or_create_source(source_id)

            # Check source mute
            if source.muted:
                return

            # Get samples
            samples = message.samples
            if not isinstance(samples, np.ndarray):
                samples = np.array(samples, dtype=np.float32)
            elif samples.dtype != np.float32:
                samples = samples.astype(np.float32)

            # Resample if needed
            if message.sample_rate != self._trans_config.sample_rate:
                samples = resample_audio(
                    samples, message.sample_rate, self._trans_config.sample_rate
                )

            # Process through VAD
            state, speech_audio = source.vad.process(samples)
            source.chunk_count += 1

            # Only transcribe when speech ends
            if state == SpeechState.SPEECH_ENDED and speech_audio is not None:
                audio_duration = len(speech_audio) / self._trans_config.sample_rate
                self._logger.info(
                    f"Speech ended from {source_id}, transcribing {audio_duration:.2f}s..."
                )

                # Transcribe
                text = self._transcribe(speech_audio)

                if text and text.strip():
                    # Publish transcription
                    trans_msg = TranscriptionMessage.create(
                        text=text.strip(),
                        audio_source=source_id,
                        language=self._trans_config.language,
                        confidence=1.0,
                        audio_duration=audio_duration,
                        model_name=self._trans_config.model_name,
                    )

                    self._publish(self._transcription_topic, trans_msg.to_bytes())
                    self._logger.info(f"Transcription from {source_id}: '{text.strip()}'")

        except Exception as e:
            self._logger.error(f"Error processing audio: {e}", exc_info=True)

    def _transcribe(self, audio: npt.NDArray[np.float32]) -> Optional[str]:
        """Transcribe audio using the STT model.

        Args:
            audio: Audio samples

        Returns:
            Transcribed text or None
        """
        if self._recognizer is None:
            self._logger.error("Model not loaded")
            return None

        try:
            stream = self._recognizer.create_stream()
            stream.accept_waveform(self._trans_config.sample_rate, audio)
            self._recognizer.decode_stream(stream)
            return stream.result.text

        except Exception as e:
            self._logger.error(f"Transcription failed: {e}")
            return None
