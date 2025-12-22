"""Speech-to-Text processor using NVIDIA Parakeet TDT with Sherpa-ONNX."""

import numpy as np
import numpy.typing as npt
from typing import Any, Optional

from ai_assistant.perception.processors.base import BaseProcessor
from ai_assistant.perception.processors.audio_utils import (
    VoiceActivityDetector,
    SpeechState,
    resample_audio,
)
from ai_assistant.shared.interfaces import IEventBus, IEvent
from ai_assistant.shared.events import AudioTranscriptionEvent
from ai_assistant.shared.logging import get_logger

logger = get_logger(__name__)


class STTProcessor(BaseProcessor):
    """Speech-to-Text processor using NVIDIA Parakeet TDT with Sherpa-ONNX.

    This processor subscribes to AudioSampleEvents, uses Voice Activity Detection
    (VAD) to detect speech segments, and performs speech-to-text transcription
    when a complete utterance is detected. Transcription results are published
    as AudioTranscriptionEvents.

    The VAD-based approach ensures natural sentence boundaries by:
    - Detecting when speech starts
    - Buffering audio during speech
    - Triggering transcription when speech ends (silence detected)

    Configuration options:
        model_name: Sherpa-ONNX model identifier
        sample_rate: Expected sample rate in Hz (default: 16000)
        min_confidence: Minimum confidence threshold (default: 0.0)
        language: Language code (default: "en")

        VAD options:
        vad_threshold: Speech detection threshold 0-1 (default: 0.5)
        vad_min_silence_duration: Silence duration to end speech in seconds (default: 0.3)
        vad_min_speech_duration: Minimum speech duration in seconds (default: 0.15)
        vad_max_speech_duration: Maximum speech duration before forced split (default: 45.0)
    """

    def __init__(
        self,
        processor_id: str,
        event_bus: IEventBus,
        config: Optional[dict[str, Any]] = None,
    ) -> None:
        """Initialize the STT processor.

        Args:
            processor_id: Unique identifier for this processor
            event_bus: Event bus for subscribing and publishing events
            config: Optional configuration dictionary
        """
        config = config or {}

        super().__init__(
            processor_id=processor_id,
            processor_type="stt",
            event_bus=event_bus,
            input_event_types=["audio.sample"],
            output_event_types=["audio.transcription"],
            config=config,
        )

        # STT Configuration
        self._model_name = config.get(
            "model_name", "csukuangfj/sherpa-onnx-nemo-parakeet-tdt-0.6b-v2"
        )
        self._target_sample_rate = config.get("sample_rate", 16000)
        self._min_confidence = config.get("min_confidence", 0.0)
        self._language = config.get("language", "en")

        # VAD Configuration
        self._vad_threshold = config.get("vad_threshold", 0.5)
        self._vad_min_silence = config.get("vad_min_silence_duration", 0.3)
        self._vad_min_speech = config.get("vad_min_speech_duration", 0.15)
        self._vad_max_speech = config.get("vad_max_speech_duration", 45.0)

        # Components (loaded during initialization)
        self._recognizer = None
        self._vad: Optional[VoiceActivityDetector] = None

    def _validate_config(self) -> None:
        """Validate processor configuration.

        Raises:
            ValueError: If configuration is invalid
        """
        if self._target_sample_rate <= 0:
            raise ValueError("sample_rate must be positive")

        if not (0.0 <= self._min_confidence <= 1.0):
            raise ValueError("min_confidence must be between 0.0 and 1.0")

        if not (0.0 <= self._vad_threshold <= 1.0):
            raise ValueError("vad_threshold must be between 0.0 and 1.0")

        if self._vad_min_silence <= 0:
            raise ValueError("vad_min_silence_duration must be positive")

        if self._vad_max_speech <= 0:
            raise ValueError("vad_max_speech_duration must be positive")

    def _initialize_processor(self) -> None:
        """Initialize STT model and VAD.

        Raises:
            RuntimeError: If model loading fails
        """
        # Initialize VAD
        self._logger.info("Initializing Voice Activity Detection...")
        self._vad = VoiceActivityDetector(
            sample_rate=self._target_sample_rate,
            threshold=self._vad_threshold,
            min_silence_duration=self._vad_min_silence,
            min_speech_duration=self._vad_min_speech,
            max_speech_duration=self._vad_max_speech,
        )
        self._vad.initialize()
        self._logger.info("VAD initialized successfully")

        # Initialize STT model
        self._logger.info(f"Loading STT model: {self._model_name}")

        try:
            import sherpa_onnx
            import os
            import urllib.request
            import tarfile

            # Model download URL and cache directory
            model_cache_dir = os.path.expanduser("~/.cache/sherpa-onnx-models")
            os.makedirs(model_cache_dir, exist_ok=True)

            # Map model names to download info
            model_info = {
                "parakeet-tdt-0.6b-v2": {
                    "url": "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8.tar.bz2",
                    "dir_name": "sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8",
                    "encoder": "encoder.int8.onnx",
                    "decoder": "decoder.int8.onnx",
                    "joiner": "joiner.int8.onnx",
                    "tokens": "tokens.txt",
                },
                "csukuangfj/sherpa-onnx-nemo-parakeet-tdt-0.6b-v2": {
                    "url": "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8.tar.bz2",
                    "dir_name": "sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8",
                    "encoder": "encoder.int8.onnx",
                    "decoder": "decoder.int8.onnx",
                    "joiner": "joiner.int8.onnx",
                    "tokens": "tokens.txt",
                },
            }

            if self._model_name not in model_info:
                raise ValueError(
                    f"Unsupported model: {self._model_name}. Supported: {list(model_info.keys())}"
                )

            info = model_info[self._model_name]
            model_dir = os.path.join(model_cache_dir, info["dir_name"])

            # Download and extract if not present
            if not os.path.exists(model_dir):
                self._logger.info(f"Downloading model from {info['url']}...")
                archive_path = os.path.join(model_cache_dir, f"{info['dir_name']}.tar.bz2")

                urllib.request.urlretrieve(info["url"], archive_path)
                self._logger.info("Extracting model...")

                with tarfile.open(archive_path, "r:bz2") as tar:
                    tar.extractall(model_cache_dir)

                os.remove(archive_path)
                self._logger.info(f"Model extracted to {model_dir}")
            else:
                self._logger.info(f"Using cached model from {model_dir}")

            # Build full paths
            encoder = os.path.join(model_dir, info["encoder"])
            decoder = os.path.join(model_dir, info["decoder"])
            joiner = os.path.join(model_dir, info["joiner"])
            tokens = os.path.join(model_dir, info["tokens"])

            self._logger.info("Creating recognizer...")

            # Create offline recognizer
            self._recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
                encoder=encoder,
                decoder=decoder,
                joiner=joiner,
                tokens=tokens,
                num_threads=4,
                sample_rate=self._target_sample_rate,
                feature_dim=80,
                decoding_method="greedy_search",
                provider="cpu",
                model_type="nemo_transducer",
            )

            self._logger.info("STT model loaded successfully")

        except ImportError as e:
            error_msg = f"Failed to import sherpa-onnx: {e}. Install with: pip install sherpa-onnx"
            self._logger.error(error_msg)
            raise RuntimeError(error_msg) from e

        except Exception as e:
            error_msg = f"Failed to load STT model '{self._model_name}': {e}"
            self._logger.error(error_msg)
            raise RuntimeError(error_msg) from e

    def _cleanup_processor(self) -> None:
        """Clean up model resources."""
        if self._recognizer is not None:
            del self._recognizer
            self._recognizer = None

        if self._vad is not None:
            self._vad.reset()
            self._vad = None

        self._logger.info("STT processor cleaned up")

    def _process_event(self, event: IEvent) -> list[IEvent]:
        """Process audio sample event using VAD.

        Uses Voice Activity Detection to buffer audio during speech
        and trigger transcription when speech ends.

        Args:
            event: AudioSampleEvent to process

        Returns:
            List containing AudioTranscriptionEvent if speech ended and was
            transcribed successfully, empty list otherwise
        """
        if event.event_type != "audio.sample":
            return []

        if self._vad is None:
            self._logger.error("VAD not initialized")
            return []

        # Extract audio data from event
        audio_data = event.data
        samples = audio_data["samples"]
        sample_rate = audio_data["sample_rate"]

        # Ensure samples are float32
        if not isinstance(samples, np.ndarray):
            samples = np.array(samples, dtype=np.float32)
        elif samples.dtype != np.float32:
            samples = samples.astype(np.float32)

        # Resample if necessary
        if sample_rate != self._target_sample_rate:
            try:
                samples = resample_audio(samples, sample_rate, self._target_sample_rate)
            except Exception as e:
                self._logger.error(f"Failed to resample audio: {e}")
                return []

        # Process audio through VAD
        state, speech_audio = self._vad.process(samples)

        # Log state changes
        if state == SpeechState.SPEECH_STARTED:
            self._logger.debug("Speech started")
        elif state == SpeechState.SPEECH_ONGOING:
            self._logger.debug(f"Speech ongoing: {self._vad.speech_duration:.2f}s")

        # Only transcribe when speech ends
        if state != SpeechState.SPEECH_ENDED or speech_audio is None:
            return []

        audio_duration = len(speech_audio) / self._target_sample_rate
        self._logger.info(f"Speech ended, transcribing {audio_duration:.2f}s of audio...")

        # Perform transcription
        try:
            transcription_result = self._transcribe(speech_audio)

            if transcription_result is None:
                return []

            text = transcription_result["text"].strip()
            confidence = transcription_result["confidence"]

            # Filter by minimum confidence
            if confidence < self._min_confidence:
                self._logger.debug(
                    f"Transcription below confidence threshold: "
                    f"{confidence:.2f} < {self._min_confidence:.2f}"
                )
                return []

            # Skip empty transcriptions
            if not text:
                self._logger.debug("Empty transcription, skipping")
                return []

            # Create transcription event
            transcription_event = AudioTranscriptionEvent.create(
                source=self._processor_id,
                text=text,
                language=transcription_result["language"],
                confidence=confidence,
                audio_duration=audio_duration,
                model_name=self._model_name,
                source_event_id=str(id(event)),
            )

            self._logger.info(f"Transcribed: '{text}' (duration: {audio_duration:.2f}s)")

            return [transcription_event]

        except Exception as e:
            self._logger.error(f"Transcription failed: {e}", exc_info=True)
            return []

    def _transcribe(self, audio: npt.NDArray[np.float32]) -> Optional[dict[str, Any]]:
        """Transcribe audio using the STT model.

        Args:
            audio: Audio samples to transcribe

        Returns:
            Dictionary containing:
                - text: Transcribed text
                - language: Detected or configured language
                - confidence: Confidence score
            Returns None if transcription fails
        """
        if self._recognizer is None:
            self._logger.error("Model not loaded, cannot transcribe")
            return None

        try:
            # Create an offline stream for this audio
            stream = self._recognizer.create_stream()

            # Feed audio to the stream
            stream.accept_waveform(self._target_sample_rate, audio)

            # Decode the stream
            self._recognizer.decode_stream(stream)

            # Get result
            result = stream.result
            text = result.text

            return {
                "text": text,
                "language": self._language,
                "confidence": 1.0,
            }

        except Exception as e:
            self._logger.error(f"Model inference failed: {e}", exc_info=True)
            return None
