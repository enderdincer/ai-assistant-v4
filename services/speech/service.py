"""Speech Service implementation.

Listens for TTS requests on MQTT and synthesizes/plays speech.
Also publishes speaker activity events for echo prevention.
"""

import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from ai_assistant.shared.logging import get_logger, LogLevel
from ai_assistant.shared.services import BaseService, ServiceConfig
from ai_assistant.shared.messages import SpeechRequestMessage, SpeakerActivityMessage
from ai_assistant.shared.mqtt.topics import Topics
from ai_assistant.actions.tts import KokoroTTS, KokoroTTSConfig

logger = get_logger(__name__)


@dataclass
class SpeechServiceConfig(ServiceConfig):
    """Configuration for Speech Service.

    Attributes:
        model_path: Path to Kokoro TTS model
        voices_path: Path to Kokoro voices file
        default_voice: Default voice to use
        default_speed: Default speech speed
    """

    model_path: Path = Path(".downloaded_models/kokoro-v1.0.onnx")
    voices_path: Path = Path(".downloaded_models/voices-v1.0.bin")
    default_voice: str = "af_bella"
    default_speed: float = 1.0

    @classmethod
    def from_env(cls) -> "SpeechServiceConfig":
        """Create configuration from environment variables."""
        return cls(
            service_name="speech-service",
            model_path=Path(
                os.getenv(
                    "TTS_MODEL_PATH",
                    ".downloaded_models/kokoro-v1.0.onnx",
                )
            ),
            voices_path=Path(
                os.getenv(
                    "TTS_VOICES_PATH",
                    ".downloaded_models/voices-v1.0.bin",
                )
            ),
            default_voice=os.getenv("TTS_VOICE", "af_bella"),
            default_speed=float(os.getenv("TTS_SPEED", "1.0")),
            log_level=LogLevel.DEBUG
            if os.getenv("DEBUG", "").lower() in ("1", "true", "yes")
            else LogLevel.INFO,
        )


class SpeechService(BaseService):
    """Service that handles text-to-speech requests.

    This service:
    1. Subscribes to all/actions/speech for TTS requests
    2. Synthesizes speech using Kokoro TTS
    3. Plays audio through speakers
    4. Publishes speaker activity events to all/events/speaker-activity
    """

    def __init__(self, config: SpeechServiceConfig) -> None:
        """Initialize the speech service.

        Args:
            config: Service configuration
        """
        super().__init__(config)
        self._speech_config = config

        # TTS engine
        self._tts: Optional[KokoroTTS] = None

        # Playback state
        self._is_speaking = False
        self._speaking_lock = threading.Lock()

        # Topics
        self._speech_topic = Topics.ACTION_SPEECH.topic
        self._activity_topic = Topics.EVENT_SPEAKER_ACTIVITY.topic

    def _setup(self) -> None:
        """Set up TTS engine and subscribe to speech requests."""
        # Initialize TTS
        try:
            tts_config = KokoroTTSConfig(
                model_path=self._speech_config.model_path,
                voices_path=self._speech_config.voices_path,
                voice=self._speech_config.default_voice,
                speed=self._speech_config.default_speed,
            )
            self._tts = KokoroTTS(tts_config)
            self._tts.initialize()

            self._logger.info(f"TTS initialized with voice: {self._speech_config.default_voice}")
            self._logger.info(f"Available voices: {self._tts.get_available_voices()}")

        except FileNotFoundError as e:
            raise RuntimeError(f"TTS model files not found: {e}")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize TTS: {e}")

        # Subscribe to speech requests
        self._subscribe(self._speech_topic, self._on_speech_request)

    def _cleanup(self) -> None:
        """Clean up TTS resources."""
        if self._tts:
            self._tts.cleanup()
            self._tts = None

        self._logger.info("Speech service cleaned up")

    def _on_speech_request(self, topic: str, payload: bytes) -> None:
        """Handle incoming speech request.

        Args:
            topic: MQTT topic (should be all/actions/speech)
            payload: Message payload
        """
        try:
            # Parse message
            message = SpeechRequestMessage.from_bytes(payload)

            if not message.text.strip():
                self._logger.warning("Received empty speech request")
                return

            self._logger.info(f"Speech request: '{message.text[:50]}...' (voice: {message.voice})")

            # Process in background thread to not block MQTT
            thread = threading.Thread(
                target=self._speak,
                args=(message,),
                name="Speech-Playback",
                daemon=True,
            )
            thread.start()

        except Exception as e:
            self._logger.error(f"Error handling speech request: {e}")

    def _speak(self, message: SpeechRequestMessage) -> None:
        """Synthesize and play speech.

        Args:
            message: Speech request message
        """
        with self._speaking_lock:
            if self._is_speaking:
                self._logger.warning("Already speaking, queueing request")
                # TODO: Implement queue for multiple requests
                return

            self._is_speaking = True

        try:
            if self._tts is None:
                self._logger.error("TTS not initialized")
                return

            # Estimate duration
            duration_ms = self._tts.estimate_duration(message.text)

            # Publish speaking started
            started_msg = SpeakerActivityMessage.create_started(
                text=message.text,
                duration_ms=duration_ms,
                request_id=message.request_id,
            )
            self._publish(self._activity_topic, started_msg.to_bytes())

            start_time = time.time()

            # Synthesize and play
            voice = message.voice or self._speech_config.default_voice
            self._logger.debug(f"Synthesizing with voice '{voice}': {message.text[:50]}...")

            self._tts.speak(message.text, voice=voice)

            actual_duration_ms = (time.time() - start_time) * 1000

            # Publish speaking ended
            ended_msg = SpeakerActivityMessage.create_ended(
                text=message.text,
                duration_ms=actual_duration_ms,
                request_id=message.request_id,
            )
            self._publish(self._activity_topic, ended_msg.to_bytes())

            self._logger.debug(f"Speech completed in {actual_duration_ms:.0f}ms")

        except Exception as e:
            self._logger.error(f"Error during speech synthesis: {e}")

            # Publish ended even on error
            ended_msg = SpeakerActivityMessage.create_ended(
                text=message.text,
                duration_ms=0,
                request_id=message.request_id,
            )
            self._publish(self._activity_topic, ended_msg.to_bytes())

        finally:
            with self._speaking_lock:
                self._is_speaking = False

    def stop_speaking(self) -> None:
        """Stop current speech playback."""
        if self._tts and self._is_speaking:
            self._tts.stop()
            self._logger.info("Speech playback stopped")
