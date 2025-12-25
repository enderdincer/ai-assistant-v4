"""Speech Service implementation.

Listens for TTS requests on MQTT and synthesizes/plays speech.
Also publishes speaker activity events for echo prevention.
"""

import os
import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from ai_assistant.shared.logging import get_logger, LogLevel
from ai_assistant.shared.services import BaseService, ServiceConfig
from ai_assistant.shared.messages import (
    SpeechRequestMessage,
    SpeakerActivityMessage,
    SpeechControlMessage,
    SpeechControlAction,
)
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
        max_queue_size: Maximum number of pending speech requests (0 = unlimited)
    """

    model_path: Path = Path(".downloaded_models/kokoro-v1.0.onnx")
    voices_path: Path = Path(".downloaded_models/voices-v1.0.bin")
    default_voice: str = "af_bella"
    default_speed: float = 1.0
    max_queue_size: int = 20

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
            max_queue_size=int(os.getenv("TTS_MAX_QUEUE_SIZE", "20")),
            log_level=LogLevel.DEBUG
            if os.getenv("DEBUG", "").lower() in ("1", "true", "yes")
            else LogLevel.INFO,
        )


class SpeechService(BaseService):
    """Service that handles text-to-speech requests.

    This service:
    1. Subscribes to all/actions/speech for TTS requests
    2. Queues incoming requests for sequential playback
    3. Synthesizes speech using Kokoro TTS
    4. Plays audio through speakers
    5. Publishes speaker activity events to all/events/speaker-activity
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

        # Speech queue for sequential playback
        self._speech_queue: queue.Queue[SpeechRequestMessage] = queue.Queue(
            maxsize=config.max_queue_size if config.max_queue_size > 0 else 0
        )
        self._playback_thread: Optional[threading.Thread] = None

        # Playback state
        self._is_speaking = False
        self._speaking_lock = threading.Lock()

        # Topics
        self._speech_topic = Topics.ACTION_SPEECH.topic
        self._control_topic = Topics.ACTION_SPEECH_CONTROL.topic
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

        # Start playback thread
        self._playback_thread = threading.Thread(
            target=self._playback_loop,
            name="Speech-Playback",
            daemon=True,
        )
        self._playback_thread.start()
        self._logger.info("Playback thread started")

        # Subscribe to speech requests
        self._subscribe(self._speech_topic, self._on_speech_request)

        # Subscribe to control commands
        self._subscribe(self._control_topic, self._on_control_request)

    def _cleanup(self) -> None:
        """Clean up TTS resources and stop playback thread."""
        # Clear any pending messages
        self.clear_queue()

        # Wait for playback thread to finish (it checks self._running)
        if self._playback_thread and self._playback_thread.is_alive():
            self._logger.info("Waiting for playback thread to finish...")
            self._playback_thread.join(timeout=2.0)

        if self._tts:
            self._tts.cleanup()
            self._tts = None

        self._logger.info("Speech service cleaned up")

    def _on_speech_request(self, topic: str, payload: bytes) -> None:
        """Handle incoming speech request by queueing it.

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

            # Try to queue the message
            try:
                self._speech_queue.put_nowait(message)
                queue_size = self._speech_queue.qsize()
                self._logger.info(
                    f"Queued speech request: '{message.text[:50]}...' "
                    f"(voice: {message.voice}, queue: {queue_size})"
                )
            except queue.Full:
                self._logger.warning(
                    f"Speech queue full ({self._speech_config.max_queue_size}), "
                    f"dropping request: '{message.text[:30]}...'"
                )

        except Exception as e:
            self._logger.error(f"Error handling speech request: {e}")

    def _on_control_request(self, topic: str, payload: bytes) -> None:
        """Handle incoming speech control request.

        Args:
            topic: MQTT topic (should be all/actions/speech-control)
            payload: Message payload
        """
        try:
            message = SpeechControlMessage.from_bytes(payload)

            self._logger.info(f"Received control command: {message.action.value}")

            if message.action == SpeechControlAction.SKIP_CURRENT:
                self.skip_current()
            elif message.action == SpeechControlAction.SKIP_ALL:
                self.skip_all()
            elif message.action == SpeechControlAction.CLEAR_QUEUE:
                cleared = self.clear_queue()
                self._logger.info(f"Cleared {cleared} pending messages")
            elif message.action == SpeechControlAction.PAUSE:
                self._logger.warning("Pause not yet implemented")
            elif message.action == SpeechControlAction.RESUME:
                self._logger.warning("Resume not yet implemented")
            else:
                self._logger.warning(f"Unknown control action: {message.action}")

        except Exception as e:
            self._logger.error(f"Error handling control request: {e}")

    def _playback_loop(self) -> None:
        """Main playback loop that consumes from the speech queue."""
        # Wait for service to be running
        while not self._running:
            time.sleep(0.1)

        self._logger.debug("Playback loop started")

        while self._running:
            try:
                # Block waiting for next message (with timeout to check _running)
                message = self._speech_queue.get(timeout=1.0)
                self._speak(message)
                self._speech_queue.task_done()
            except queue.Empty:
                # No message, just continue to check if still running
                continue
            except Exception as e:
                self._logger.error(f"Error in playback loop: {e}")

        self._logger.debug("Playback loop stopped")

    def _speak(self, message: SpeechRequestMessage) -> None:
        """Synthesize and play speech.

        Args:
            message: Speech request message
        """
        with self._speaking_lock:
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
            speed = message.speed if message.speed > 0 else self._speech_config.default_speed
            self._logger.debug(
                f"Synthesizing with voice '{voice}' at speed {speed}: {message.text[:50]}..."
            )

            self._tts.speak(message.text, voice=voice, speed=speed)

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

    def clear_queue(self) -> int:
        """Clear all pending speech requests from the queue.

        Returns:
            Number of messages cleared
        """
        cleared = 0
        while True:
            try:
                self._speech_queue.get_nowait()
                self._speech_queue.task_done()
                cleared += 1
            except queue.Empty:
                break

        if cleared > 0:
            self._logger.info(f"Cleared {cleared} pending speech requests")
        return cleared

    def get_queue_depth(self) -> int:
        """Get the number of pending speech requests.

        Returns:
            Number of messages in the queue
        """
        return self._speech_queue.qsize()

    def skip_current(self) -> None:
        """Stop current speech and move to next in queue."""
        self.stop_speaking()

    def skip_all(self) -> None:
        """Stop current speech and clear the queue."""
        self.stop_speaking()
        self.clear_queue()
