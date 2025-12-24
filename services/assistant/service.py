"""Assistant Service implementation.

The brain of the AI assistant that:
1. Receives transcriptions and text input
2. Generates responses using an LLM
3. Publishes responses and speech requests
"""

import os
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from ai_assistant.shared.logging import get_logger, LogLevel
from ai_assistant.shared.services import BaseService, ServiceConfig
from ai_assistant.shared.messages import (
    TranscriptionMessage,
    TextInputMessage,
    AssistantResponseMessage,
    SpeechRequestMessage,
    SpeakerActivityMessage,
)
from ai_assistant.shared.messages.speech import SpeakerState
from ai_assistant.shared.mqtt.topics import Topics
from ai_assistant.shared.ollama import OllamaClient, OllamaError

logger = get_logger(__name__)


def _strip_emojis(text: str) -> str:
    """Remove emojis from text."""
    emoji_pattern = re.compile(
        "["
        "\U0001f600-\U0001f64f"
        "\U0001f300-\U0001f5ff"
        "\U0001f680-\U0001f6ff"
        "\U0001f1e0-\U0001f1ff"
        "\U00002702-\U000027b0"
        "\U000024c2-\U0001f251"
        "\U0001f900-\U0001f9ff"
        "\U0001fa00-\U0001fa6f"
        "\U0001fa70-\U0001faff"
        "\U00002600-\U000026ff"
        "\U00002300-\U000023ff"
        "\U0001f700-\U0001f77f"
        "]+",
        flags=re.UNICODE,
    )
    text = emoji_pattern.sub("", text)
    text = re.sub(r"  +", " ", text)
    return text.strip()


@dataclass
class AssistantServiceConfig(ServiceConfig):
    """Configuration for Assistant Service.

    Attributes:
        ollama_host: Ollama API host
        ollama_model: LLM model name
        ollama_timeout: Request timeout in seconds
        system_prompt: System prompt for the LLM
        tts_voice: Voice for TTS responses
        tts_speed: Speed for TTS responses
        max_history: Maximum conversation history turns
        enable_tts: Whether to send TTS requests
    """

    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "qwen3:1.7b"
    ollama_timeout: int = 120
    system_prompt: str = (
        "You are a helpful, friendly voice assistant. "
        "Keep your responses concise unless the user asks for detailed information. "
        "Never use emojis."
    )
    tts_voice: str = "af_bella"
    tts_speed: float = 1.0
    max_history: int = 10
    enable_tts: bool = True

    @classmethod
    def from_env(cls) -> "AssistantServiceConfig":
        """Create configuration from environment variables."""
        return cls(
            service_name="assistant-service",
            ollama_host=os.getenv("OLLAMA_HOST", "http://localhost:11434"),
            ollama_model=os.getenv("OLLAMA_MODEL", "qwen3:1.7b"),
            ollama_timeout=int(os.getenv("OLLAMA_TIMEOUT", "120")),
            system_prompt=os.getenv(
                "SYSTEM_PROMPT",
                "You are a helpful, friendly voice assistant. "
                "Keep your responses concise unless the user asks for detailed information. "
                "Never use emojis.",
            ),
            tts_voice=os.getenv("TTS_VOICE", "af_bella"),
            tts_speed=float(os.getenv("TTS_SPEED", "1.0")),
            max_history=int(os.getenv("MAX_HISTORY", "10")),
            enable_tts=os.getenv("ENABLE_TTS", "true").lower() in ("1", "true", "yes"),
            log_level=LogLevel.DEBUG
            if os.getenv("DEBUG", "").lower() in ("1", "true", "yes")
            else LogLevel.INFO,
        )


class ConversationHistory:
    """Thread-safe conversation history."""

    def __init__(self, max_turns: int = 10) -> None:
        """Initialize conversation history.

        Args:
            max_turns: Maximum number of conversation turns to keep
        """
        self._messages: list[dict[str, str]] = []
        self._max_turns = max_turns
        self._lock = threading.Lock()

    def add_user_message(self, text: str) -> None:
        """Add a user message."""
        with self._lock:
            self._messages.append({"role": "user", "content": text})
            self._trim()

    def add_assistant_message(self, text: str) -> None:
        """Add an assistant message."""
        with self._lock:
            self._messages.append({"role": "assistant", "content": text})
            self._trim()

    def get_messages(self) -> list[dict[str, str]]:
        """Get all messages."""
        with self._lock:
            return self._messages.copy()

    def clear(self) -> None:
        """Clear history."""
        with self._lock:
            self._messages.clear()

    def _trim(self) -> None:
        """Trim to max turns."""
        max_messages = self._max_turns * 2
        if len(self._messages) > max_messages:
            self._messages = self._messages[-max_messages:]


class AssistantService(BaseService):
    """Service that processes inputs and generates responses.

    This service:
    1. Subscribes to transcriptions and text input
    2. Maintains conversation history
    3. Generates responses using Ollama LLM
    4. Publishes responses to all/events/assistant-response
    5. Optionally sends TTS requests to all/actions/speech
    """

    def __init__(self, config: AssistantServiceConfig) -> None:
        """Initialize the assistant service.

        Args:
            config: Service configuration
        """
        super().__init__(config)
        self._assistant_config = config

        # LLM client
        self._ollama: Optional[OllamaClient] = None

        # Conversation history
        self._history = ConversationHistory(max_turns=config.max_history)

        # Processing state
        self._processing_lock = threading.Lock()
        self._is_speaking = False
        self._tts_end_time: float = 0.0
        self._post_tts_grace_period: float = 1.5

        # Echo filter - simple word overlap check
        self._recent_responses: list[str] = []
        self._max_recent_responses = 5

        # Topics
        self._transcription_topic = Topics.EVENT_AUDIO_TRANSCRIBED.topic
        self._text_input_topic = Topics.EVENT_TEXT_INPUT.topic
        self._response_topic = Topics.EVENT_ASSISTANT_RESPONSE.topic
        self._speech_topic = Topics.ACTION_SPEECH.topic
        self._speaker_activity_topic = Topics.EVENT_SPEAKER_ACTIVITY.topic

    def _setup(self) -> None:
        """Set up LLM client and subscribe to inputs."""
        # Initialize Ollama client
        self._ollama = OllamaClient(
            base_url=self._assistant_config.ollama_host,
            timeout=self._assistant_config.ollama_timeout,
        )
        self._logger.info(f"Ollama client initialized: {self._assistant_config.ollama_host}")

        # Verify connection
        try:
            models_response = self._ollama.list_models()
            available = [m["name"] for m in models_response.get("models", [])]
            self._logger.info(f"Available Ollama models: {available}")

            if self._assistant_config.ollama_model not in available:
                model_with_latest = f"{self._assistant_config.ollama_model}:latest"
                if model_with_latest not in available:
                    self._logger.warning(
                        f"Model '{self._assistant_config.ollama_model}' not found. "
                        f"Available: {available}"
                    )
        except OllamaError as e:
            raise RuntimeError(f"Ollama connection failed: {e}")

        # Subscribe to inputs
        self._subscribe(self._transcription_topic, self._on_transcription)
        self._subscribe(self._text_input_topic, self._on_text_input)
        self._subscribe(self._speaker_activity_topic, self._on_speaker_activity)

        self._logger.info(f"Using model: {self._assistant_config.ollama_model}")

    def _cleanup(self) -> None:
        """Clean up resources."""
        if self._ollama:
            self._ollama.close()
            self._ollama = None

        self._history.clear()
        self._logger.info("Assistant service cleaned up")

    def _on_speaker_activity(self, topic: str, payload: bytes) -> None:
        """Handle speaker activity for echo prevention.

        Args:
            topic: MQTT topic
            payload: Message payload
        """
        try:
            message = SpeakerActivityMessage.from_bytes(payload)

            if message.state == SpeakerState.SPEAKING_STARTED:
                self._is_speaking = True
            elif message.state == SpeakerState.SPEAKING_ENDED:
                self._is_speaking = False
                self._tts_end_time = time.time()

        except Exception as e:
            self._logger.error(f"Error handling speaker activity: {e}")

    def _on_transcription(self, topic: str, payload: bytes) -> None:
        """Handle incoming transcription.

        Args:
            topic: MQTT topic
            payload: Message payload
        """
        try:
            message = TranscriptionMessage.from_bytes(payload)
            text = message.text.strip()

            if not text:
                return

            # Check if we're speaking (echo prevention)
            if self._is_speaking:
                self._logger.debug(f"Blocked during TTS: '{text}'")
                return

            # Check grace period after TTS
            time_since_tts = time.time() - self._tts_end_time
            if time_since_tts < self._post_tts_grace_period:
                # Check for echo
                if self._is_echo(text):
                    self._logger.debug(f"Echo blocked during grace period: '{text}'")
                    return

            self._logger.info(f"User (voice from {message.audio_source}): {text}")
            self._process_input(text, source="voice")

        except Exception as e:
            self._logger.error(f"Error handling transcription: {e}")

    def _on_text_input(self, topic: str, payload: bytes) -> None:
        """Handle incoming text input.

        Args:
            topic: MQTT topic
            payload: Message payload
        """
        try:
            message = TextInputMessage.from_bytes(payload)
            text = message.text.strip()

            if not text:
                return

            self._logger.info(f"User (text from {message.client_id}): {text}")
            self._process_input(text, source="text", session_id=message.session_id)

        except Exception as e:
            self._logger.error(f"Error handling text input: {e}")

    def _is_echo(self, text: str) -> bool:
        """Check if text is likely an echo of a recent response.

        Args:
            text: Text to check

        Returns:
            True if likely an echo
        """
        text_words = set(text.lower().split())

        for response in self._recent_responses:
            response_words = set(response.lower().split())
            if not response_words:
                continue

            overlap = len(text_words & response_words) / len(response_words)
            if overlap > 0.3:
                return True

        return False

    def _process_input(
        self,
        text: str,
        source: str = "unknown",
        session_id: str = "",
    ) -> None:
        """Process user input and generate response.

        Args:
            text: User input text
            source: Input source (voice/text)
            session_id: Session ID for tracking
        """
        # Process in background to not block MQTT
        thread = threading.Thread(
            target=self._generate_and_respond,
            args=(text, source, session_id),
            name="Assistant-Response",
            daemon=True,
        )
        thread.start()

    def _generate_and_respond(
        self,
        text: str,
        source: str,
        session_id: str,
    ) -> None:
        """Generate response and publish.

        Args:
            text: User input text
            source: Input source
            session_id: Session ID
        """
        with self._processing_lock:
            if not self._running:
                return

            try:
                # Add to history
                self._history.add_user_message(text)

                # Generate response
                response_text = self._generate_response(text)

                if not response_text:
                    self._logger.warning("Empty response from LLM")
                    return

                self._logger.info(f"Assistant: {response_text}")

                # Add to history
                self._history.add_assistant_message(response_text)

                # Track for echo prevention
                self._recent_responses.append(response_text)
                if len(self._recent_responses) > self._max_recent_responses:
                    self._recent_responses.pop(0)

                # Publish response
                response_msg = AssistantResponseMessage.create(
                    text=response_text,
                    session_id=session_id,
                    input_text=text,
                    model_name=self._assistant_config.ollama_model,
                )
                self._publish(self._response_topic, response_msg.to_bytes())

                # Send TTS request if enabled
                if self._assistant_config.enable_tts:
                    speech_msg = SpeechRequestMessage.create(
                        text=response_text,
                        voice=self._assistant_config.tts_voice,
                        speed=self._assistant_config.tts_speed,
                        source=self.service_name,
                    )
                    self._publish(self._speech_topic, speech_msg.to_bytes())

            except Exception as e:
                self._logger.error(f"Error generating response: {e}", exc_info=True)

    def _generate_response(self, user_text: str) -> str:
        """Generate LLM response.

        Args:
            user_text: User input

        Returns:
            Response text
        """
        if self._ollama is None:
            raise RuntimeError("Ollama client not initialized")

        messages = [{"role": "system", "content": self._assistant_config.system_prompt}]
        messages.extend(self._history.get_messages())

        try:
            response: dict[str, Any] = self._ollama.chat(
                model=self._assistant_config.ollama_model,
                messages=messages,
                stream=False,
                options={
                    "temperature": 0.7,
                    "top_p": 0.9,
                },
            )

            message = response.get("message", {})
            if isinstance(message, dict):
                content = str(message.get("content", ""))
                # Remove thinking tags
                content = re.sub(r"<think>.*?</think>\s*", "", content, flags=re.DOTALL)
                content = _strip_emojis(content)
                return content.strip()
            return ""

        except OllamaError as e:
            self._logger.error(f"LLM request failed: {e}")
            return "I'm sorry, I encountered an error processing your request."
