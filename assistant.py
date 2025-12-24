#!/usr/bin/env python3

import os
import re
import signal
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from ai_assistant.shared.logging import get_logger, setup_logging, LogConfig, LogLevel
from ai_assistant.shared.interfaces import IEvent
from ai_assistant.shared.ollama import OllamaClient, OllamaError
from ai_assistant.perception.core import PerceptionSystem, PerceptionConfig
from ai_assistant.actions.tts import KokoroTTS, KokoroTTSConfig
from echo_filter import EchoFilter

logger = get_logger(__name__)


def _strip_emojis(text: str) -> str:
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
class AssistantConfig:
    ollama_host: str = field(
        default_factory=lambda: os.getenv("OLLAMA_HOST", "http://localhost:11434")
    )
    ollama_model: str = field(default_factory=lambda: os.getenv("OLLAMA_MODEL", "qwen3:1.7b"))
    ollama_timeout: int = field(default_factory=lambda: int(os.getenv("OLLAMA_TIMEOUT", "120")))
    system_prompt: str = field(
        default_factory=lambda: os.getenv(
            "SYSTEM_PROMPT",
            "You are a helpful, friendly voice assistant. "
            "Keep your responses concise unless the user asks for detailed information. "
            "Never use emojis.",
        )
    )
    tts_model_path: Path = field(
        default_factory=lambda: Path(
            os.getenv(
                "TTS_MODEL_PATH",
                str(Path(__file__).parent / ".downloaded_models" / "kokoro-v1.0.onnx"),
            )
        )
    )
    tts_voices_path: Path = field(
        default_factory=lambda: Path(
            os.getenv(
                "TTS_VOICES_PATH",
                str(Path(__file__).parent / ".downloaded_models" / "voices-v1.0.bin"),
            )
        )
    )
    tts_voice: str = field(default_factory=lambda: os.getenv("TTS_VOICE", "am_santa"))
    tts_speed: float = field(default_factory=lambda: float(os.getenv("TTS_SPEED", "1.0")))
    audio_source_id: str = "microphone"
    stt_processor_id: str = "stt"
    max_history: int = field(default_factory=lambda: int(os.getenv("MAX_HISTORY", "10")))
    log_level: LogLevel = field(
        default_factory=lambda: LogLevel.DEBUG
        if os.getenv("DEBUG", "").lower() in ("1", "true", "yes")
        else LogLevel.INFO
    )


class ConversationHistory:
    def __init__(self, max_turns: int = 10):
        self._messages: list[dict[str, str]] = []
        self._max_turns = max_turns
        self._lock = threading.Lock()

    def add_user_message(self, text: str) -> None:
        with self._lock:
            self._messages.append({"role": "user", "content": text})
            self._trim()

    def add_assistant_message(self, text: str) -> None:
        with self._lock:
            self._messages.append({"role": "assistant", "content": text})
            self._trim()

    def get_messages(self) -> list[dict[str, str]]:
        with self._lock:
            return self._messages.copy()

    def clear(self) -> None:
        with self._lock:
            self._messages.clear()

    def _trim(self) -> None:
        max_messages = self._max_turns * 2
        if len(self._messages) > max_messages:
            self._messages = self._messages[-max_messages:]


class Assistant:
    def __init__(self, config: Optional[AssistantConfig] = None):
        self._config = config or AssistantConfig()
        setup_logging(LogConfig(level=self._config.log_level, colored_console=True))
        logger.info("Initializing AI Assistant...")

        self._running = False
        self._processing_lock = threading.Lock()
        self._is_speaking = False
        self._tts_end_time: float = 0.0
        self._post_tts_grace_period: float = 1.5

        self._history = ConversationHistory(max_turns=self._config.max_history)
        self._echo_filter = EchoFilter(
            buffer_ms=2500,
            similarity_threshold=0.40,
            word_overlap_threshold=0.30,
            max_stored_responses=5,
        )

        self._perception: Optional[PerceptionSystem] = None
        self._ollama: Optional[OllamaClient] = None
        self._tts: Optional[KokoroTTS] = None
        self._audio_source: Optional[Any] = None
        self._stt_processor: Optional[Any] = None

    def initialize(self) -> None:
        logger.info("Initializing components...")

        self._ollama = OllamaClient(
            base_url=self._config.ollama_host,
            timeout=self._config.ollama_timeout,
        )
        logger.info(f"Ollama client initialized: {self._config.ollama_host}")

        try:
            models_response = self._ollama.list_models()
            available = [m["name"] for m in models_response.get("models", [])]
            logger.info(f"Available Ollama models: {available}")

            if self._config.ollama_model not in available:
                model_with_latest = f"{self._config.ollama_model}:latest"
                if model_with_latest not in available:
                    logger.warning(
                        f"Model '{self._config.ollama_model}' not found. Available: {available}"
                    )
        except OllamaError as e:
            logger.error(f"Failed to connect to Ollama: {e}")
            raise RuntimeError(f"Ollama connection failed: {e}") from e

        try:
            tts_config = KokoroTTSConfig(
                model_path=self._config.tts_model_path,
                voices_path=self._config.tts_voices_path,
                voice=self._config.tts_voice,
                speed=self._config.tts_speed,
            )
            self._tts = KokoroTTS(tts_config)
            self._tts.initialize()
            logger.info(f"TTS initialized with voice: {self._config.tts_voice}")
        except FileNotFoundError as e:
            logger.error(f"TTS model files not found: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize TTS: {e}")
            raise

        perception_config = PerceptionConfig.with_audio_and_stt(
            audio_source_id=self._config.audio_source_id,
            stt_processor_id=self._config.stt_processor_id,
        )
        perception_config.log_level = self._config.log_level
        self._perception = PerceptionSystem(perception_config)

        logger.info("All components initialized")

    def start(self) -> None:
        if self._running:
            raise RuntimeError("Assistant is already running")

        if self._perception is None:
            raise RuntimeError("Assistant not initialized. Call initialize() first.")

        logger.info("Starting AI Assistant...")
        self._running = True

        self._perception.subscribe("audio.transcription", self._on_transcription)
        self._perception.start()

        self._audio_source = self._perception.get_input_source(self._config.audio_source_id)
        if self._audio_source:
            logger.info("Audio source acquired")
        else:
            logger.warning("Audio source not found")

        self._stt_processor = self._perception.get_processor(self._config.stt_processor_id)
        if self._stt_processor:
            logger.info("STT processor acquired")
        else:
            logger.warning("STT processor not found")

        logger.info("AI Assistant started - listening for speech...")
        logger.info(f"Using model: {self._config.ollama_model}")
        logger.info(f"Using voice: {self._config.tts_voice}")

    def stop(self) -> None:
        if not self._running:
            return

        logger.info("Stopping AI Assistant...")
        self._running = False

        if self._perception:
            self._perception.unsubscribe("audio.transcription", self._on_transcription)
            self._perception.stop()

        if self._tts:
            self._tts.cleanup()

        if self._ollama:
            self._ollama.close()

        logger.info("AI Assistant stopped")

    def is_running(self) -> bool:
        return self._running

    def _on_transcription(self, event: IEvent) -> None:
        if not self._running:
            return

        text = event.data.get("text", "").strip()
        confidence = event.data.get("confidence", 0.0)

        if not text:
            return

        if self._is_speaking:
            logger.debug(f"Blocked during TTS playback: '{text}'")
            return

        time_since_tts = time.time() - self._tts_end_time
        is_in_grace_period = time_since_tts < self._post_tts_grace_period

        if is_in_grace_period:
            if self._echo_filter.is_echo(text):
                logger.debug(f"Echo blocked during grace period ({time_since_tts:.2f}s): '{text}'")
                return
            else:
                logger.info(
                    f"Potential user input during grace period ({time_since_tts:.2f}s): '{text}'"
                )
        else:
            if self._echo_filter.is_echo(text):
                logger.debug(f"Echo filtered: '{text}'")
                return

        logger.info(f"User: {text} (confidence: {confidence:.2f})")

        thread = threading.Thread(
            target=self._process_and_respond,
            args=(text,),
            name="Assistant-Response",
            daemon=True,
        )
        thread.start()

    def _process_and_respond(self, user_text: str) -> None:
        with self._processing_lock:
            if not self._running:
                return

            try:
                self._history.add_user_message(user_text)
                response_text = self._generate_response(user_text)

                if not response_text:
                    logger.warning("Empty response from LLM")
                    return

                logger.info(f"Assistant: {response_text}")
                self._history.add_assistant_message(response_text)
                self._speak(response_text)

            except Exception as e:
                logger.error(f"Error processing response: {e}", exc_info=True)

    def _generate_response(self, user_text: str) -> str:
        if self._ollama is None:
            raise RuntimeError("Ollama client not initialized")

        messages = [{"role": "system", "content": self._config.system_prompt}]
        messages.extend(self._history.get_messages())

        try:
            response: dict[str, Any] = self._ollama.chat(
                model=self._config.ollama_model,
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
                content = re.sub(r"<think>.*?</think>\s*", "", content, flags=re.DOTALL)
                content = _strip_emojis(content)
                return content.strip()
            return ""

        except OllamaError as e:
            logger.error(f"LLM request failed: {e}")
            return "I'm sorry, I encountered an error processing your request."

    def _speak(self, text: str) -> None:
        if self._tts is None:
            logger.warning("TTS not initialized, skipping speech")
            return

        try:
            self._is_speaking = True

            if self._stt_processor is not None and hasattr(self._stt_processor, "reset_vad"):
                self._stt_processor.reset_vad()
                logger.debug("VAD reset before TTS playback")

            duration_ms = self._tts.estimate_duration(text)
            self._echo_filter.set_tts_response(text, duration_ms)

            logger.debug(f"Speaking ({duration_ms:.0f}ms): {text[:50]}...")
            self._tts.speak(text)

        except Exception as e:
            logger.error(f"TTS error: {e}")
        finally:
            self._is_speaking = False
            self._tts_end_time = time.time()

            if self._stt_processor is not None and hasattr(self._stt_processor, "reset_vad"):
                self._stt_processor.reset_vad()
                logger.debug("VAD reset after TTS playback")

            logger.debug(f"TTS ended, grace period of {self._post_tts_grace_period}s started")

    def run_forever(self) -> None:
        self.initialize()
        self.start()

        def signal_handler(signum: int, frame: Any) -> None:
            logger.info(f"Received signal {signum}, shutting down...")
            self.stop()
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        try:
            while self._running:
                time.sleep(0.1)
        except KeyboardInterrupt:
            pass
        finally:
            self.stop()


def main() -> None:
    assistant = Assistant()
    try:
        assistant.run_forever()
    except Exception as e:
        logger.error(f"Assistant error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
