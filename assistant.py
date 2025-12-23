#!/usr/bin/env python3
"""AI Assistant - Voice-driven conversational assistant.

This is the main application that orchestrates all components:
- Perception System: Audio input -> STT -> transcription events
- Decision Making: LLM (via Ollama) processes transcriptions
- Actions: TTS speaks the response

Flow:
    Audio Input -> STT Processor -> audio.transcription event
                                          |
                                          v
                                    Ollama LLM
                                          |
                                          v
                                    TTS Output

Usage:
    python assistant.py

    # Or with custom settings:
    python assistant.py --model gemma3:4b --voice af_heart

Environment Variables:
    OLLAMA_HOST: Ollama server URL (default: http://localhost:11434)
    OLLAMA_MODEL: Model to use (default: qwen3:0.6b)
    TTS_MODEL_PATH: Path to Kokoro ONNX model (default: .downloaded_models/kokoro-v1.0.onnx)
    TTS_VOICES_PATH: Path to Kokoro voices file (default: .downloaded_models/voices-v1.0.bin)
    TTS_VOICE: Voice to use (default: af_heart)
"""

import os
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

logger = get_logger(__name__)


@dataclass
class AssistantConfig:
    """Configuration for the AI Assistant.

    Attributes:
        # LLM Configuration
        ollama_host: Ollama server URL
        ollama_model: Model name to use for chat
        ollama_timeout: Request timeout in seconds
        system_prompt: System prompt for the assistant

        # TTS Configuration
        tts_model_path: Path to Kokoro ONNX model
        tts_voices_path: Path to Kokoro voices file
        tts_voice: Voice name to use
        tts_speed: Speech speed multiplier

        # Perception Configuration
        audio_source_id: Audio input source identifier
        stt_processor_id: STT processor identifier

        # Conversation settings
        max_history: Maximum conversation turns to keep
        log_level: Logging level
    """

    # LLM Configuration
    ollama_host: str = field(
        default_factory=lambda: os.getenv("OLLAMA_HOST", "http://localhost:11434")
    )
    ollama_model: str = field(default_factory=lambda: os.getenv("OLLAMA_MODEL", "qwen3:0.6b"))
    ollama_timeout: int = 120
    system_prompt: str = (
        "You are a helpful, friendly voice assistant. "
        "Keep your responses concise and conversational since they will be spoken aloud. "
        "Aim for 1-3 sentences unless the user asks for detailed information."
    )

    # TTS Configuration
    tts_model_path: Path = field(
        default_factory=lambda: Path(
            os.getenv(
                "TTS_MODEL_PATH", Path(__file__).parent / ".downloaded_models" / "kokoro-v1.0.onnx"
            )
        )
    )
    tts_voices_path: Path = field(
        default_factory=lambda: Path(
            os.getenv(
                "TTS_VOICES_PATH", Path(__file__).parent / ".downloaded_models" / "voices-v1.0.bin"
            )
        )
    )
    tts_voice: str = field(default_factory=lambda: os.getenv("TTS_VOICE", "af_heart"))
    tts_speed: float = 1.0

    # Perception Configuration
    audio_source_id: str = "microphone"
    stt_processor_id: str = "stt"

    # Conversation settings
    max_history: int = 10
    log_level: LogLevel = LogLevel.INFO

    @classmethod
    def from_env(cls) -> "AssistantConfig":
        """Create configuration from environment variables."""
        return cls()


class ConversationHistory:
    """Manages conversation history for context-aware responses."""

    def __init__(self, max_turns: int = 10):
        """Initialize conversation history.

        Args:
            max_turns: Maximum number of turns to keep
        """
        self._messages: list[dict[str, str]] = []
        self._max_turns = max_turns
        self._lock = threading.Lock()

    def add_user_message(self, text: str) -> None:
        """Add a user message to history."""
        with self._lock:
            self._messages.append({"role": "user", "content": text})
            self._trim()

    def add_assistant_message(self, text: str) -> None:
        """Add an assistant message to history."""
        with self._lock:
            self._messages.append({"role": "assistant", "content": text})
            self._trim()

    def get_messages(self) -> list[dict[str, str]]:
        """Get all messages in history."""
        with self._lock:
            return self._messages.copy()

    def clear(self) -> None:
        """Clear conversation history."""
        with self._lock:
            self._messages.clear()

    def _trim(self) -> None:
        """Trim history to max_turns (keeping pairs of user/assistant messages)."""
        # Keep max_turns * 2 messages (user + assistant pairs)
        max_messages = self._max_turns * 2
        if len(self._messages) > max_messages:
            self._messages = self._messages[-max_messages:]


class Assistant:
    """Main AI Assistant application.

    Orchestrates perception (audio -> transcription), decision making (LLM),
    and action execution (TTS) to create a voice-driven conversational assistant.
    """

    def __init__(self, config: Optional[AssistantConfig] = None):
        """Initialize the assistant.

        Args:
            config: Assistant configuration. Uses defaults if None.
        """
        self._config = config or AssistantConfig.from_env()

        # Setup logging
        setup_logging(LogConfig(level=self._config.log_level, colored_console=True))

        logger.info("Initializing AI Assistant...")

        # State
        self._running = False
        self._processing_lock = threading.Lock()
        self._is_speaking = False

        # Conversation history
        self._history = ConversationHistory(max_turns=self._config.max_history)

        # Components (initialized lazily)
        self._perception: Optional[PerceptionSystem] = None
        self._ollama: Optional[OllamaClient] = None
        self._tts: Optional[KokoroTTS] = None

    def initialize(self) -> None:
        """Initialize all components."""
        logger.info("Initializing components...")

        # Initialize Ollama client
        self._ollama = OllamaClient(
            base_url=self._config.ollama_host,
            timeout=self._config.ollama_timeout,
        )
        logger.info(f"Ollama client initialized: {self._config.ollama_host}")

        # Verify Ollama connection
        try:
            models_response = self._ollama.list_models()
            available = [m["name"] for m in models_response.get("models", [])]
            logger.info(f"Available Ollama models: {available}")

            if self._config.ollama_model not in available:
                # Try with :latest suffix
                model_with_latest = f"{self._config.ollama_model}:latest"
                if model_with_latest not in available:
                    logger.warning(
                        f"Model '{self._config.ollama_model}' not found. Available: {available}"
                    )
        except OllamaError as e:
            logger.error(f"Failed to connect to Ollama: {e}")
            raise RuntimeError(f"Ollama connection failed: {e}") from e

        # Initialize TTS
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
            logger.error(
                "Download Kokoro model files from: "
                "https://github.com/thewh1teagle/kokoro-onnx/releases"
            )
            raise
        except Exception as e:
            logger.error(f"Failed to initialize TTS: {e}")
            raise

        # Initialize perception system with audio + STT
        perception_config = PerceptionConfig.with_audio_and_stt(
            audio_source_id=self._config.audio_source_id,
            stt_processor_id=self._config.stt_processor_id,
        )
        perception_config.log_level = self._config.log_level

        self._perception = PerceptionSystem(perception_config)

        logger.info("All components initialized")

    def start(self) -> None:
        """Start the assistant."""
        if self._running:
            raise RuntimeError("Assistant is already running")

        if self._perception is None:
            raise RuntimeError("Assistant not initialized. Call initialize() first.")

        logger.info("Starting AI Assistant...")

        self._running = True

        # Subscribe to transcription events
        self._perception.subscribe("audio.transcription", self._on_transcription)

        # Start perception system
        self._perception.start()

        logger.info("AI Assistant started - listening for speech...")
        logger.info(f"Using model: {self._config.ollama_model}")
        logger.info(f"Using voice: {self._config.tts_voice}")
        logger.info("Speak into your microphone. Press Ctrl+C to stop.")

    def stop(self) -> None:
        """Stop the assistant gracefully."""
        if not self._running:
            return

        logger.info("Stopping AI Assistant...")

        self._running = False

        # Stop perception
        if self._perception:
            self._perception.unsubscribe("audio.transcription", self._on_transcription)
            self._perception.stop()

        # Cleanup TTS
        if self._tts:
            self._tts.cleanup()

        # Close Ollama client
        if self._ollama:
            self._ollama.close()

        logger.info("AI Assistant stopped")

    def is_running(self) -> bool:
        """Check if assistant is running."""
        return self._running

    def _on_transcription(self, event: IEvent) -> None:
        """Handle audio transcription events.

        This is called when the STT processor produces a transcription.

        Args:
            event: AudioTranscriptionEvent with transcribed text
        """
        if not self._running:
            return

        # Don't process while speaking (prevents echo)
        if self._is_speaking:
            logger.debug("Ignoring transcription while speaking")
            return

        # Extract transcription data
        text = event.data.get("text", "").strip()
        confidence = event.data.get("confidence", 0.0)

        if not text:
            return

        logger.info(f"User: {text} (confidence: {confidence:.2f})")

        # Process in a separate thread to not block the event bus
        thread = threading.Thread(
            target=self._process_and_respond,
            args=(text,),
            name="Assistant-Response",
            daemon=True,
        )
        thread.start()

    def _process_and_respond(self, user_text: str) -> None:
        """Process user input and generate/speak response.

        Args:
            user_text: Transcribed user input
        """
        # Use lock to prevent concurrent processing
        with self._processing_lock:
            if not self._running:
                return

            try:
                # Add user message to history
                self._history.add_user_message(user_text)

                # Generate response from LLM
                response_text = self._generate_response(user_text)

                if not response_text:
                    logger.warning("Empty response from LLM")
                    return

                logger.info(f"Assistant: {response_text}")

                # Add assistant message to history
                self._history.add_assistant_message(response_text)

                # Speak the response
                self._speak(response_text)

            except Exception as e:
                logger.error(f"Error processing response: {e}", exc_info=True)

    def _generate_response(self, user_text: str) -> str:
        """Generate a response using the LLM.

        Args:
            user_text: User's input text

        Returns:
            Generated response text
        """
        if self._ollama is None:
            raise RuntimeError("Ollama client not initialized")

        # Build messages with system prompt and history
        messages = [{"role": "system", "content": self._config.system_prompt}]
        messages.extend(self._history.get_messages())

        try:
            response: dict[str, Any] = self._ollama.chat(  # type: ignore[assignment]
                model=self._config.ollama_model,
                messages=messages,
                stream=False,
                options={
                    "temperature": 0.7,
                    "top_p": 0.9,
                },
            )

            # Extract response text
            message = response.get("message", {})
            if isinstance(message, dict):
                return str(message.get("content", ""))
            return ""

        except OllamaError as e:
            logger.error(f"LLM request failed: {e}")
            return "I'm sorry, I encountered an error processing your request."

    def _speak(self, text: str) -> None:
        """Speak text using TTS.

        Args:
            text: Text to speak
        """
        if self._tts is None:
            logger.warning("TTS not initialized, skipping speech")
            return

        try:
            self._is_speaking = True
            logger.debug(f"Speaking: {text[:50]}...")

            # Synthesize and play
            self._tts.speak(text)

        except Exception as e:
            logger.error(f"TTS error: {e}")
        finally:
            self._is_speaking = False

    def run_forever(self) -> None:
        """Run the assistant until interrupted."""
        self.initialize()
        self.start()

        # Setup signal handlers
        def signal_handler(signum: int, frame: Any) -> None:
            logger.info(f"Received signal {signum}, shutting down...")
            self.stop()
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        # Keep running
        try:
            while self._running:
                time.sleep(0.1)
        except KeyboardInterrupt:
            pass
        finally:
            self.stop()


def main() -> None:
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="AI Assistant - Voice-driven conversational assistant",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python assistant.py
    python assistant.py --model qwen3:0.6b --voice af_heart
    python assistant.py --verbose

Environment Variables:
    OLLAMA_HOST     Ollama server URL (default: http://localhost:11434)
    OLLAMA_MODEL    Model to use (default: qwen3:0.6b)
    TTS_MODEL_PATH  Path to Kokoro ONNX model (default: .downloaded_models/kokoro-v1.0.onnx)
    TTS_VOICES_PATH Path to Kokoro voices file (default: .downloaded_models/voices-v1.0.bin)
    TTS_VOICE       Voice to use (default: af_heart)
        """,
    )

    parser.add_argument(
        "--model",
        type=str,
        default=os.getenv("OLLAMA_MODEL", "qwen3:0.6b"),
        help="Ollama model to use",
    )
    parser.add_argument(
        "--voice",
        type=str,
        default=os.getenv("TTS_VOICE", "af_heart"),
        help="TTS voice to use",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="TTS speech speed (default: 1.0)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )
    parser.add_argument(
        "--list-voices",
        action="store_true",
        help="List available TTS voices and exit",
    )

    args = parser.parse_args()

    # Handle --list-voices
    if args.list_voices:
        try:
            config = AssistantConfig()
            tts_config = KokoroTTSConfig(
                model_path=config.tts_model_path,
                voices_path=config.tts_voices_path,
            )
            tts = KokoroTTS(tts_config)
            tts.initialize()
            voices = tts.get_available_voices()
            print("Available voices:")
            for voice in sorted(voices):
                print(f"  {voice}")
            tts.cleanup()
        except Exception as e:
            print(f"Error listing voices: {e}")
            sys.exit(1)
        return

    # Create configuration
    config = AssistantConfig(
        ollama_model=args.model,
        tts_voice=args.voice,
        tts_speed=args.speed,
        log_level=LogLevel.DEBUG if args.verbose else LogLevel.INFO,
    )

    # Create and run assistant
    assistant = Assistant(config)

    try:
        assistant.run_forever()
    except Exception as e:
        logger.error(f"Assistant error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
