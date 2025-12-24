"""Text Interaction Service implementation.

Provides a CLI interface for text-based interaction with the assistant:
1. Reads user input from stdin
2. Publishes TextInputMessage to MQTT
3. Subscribes to assistant responses and displays them
"""

import os
import sys
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Optional

from ai_assistant.shared.logging import get_logger, LogLevel
from ai_assistant.shared.services import BaseService, ServiceConfig
from ai_assistant.shared.messages import (
    TextInputMessage,
    AssistantResponseMessage,
)
from ai_assistant.shared.mqtt.topics import Topics

logger = get_logger(__name__)


@dataclass
class TextInteractionServiceConfig(ServiceConfig):
    """Configuration for Text Interaction Service.

    Attributes:
        client_id: Unique client identifier
        session_id: Session ID for conversation continuity
        show_metadata: Whether to show message metadata
        prompt: Input prompt string
    """

    client_id: str = ""
    session_id: str = ""
    show_metadata: bool = False
    prompt: str = "You: "

    @classmethod
    def from_env(cls) -> "TextInteractionServiceConfig":
        """Create configuration from environment variables."""
        return cls(
            service_name="text-interaction-service",
            client_id=os.getenv("CLIENT_ID", f"cli-{uuid.uuid4().hex[:8]}"),
            session_id=os.getenv("SESSION_ID", uuid.uuid4().hex),
            show_metadata=os.getenv("SHOW_METADATA", "false").lower() in ("1", "true", "yes"),
            prompt=os.getenv("PROMPT", "You: "),
            log_level=LogLevel.DEBUG
            if os.getenv("DEBUG", "").lower() in ("1", "true", "yes")
            else LogLevel.INFO,  # Show info messages by default
        )


class TextInteractionService(BaseService):
    """Service providing CLI interface for text interaction.

    This service:
    1. Runs an input loop reading from stdin
    2. Publishes TextInputMessage to all/events/text-input
    3. Subscribes to all/events/assistant-response
    4. Displays assistant responses to stdout
    """

    def __init__(self, config: TextInteractionServiceConfig) -> None:
        """Initialize the text interaction service.

        Args:
            config: Service configuration
        """
        super().__init__(config)
        self._text_config = config

        # Topics
        self._text_input_topic = Topics.EVENT_TEXT_INPUT.topic
        self._response_topic = Topics.EVENT_ASSISTANT_RESPONSE.topic

        # Input thread
        self._input_thread: Optional[threading.Thread] = None
        self._waiting_for_response = threading.Event()

    def _setup(self) -> None:
        """Set up subscriptions and start input loop."""
        # Subscribe to responses
        self._subscribe(self._response_topic, self._on_response)

        # Print welcome message
        print("\n" + "=" * 60)
        print("Text Interaction Service")
        print("=" * 60)
        print(f"Client ID: {self._text_config.client_id}")
        print(f"Session ID: {self._text_config.session_id}")
        print("Type your message and press Enter. Type 'quit' or 'exit' to stop.")
        print("=" * 60 + "\n")

        # Start input thread
        self._input_thread = threading.Thread(
            target=self._input_loop,
            name="Input-Loop",
            daemon=True,
        )
        self._input_thread.start()

        self._logger.info("Text interaction service ready")

    def _cleanup(self) -> None:
        """Clean up resources."""
        print("\nGoodbye!")
        self._logger.info("Text interaction service cleaned up")

    def _on_response(self, topic: str, payload: bytes) -> None:
        """Handle incoming assistant response.

        Args:
            topic: MQTT topic
            payload: Message payload
        """
        try:
            message = AssistantResponseMessage.from_bytes(payload)

            # Only show responses for our session (or all if no session filter)
            if self._text_config.session_id and message.session_id:
                if message.session_id != self._text_config.session_id:
                    return

            # Display response
            print()  # Newline for separation
            if self._text_config.show_metadata:
                print(f"[{message.model_name}] Assistant: {message.text}")
            else:
                print(f"Assistant: {message.text}")
            print()

            # Signal that response was received
            self._waiting_for_response.clear()

            # Show prompt again
            print(self._text_config.prompt, end="", flush=True)

        except Exception as e:
            self._logger.error(f"Error handling response: {e}")

    def _input_loop(self) -> None:
        """Main input loop reading from stdin."""
        # Wait for service to be running
        while not self._running:
            time.sleep(0.1)

        while self._running:
            try:
                # Show prompt
                print(self._text_config.prompt, end="", flush=True)

                # Read input (blocking)
                line = sys.stdin.readline()

                if not line:
                    # EOF - stdin closed
                    self._logger.info("Input stream closed")
                    self.stop()
                    break

                text = line.strip()

                if not text:
                    continue

                # Check for exit commands
                if text.lower() in ("quit", "exit", "q", ":q"):
                    self.stop()
                    break

                # Special commands
                if text.lower() == "clear":
                    print("\033[H\033[J", end="")  # Clear screen
                    continue

                if text.lower() == "help":
                    self._show_help()
                    continue

                # Send message
                self._send_message(text)

            except KeyboardInterrupt:
                self._logger.info("Keyboard interrupt in input loop")
                self.stop()
                break
            except Exception as e:
                self._logger.error(f"Error in input loop: {e}")

    def _send_message(self, text: str) -> None:
        """Send a text message to the assistant.

        Args:
            text: User's text message
        """
        message = TextInputMessage.create(
            text=text,
            session_id=self._text_config.session_id,
            client_id=self._text_config.client_id,
        )

        self._publish(self._text_input_topic, message.to_bytes())
        self._logger.info(f"Sent message to {self._text_input_topic}: {text}")

        # Mark that we're waiting for response
        self._waiting_for_response.set()

    def _show_help(self) -> None:
        """Show help message."""
        print()
        print("Commands:")
        print("  quit, exit, q, :q  - Exit the program")
        print("  clear              - Clear the screen")
        print("  help               - Show this help message")
        print()
        print("Just type your message and press Enter to chat with the assistant.")
        print()
