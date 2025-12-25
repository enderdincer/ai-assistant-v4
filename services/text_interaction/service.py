"""Text Interaction Service implementation.

Provides a CLI interface for text-based interaction with the assistant:
1. Reads user input from stdin
2. Publishes TextInputMessage to MQTT
3. Subscribes to assistant responses and displays them
4. Supports session management commands (/new, /switch, /sessions, /session)
"""

import json
import os
import sys
import threading
import time
import urllib.request
import urllib.error
import uuid
from dataclasses import dataclass
from typing import Optional

from ai_assistant.shared.logging import get_logger, LogLevel
from ai_assistant.shared.services import BaseService, ServiceConfig
from ai_assistant.shared.messages import (
    TextInputMessage,
    AssistantResponseMessage,
    SessionChangedMessage,
)
from ai_assistant.shared.mqtt.topics import Topics

logger = get_logger(__name__)


@dataclass
class TextInteractionServiceConfig(ServiceConfig):
    """Configuration for Text Interaction Service.

    Attributes:
        client_id: Unique client identifier
        session_id: Initial session ID (fetched from memory service if empty)
        show_metadata: Whether to show message metadata
        prompt: Input prompt string
        memory_service_url: URL for memory service HTTP API
    """

    client_id: str = ""
    session_id: str = ""
    show_metadata: bool = False
    prompt: str = "You: "
    memory_service_url: str = "http://localhost:8080"

    @classmethod
    def from_env(cls) -> "TextInteractionServiceConfig":
        """Create configuration from environment variables."""
        return cls(
            service_name="text-interaction-service",
            client_id=os.getenv("CLIENT_ID", f"cli-{uuid.uuid4().hex[:8]}"),
            session_id=os.getenv("SESSION_ID", ""),  # Empty = fetch from memory service
            show_metadata=os.getenv("SHOW_METADATA", "false").lower() in ("1", "true", "yes"),
            prompt=os.getenv("PROMPT", "You: "),
            memory_service_url=os.getenv("MEMORY_SERVICE_URL", "http://localhost:8080"),
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
    5. Supports session management commands
    """

    def __init__(self, config: TextInteractionServiceConfig) -> None:
        """Initialize the text interaction service.

        Args:
            config: Service configuration
        """
        super().__init__(config)
        self._text_config = config

        # Session management - use valid UUID as fallback
        self._current_session: str = config.session_id or "00000000-0000-0000-0000-000000000000"
        self._session_lock = threading.Lock()

        # Topics
        self._text_input_topic = Topics.EVENT_TEXT_INPUT.topic
        self._response_topic = Topics.EVENT_ASSISTANT_RESPONSE.topic
        self._session_changed_topic = Topics.EVENT_SESSION_CHANGED.topic

        # Input thread
        self._input_thread: Optional[threading.Thread] = None
        self._waiting_for_response = threading.Event()

    def _setup(self) -> None:
        """Set up subscriptions and start input loop."""
        # Subscribe to responses
        self._subscribe(self._response_topic, self._on_response)

        # Subscribe to session changes
        self._subscribe(self._session_changed_topic, self._on_session_changed)

        # Fetch current session from memory service if not specified
        if not self._text_config.session_id:
            self._fetch_current_session()

        # Print welcome message
        print("\n" + "=" * 60)
        print("Text Interaction Service")
        print("=" * 60)
        print(f"Client ID: {self._text_config.client_id}")
        with self._session_lock:
            print(f"Session ID: {self._current_session}")
        print("Type your message and press Enter. Type 'quit' or 'exit' to stop.")
        print("Type '/help' for session commands.")
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
            with self._session_lock:
                current_session = self._current_session

            if current_session and message.session_id:
                if message.session_id != current_session:
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

                # Check for commands (starting with /)
                if text.startswith("/"):
                    self._handle_command(text)
                    continue

                # Special commands (legacy, without /)
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
        with self._session_lock:
            session_id = self._current_session

        message = TextInputMessage.create(
            text=text,
            session_id=session_id,
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
        print("Session Commands:")
        print("  /session           - Show current session info")
        print("  /sessions          - List all sessions")
        print("  /new               - Create a new session")
        print("  /switch <id>       - Switch to a different session")
        print()
        print("Just type your message and press Enter to chat with the assistant.")
        print()

    # =========================================================================
    # Session Management
    # =========================================================================

    def _fetch_current_session(self) -> None:
        """Fetch the current session from memory service."""
        try:
            url = f"{self._text_config.memory_service_url}/sessions/current"
            self._logger.debug(f"Fetching current session from {url}")

            req = urllib.request.Request(url, method="GET")
            req.add_header("Accept", "application/json")

            with urllib.request.urlopen(req, timeout=5.0) as response:
                data = json.loads(response.read().decode("utf-8"))
                session_id = data.get("session_id")

                if session_id:
                    with self._session_lock:
                        self._current_session = session_id
                    self._logger.info(f"Using session from memory: {session_id}")

        except Exception as e:
            self._logger.warning(
                f"Could not fetch session from memory service: {e}. Using default session."
            )

    def _on_session_changed(self, topic: str, payload: bytes) -> None:
        """Handle session change events from memory service.

        Args:
            topic: MQTT topic
            payload: Message payload
        """
        try:
            message = SessionChangedMessage.from_bytes(payload)
            new_session = message.session_id

            if not new_session:
                return

            with self._session_lock:
                old_session = self._current_session
                if new_session != old_session:
                    self._current_session = new_session
                    print(f"\n[Session changed: {old_session} -> {new_session}]")
                    print(self._text_config.prompt, end="", flush=True)

        except Exception as e:
            self._logger.error(f"Error handling session change: {e}")

    def _handle_command(self, text: str) -> None:
        """Handle slash commands.

        Args:
            text: Command text (starts with /)
        """
        parts = text.split(maxsplit=1)
        cmd = parts[0].lower()
        arg = parts[1] if len(parts) > 1 else ""

        if cmd == "/help":
            self._show_help()
        elif cmd == "/session":
            self._cmd_session()
        elif cmd == "/sessions":
            self._cmd_list_sessions()
        elif cmd == "/new":
            self._cmd_new_session()
        elif cmd == "/switch":
            if arg:
                self._cmd_switch_session(arg)
            else:
                print("Usage: /switch <session_id>")
        else:
            print(f"Unknown command: {cmd}")
            print("Type /help for available commands.")

    def _cmd_session(self) -> None:
        """Show current session info."""
        with self._session_lock:
            session_id = self._current_session

        try:
            url = f"{self._text_config.memory_service_url}/sessions/{session_id}"
            req = urllib.request.Request(url, method="GET")
            req.add_header("Accept", "application/json")

            with urllib.request.urlopen(req, timeout=5.0) as response:
                data = json.loads(response.read().decode("utf-8"))
                print()
                print(f"Current Session: {data.get('session_id', session_id)}")
                print(f"  Messages: {data.get('message_count', 0)}")
                if data.get("created_at"):
                    print(f"  Created: {data.get('created_at')}")
                if data.get("last_activity"):
                    print(f"  Last Activity: {data.get('last_activity')}")
                print()

        except urllib.error.HTTPError as e:
            if e.code == 404:
                print(f"\nSession {session_id} (new, no messages yet)\n")
            else:
                print(f"\nError fetching session info: {e}\n")
        except Exception as e:
            print(f"\nError: Could not fetch session info: {e}\n")

    def _cmd_list_sessions(self) -> None:
        """List all sessions."""
        try:
            url = f"{self._text_config.memory_service_url}/sessions?limit=20"
            req = urllib.request.Request(url, method="GET")
            req.add_header("Accept", "application/json")

            with urllib.request.urlopen(req, timeout=5.0) as response:
                data = json.loads(response.read().decode("utf-8"))
                sessions = data.get("sessions", [])
                total = data.get("total", len(sessions))

                print()
                print(f"Sessions ({total} total):")
                print("-" * 60)

                with self._session_lock:
                    current = self._current_session

                for s in sessions:
                    sid = s.get("session_id", "")
                    marker = " *" if sid == current else ""
                    msgs = s.get("message_count", 0)
                    print(f"  {sid}{marker} ({msgs} messages)")

                if not sessions:
                    print("  (no sessions)")

                print("-" * 60)
                print("* = current session")
                print()

        except Exception as e:
            print(f"\nError: Could not list sessions: {e}\n")

    def _cmd_new_session(self) -> None:
        """Create a new session via memory service."""
        try:
            url = f"{self._text_config.memory_service_url}/sessions"
            req = urllib.request.Request(
                url,
                method="POST",
                data=json.dumps({"set_active": True}).encode("utf-8"),
            )
            req.add_header("Content-Type", "application/json")
            req.add_header("Accept", "application/json")

            with urllib.request.urlopen(req, timeout=5.0) as response:
                data = json.loads(response.read().decode("utf-8"))
                session_id = data.get("session_id", "")

                if session_id:
                    with self._session_lock:
                        old_session = self._current_session
                        self._current_session = session_id

                    print(f"\nNew session created: {session_id}")
                    print(f"  (previous: {old_session})")
                    print()

        except Exception as e:
            print(f"\nError: Could not create session: {e}\n")

    def _cmd_switch_session(self, session_id: str) -> None:
        """Switch to a different session.

        Args:
            session_id: Session ID to switch to
        """
        try:
            url = f"{self._text_config.memory_service_url}/sessions/current"
            req = urllib.request.Request(
                url,
                method="PUT",
                data=json.dumps({"session_id": session_id}).encode("utf-8"),
            )
            req.add_header("Content-Type", "application/json")
            req.add_header("Accept", "application/json")

            with urllib.request.urlopen(req, timeout=5.0) as response:
                data = json.loads(response.read().decode("utf-8"))
                new_session = data.get("session_id", session_id)

                with self._session_lock:
                    old_session = self._current_session
                    self._current_session = new_session

                print(f"\nSwitched to session: {new_session}")
                print(f"  Messages: {data.get('message_count', 0)}")
                print()

        except Exception as e:
            print(f"\nError: Could not switch session: {e}\n")
