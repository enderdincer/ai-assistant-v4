"""Text input source implementation (mock version for demo)."""

import time
from typing import Any, Optional
from ai_assistant.shared.interfaces import IEventBus, EventPriority
from ai_assistant.shared.events import TextInputEvent
from ai_assistant.shared.logging import get_logger
from ai_assistant.perception.input_sources.base import BaseInputSource

logger = get_logger(__name__)


class TextInputSource(BaseInputSource):
    """Text input source (mock version for demonstration).

    Note: This is a mock implementation that generates sample text events.
    In a real implementation, you would read from stdin, a queue, or websocket.

    Configuration:
        - interval: How often to generate text in seconds (default: 5.0)
        - priority: Event priority (default: HIGH for user input)
        - messages: List of messages to cycle through (default: sample messages)
    """

    def __init__(
        self,
        source_id: str,
        event_bus: IEventBus,
        config: Optional[dict[str, Any]] = None,
    ) -> None:
        """Initialize text input source.

        Args:
            source_id: Unique identifier for this text source
            event_bus: Event bus for publishing text events
            config: Text input configuration
        """
        super().__init__(source_id, "text", event_bus, config)

        self._interval = self._config.get("interval", 5.0)
        self._priority = self._config.get("priority", EventPriority.HIGH)
        self._messages = self._config.get(
            "messages",
            [
                "Hello, AI assistant!",
                "What's the weather like today?",
                "Tell me a joke",
            ],
        )
        self._input_count = 0
        self._message_index = 0

    def _initialize_source(self) -> None:
        """Initialize the text input source."""
        logger.info(f"Text input source {self._source_id} initialized (mock mode)")
        logger.info(f"Will cycle through {len(self._messages)} messages every {self._interval}s")

    def _cleanup_source(self) -> None:
        """Clean up the text input source."""
        logger.info(f"Text input source {self._source_id} cleaned up")

    def _capture_and_publish(self) -> None:
        """Generate sample text input and publish it as an event."""
        # Wait for the interval
        time.sleep(self._interval)

        # Get next message
        text = self._messages[self._message_index]
        self._message_index = (self._message_index + 1) % len(self._messages)
        self._input_count += 1

        # Create and publish event
        event = TextInputEvent.create(
            source=self._source_id,
            text=text,
            priority=self._priority,
        )

        try:
            self._event_bus.publish(event)
            logger.info(f"Published text input #{self._input_count} from {self._source_id}: {text}")
        except Exception as e:
            logger.error(f"Failed to publish text input: {e}")
