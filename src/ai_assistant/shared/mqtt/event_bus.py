"""MQTT-based event bus implementation.

This module provides an event bus that uses MQTT as the transport layer,
allowing distributed components to communicate via a shared MQTT broker.
It implements the same IEventBus interface as the local EventBus.
"""

import threading
from collections import defaultdict
from typing import DefaultDict, Optional

from ai_assistant.shared.interfaces import IEvent, EventHandler
from ai_assistant.shared.logging import get_logger
from ai_assistant.shared.mqtt.client import MQTTClient
from ai_assistant.shared.mqtt.config import MQTTConfig
from ai_assistant.shared.mqtt.serialization import JSONSerializer
from ai_assistant.shared.mqtt.topics import (
    event_type_to_topic,
    topic_to_event_type,
    get_subscription_pattern,
    TopicCategory,
)

logger = get_logger(__name__)


class MQTTEventBus:
    """MQTT-based event bus for distributed pub-sub communication.

    This implementation routes events through an MQTT broker instead of
    a local queue, enabling:
    - Distributed components across processes/machines
    - Topic-based routing via MQTT wildcards
    - Message persistence (with QoS 1/2)
    - Decoupled deployment

    The bus maps internal event types to MQTT topics:
    - Raw sensory data -> all/raw/{type}
    - Processed events -> all/events/{type}
    - Action requests  -> all/actions/{type}
    """

    def __init__(
        self,
        config: Optional[MQTTConfig] = None,
        client: Optional[MQTTClient] = None,
    ):
        """Initialize the MQTT event bus.

        Args:
            config: MQTT configuration (used if client not provided)
            client: Optional pre-configured MQTT client
        """
        self._config = config or MQTTConfig.from_env()
        self._client = client or MQTTClient(self._config)
        self._owns_client = client is None  # We manage lifecycle if we created it
        self._serializer = JSONSerializer()
        self._subscribers: DefaultDict[str, list[EventHandler]] = defaultdict(list)
        self._topic_subscriptions: set[str] = set()
        self._lock = threading.RLock()
        self._running = False
        self._message_count = 0
        self._published_count = 0

    def initialize(self) -> None:
        """Initialize the event bus and underlying MQTT client."""
        if self._owns_client:
            self._client.initialize()
        logger.info("MQTT event bus initialized")

    def start(self) -> None:
        """Start the event bus and connect to MQTT broker."""
        with self._lock:
            if self._running:
                raise RuntimeError("MQTT event bus is already running")
            self._running = True

        if self._owns_client:
            self._client.start()

        # Wait briefly for connection
        import time

        for _ in range(50):  # Up to 5 seconds
            if self._client.is_connected():
                break
            time.sleep(0.1)

        if not self._client.is_connected():
            logger.warning("MQTT broker not connected yet, will retry in background")

        logger.info("MQTT event bus started")

    def stop(self) -> None:
        """Stop the event bus gracefully."""
        with self._lock:
            if not self._running:
                return
            self._running = False

        if self._owns_client:
            self._client.stop()

        logger.info(
            f"MQTT event bus stopped. "
            f"Published: {self._published_count}, Received: {self._message_count}"
        )

    def is_running(self) -> bool:
        """Check if the event bus is running."""
        return self._running

    def publish(self, event: IEvent) -> None:
        """Publish an event to MQTT.

        The event is serialized and published to the appropriate topic
        based on the event type mapping.

        Args:
            event: The event to publish

        Raises:
            RuntimeError: If event bus is not running
        """
        if not self._running:
            raise RuntimeError("MQTT event bus is not running")

        # Map event type to MQTT topic
        topic = event_type_to_topic(event.event_type)

        # Serialize the event
        try:
            payload = self._serializer.serialize(event)
        except Exception as e:
            logger.error(f"Failed to serialize event {event.event_type}: {e}")
            return

        # Publish to MQTT
        success = self._client.publish(topic, payload)
        if success:
            self._published_count += 1
            logger.debug(f"Published {event.event_type} to {topic}")
        else:
            logger.error(f"Failed to publish {event.event_type} to {topic}")

    def subscribe(self, event_type: str, handler: EventHandler) -> None:
        """Subscribe to events of a specific type.

        The handler will be called when matching events are received
        from the MQTT broker.

        Args:
            event_type: The event type to subscribe to (e.g., 'camera.frame')
            handler: Callback function to handle matching events
        """
        # Map event type to MQTT topic
        topic = event_type_to_topic(event_type)

        with self._lock:
            # Add handler to local registry
            if handler not in self._subscribers[event_type]:
                self._subscribers[event_type].append(handler)
                logger.debug(
                    f"Subscribed handler to {event_type} "
                    f"(total: {len(self._subscribers[event_type])})"
                )

            # Subscribe to MQTT topic if not already
            if topic not in self._topic_subscriptions:
                self._topic_subscriptions.add(topic)
                self._client.subscribe(topic, self._on_message)
                logger.debug(f"Subscribed to MQTT topic: {topic}")

    def unsubscribe(self, event_type: str, handler: EventHandler) -> None:
        """Unsubscribe from events of a specific type.

        Args:
            event_type: The event type to unsubscribe from
            handler: The handler to remove
        """
        topic = event_type_to_topic(event_type)

        with self._lock:
            if handler in self._subscribers[event_type]:
                self._subscribers[event_type].remove(handler)
                logger.debug(
                    f"Unsubscribed from {event_type} "
                    f"(remaining: {len(self._subscribers[event_type])})"
                )

            # Unsubscribe from MQTT if no more handlers for this topic
            if not self._subscribers[event_type] and topic in self._topic_subscriptions:
                self._topic_subscriptions.remove(topic)
                self._client.unsubscribe(topic, self._on_message)

    def subscribe_category(
        self,
        category: TopicCategory,
        handler: EventHandler,
    ) -> None:
        """Subscribe to all events in a category using MQTT wildcard.

        This is more efficient than subscribing to individual topics
        when you want all events in a category.

        Args:
            category: The category to subscribe to (RAW, EVENTS, ACTIONS)
            handler: Callback function to handle all matching events
        """
        pattern = get_subscription_pattern(category)

        with self._lock:
            # Store under the pattern as a pseudo event-type
            pattern_key = f"__category_{category.value}"
            if handler not in self._subscribers[pattern_key]:
                self._subscribers[pattern_key].append(handler)

            if pattern not in self._topic_subscriptions:
                self._topic_subscriptions.add(pattern)
                self._client.subscribe(pattern, self._on_message)
                logger.debug(f"Subscribed to category pattern: {pattern}")

    def subscribe_all(self, handler: EventHandler) -> None:
        """Subscribe to all events using MQTT wildcard.

        Args:
            handler: Callback function to handle all events
        """
        pattern = get_subscription_pattern()  # Returns "all/#"

        with self._lock:
            pattern_key = "__all"
            if handler not in self._subscribers[pattern_key]:
                self._subscribers[pattern_key].append(handler)

            if pattern not in self._topic_subscriptions:
                self._topic_subscriptions.add(pattern)
                self._client.subscribe(pattern, self._on_message)
                logger.debug(f"Subscribed to all events: {pattern}")

    def get_queue_size(self) -> int:
        """Get the number of pending events.

        Note: With MQTT, this is not directly available. Returns 0.
        The MQTT broker handles queuing.

        Returns:
            int: Always 0 (MQTT broker manages the queue)
        """
        return 0

    def clear(self) -> None:
        """Clear is a no-op for MQTT - broker manages persistence."""
        logger.debug("Clear called on MQTT event bus (no-op)")

    def get_stats(self) -> dict[str, int]:
        """Get event bus statistics.

        Returns:
            Dictionary with published/received counts
        """
        return {
            "published": self._published_count,
            "received": self._message_count,
            "subscribers": sum(len(h) for h in self._subscribers.values()),
            "topic_subscriptions": len(self._topic_subscriptions),
        }

    def _on_message(self, topic: str, payload: bytes) -> None:
        """Handle incoming MQTT message.

        Args:
            topic: MQTT topic the message was received on
            payload: Raw message payload
        """
        self._message_count += 1

        # Deserialize the event
        try:
            event = self._serializer.deserialize(payload, topic)
        except Exception as e:
            logger.error(f"Failed to deserialize message from {topic}: {e}")
            return

        # Find matching handlers
        handlers: list[EventHandler] = []
        with self._lock:
            # Check specific event type handlers
            handlers.extend(self._subscribers.get(event.event_type, []))

            # Check category handlers
            if topic.startswith("all/raw/"):
                handlers.extend(self._subscribers.get("__category_raw", []))
            elif topic.startswith("all/events/"):
                handlers.extend(self._subscribers.get("__category_events", []))
            elif topic.startswith("all/actions/"):
                handlers.extend(self._subscribers.get("__category_actions", []))

            # Check "all" handlers
            handlers.extend(self._subscribers.get("__all", []))

        # Dispatch to handlers
        for handler in handlers:
            try:
                handler(event)
            except Exception as e:
                logger.error(
                    f"Error in event handler for {event.event_type}: {e}",
                    exc_info=True,
                )
