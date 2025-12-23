"""MQTT client wrapper with connection management and auto-reconnection."""

import threading
import time
import uuid
from collections.abc import Callable
from typing import Optional

from ai_assistant.shared.logging import get_logger
from ai_assistant.shared.mqtt.config import MQTTConfig, MQTTQoS

logger = get_logger(__name__)

# Type alias for message callback
MessageCallback = Callable[[str, bytes], None]


class MQTTClient:
    """MQTT client wrapper with connection management.

    Provides a simplified interface to paho-mqtt with:
    - Automatic reconnection with exponential backoff
    - Thread-safe operations
    - Connection state tracking
    - Lifecycle management (ILifecycle compatible)
    """

    def __init__(self, config: Optional[MQTTConfig] = None):
        """Initialize the MQTT client.

        Args:
            config: MQTT configuration. If None, loads from environment.
        """
        # Import here to make paho-mqtt optional at module level
        try:
            import paho.mqtt.client as mqtt
        except ImportError as e:
            raise ImportError(
                "paho-mqtt is required for MQTT support. Install with: pip install paho-mqtt"
            ) from e

        self._mqtt = mqtt
        self._config = config or MQTTConfig.from_env()
        self._client: Optional[mqtt.Client] = None
        self._connected = False
        self._running = False
        self._lock = threading.RLock()
        self._subscriptions: dict[str, list[MessageCallback]] = {}
        self._reconnect_thread: Optional[threading.Thread] = None

    def initialize(self) -> None:
        """Initialize the MQTT client (create client instance)."""
        client_id = self._config.client_id or f"ai-assistant-{uuid.uuid4().hex[:8]}"

        # paho-mqtt 2.0+ uses CallbackAPIVersion
        try:
            self._client = self._mqtt.Client(
                callback_api_version=self._mqtt.CallbackAPIVersion.VERSION2,
                client_id=client_id,
                clean_session=self._config.clean_session,
            )
        except (AttributeError, TypeError):
            # Fallback for older paho-mqtt versions
            self._client = self._mqtt.Client(
                client_id=client_id,
                clean_session=self._config.clean_session,
            )

        # Set up callbacks
        self._client.on_connect = self._on_connect
        self._client.on_disconnect = self._on_disconnect
        self._client.on_message = self._on_message

        # Configure authentication if provided
        if self._config.username:
            self._client.username_pw_set(
                self._config.username,
                self._config.password,
            )

        logger.info(f"MQTT client initialized with ID: {client_id}")

    def start(self) -> None:
        """Start the MQTT client and connect to broker."""
        with self._lock:
            if self._running:
                raise RuntimeError("MQTT client is already running")

            if self._client is None:
                raise RuntimeError("MQTT client not initialized. Call initialize() first.")

            self._running = True

        self._connect()

        # Start the network loop in a background thread
        self._client.loop_start()
        logger.info("MQTT client started")

    def stop(self) -> None:
        """Stop the MQTT client gracefully."""
        with self._lock:
            if not self._running:
                return

            self._running = False

        if self._client:
            self._client.loop_stop()
            self._client.disconnect()

        logger.info("MQTT client stopped")

    def is_running(self) -> bool:
        """Check if the client is running."""
        return self._running

    def is_connected(self) -> bool:
        """Check if connected to the broker."""
        return self._connected

    def publish(
        self,
        topic: str,
        payload: bytes,
        qos: Optional[MQTTQoS] = None,
        retain: bool = False,
    ) -> bool:
        """Publish a message to a topic.

        Args:
            topic: MQTT topic to publish to
            payload: Message payload as bytes
            qos: Quality of Service level (uses default if None)
            retain: Whether to retain the message on the broker

        Returns:
            True if message was queued for sending, False otherwise
        """
        if not self._running or self._client is None:
            logger.warning("Cannot publish: client not running")
            return False

        qos_level = (qos or self._config.qos).value

        try:
            result = self._client.publish(topic, payload, qos=qos_level, retain=retain)
            if result.rc != self._mqtt.MQTT_ERR_SUCCESS:
                logger.error(f"Failed to publish to {topic}: rc={result.rc}")
                return False
            return True
        except Exception as e:
            logger.error(f"Error publishing to {topic}: {e}")
            return False

    def subscribe(
        self,
        topic: str,
        callback: MessageCallback,
        qos: Optional[MQTTQoS] = None,
    ) -> None:
        """Subscribe to a topic with a callback.

        Args:
            topic: MQTT topic pattern (supports wildcards: + and #)
            callback: Function to call when message received (topic, payload)
            qos: Quality of Service level (uses default if None)
        """
        with self._lock:
            if topic not in self._subscriptions:
                self._subscriptions[topic] = []
            if callback not in self._subscriptions[topic]:
                self._subscriptions[topic].append(callback)

        if self._connected and self._client:
            qos_level = (qos or self._config.qos).value
            self._client.subscribe(topic, qos=qos_level)
            logger.debug(f"Subscribed to topic: {topic}")

    def unsubscribe(self, topic: str, callback: MessageCallback) -> None:
        """Unsubscribe a callback from a topic.

        Args:
            topic: MQTT topic to unsubscribe from
            callback: The callback to remove
        """
        with self._lock:
            if topic in self._subscriptions:
                if callback in self._subscriptions[topic]:
                    self._subscriptions[topic].remove(callback)
                if not self._subscriptions[topic]:
                    del self._subscriptions[topic]
                    if self._connected and self._client:
                        self._client.unsubscribe(topic)
                        logger.debug(f"Unsubscribed from topic: {topic}")

    def _connect(self) -> None:
        """Attempt to connect to the broker."""
        if self._client is None:
            return

        try:
            logger.info(f"Connecting to MQTT broker at {self._config.host}:{self._config.port}")
            self._client.connect(
                self._config.host,
                self._config.port,
                keepalive=self._config.keepalive,
            )
        except Exception as e:
            logger.error(f"Failed to connect to MQTT broker: {e}")
            if self._config.reconnect_on_failure and self._running:
                self._schedule_reconnect()

    def _schedule_reconnect(self) -> None:
        """Schedule a reconnection attempt with exponential backoff."""
        if self._reconnect_thread and self._reconnect_thread.is_alive():
            return

        def reconnect_loop() -> None:
            delay = self._config.reconnect_delay_min
            while self._running and not self._connected:
                logger.info(f"Reconnecting in {delay:.1f} seconds...")
                time.sleep(delay)
                if self._running and not self._connected:
                    self._connect()
                delay = min(delay * 2, self._config.reconnect_delay_max)

        self._reconnect_thread = threading.Thread(
            target=reconnect_loop,
            name="MQTTClient-Reconnect",
            daemon=True,
        )
        self._reconnect_thread.start()

    def _on_connect(
        self,
        client: "paho.mqtt.client.Client",  # noqa: F821
        userdata: object,
        flags: dict,
        reason_code: object,
        properties: object = None,
    ) -> None:
        """Callback when connected to broker."""
        # Handle both paho-mqtt 1.x and 2.x callback signatures
        self._connected = True
        logger.info(f"Connected to MQTT broker at {self._config.host}:{self._config.port}")

        # Re-subscribe to all topics
        with self._lock:
            for topic in self._subscriptions:
                qos_level = self._config.qos.value
                self._client.subscribe(topic, qos=qos_level)
                logger.debug(f"Re-subscribed to topic: {topic}")

    def _on_disconnect(
        self,
        client: "paho.mqtt.client.Client",  # noqa: F821
        userdata: object,
        disconnect_flags: object = None,
        reason_code: object = None,
        properties: object = None,
    ) -> None:
        """Callback when disconnected from broker."""
        self._connected = False
        logger.warning("Disconnected from MQTT broker")

        if self._config.reconnect_on_failure and self._running:
            self._schedule_reconnect()

    def _on_message(
        self,
        client: "paho.mqtt.client.Client",  # noqa: F821
        userdata: object,
        message: "paho.mqtt.client.MQTTMessage",  # noqa: F821
    ) -> None:
        """Callback when message received."""
        topic = message.topic
        payload = message.payload

        # Find matching subscriptions (handle wildcards)
        callbacks: list[MessageCallback] = []
        with self._lock:
            for pattern, handlers in self._subscriptions.items():
                if self._topic_matches(pattern, topic):
                    callbacks.extend(handlers)

        # Invoke callbacks
        for callback in callbacks:
            try:
                callback(topic, payload)
            except Exception as e:
                logger.error(f"Error in message callback for {topic}: {e}", exc_info=True)

    def _topic_matches(self, pattern: str, topic: str) -> bool:
        """Check if a topic matches a subscription pattern.

        Args:
            pattern: Subscription pattern (may contain + and # wildcards)
            topic: Actual topic to match

        Returns:
            True if topic matches pattern
        """
        pattern_parts = pattern.split("/")
        topic_parts = topic.split("/")

        i = 0
        for i, p in enumerate(pattern_parts):
            if p == "#":
                # Multi-level wildcard matches everything after
                return True
            if i >= len(topic_parts):
                return False
            if p == "+":
                # Single-level wildcard matches any single level
                continue
            if p != topic_parts[i]:
                return False

        # Pattern fully consumed; topic must also be fully consumed
        return i + 1 == len(topic_parts)
