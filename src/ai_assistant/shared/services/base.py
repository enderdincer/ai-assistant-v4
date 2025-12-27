"""Base service class for microservices architecture.

All services inherit from BaseService which provides:
- MQTT connection management
- Lifecycle management (initialize, start, stop)
- Health check publishing
- Graceful shutdown handling
- Signal handling
"""

import os
import signal
import socket
import sys
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from ai_assistant.shared.logging import get_logger, setup_logging, LogConfig, LogLevel
from ai_assistant.shared.mqtt import MQTTClient, MQTTConfig


@dataclass
class ServiceConfig:
    """Base configuration for all services.

    Attributes:
        service_name: Unique name for this service instance
        machine_id: Identifier for the machine running this service
        mqtt_host: MQTT broker host
        mqtt_port: MQTT broker port
        mqtt_username: MQTT authentication username
        mqtt_password: MQTT authentication password
        log_level: Logging level
        health_check_interval: Seconds between health check publishes
    """

    service_name: str
    machine_id: str = field(default_factory=lambda: os.getenv("MACHINE_ID", socket.gethostname()))
    mqtt_host: str = field(default_factory=lambda: os.getenv("MQTT_HOST", "localhost"))
    mqtt_port: int = field(default_factory=lambda: int(os.getenv("MQTT_PORT", "1883")))
    mqtt_username: Optional[str] = field(default_factory=lambda: os.getenv("MQTT_USERNAME"))
    mqtt_password: Optional[str] = field(default_factory=lambda: os.getenv("MQTT_PASSWORD"))
    log_level: LogLevel = field(
        default_factory=lambda: LogLevel.DEBUG
        if os.getenv("DEBUG", "").lower() in ("1", "true", "yes")
        else LogLevel.INFO
    )
    health_check_interval: float = field(
        default_factory=lambda: float(os.getenv("HEALTH_CHECK_INTERVAL", "1"))
    )

    def to_mqtt_config(self) -> MQTTConfig:
        """Convert to MQTTConfig for the MQTT client."""
        return MQTTConfig(
            host=self.mqtt_host,
            port=self.mqtt_port,
            username=self.mqtt_username,
            password=self.mqtt_password,
            client_id=f"{self.service_name}-{self.machine_id}",
        )


class BaseService(ABC):
    """Abstract base class for all microservices.

    Provides common functionality for:
    - MQTT connection and message handling
    - Service lifecycle management
    - Health check publishing
    - Graceful shutdown

    Subclasses must implement:
    - _setup(): Called during initialization to set up subscriptions
    - _cleanup(): Called during shutdown to clean up resources

    Example:
        ```python
        class MyService(BaseService):
            def __init__(self, config: ServiceConfig):
                super().__init__(config)

            def _setup(self) -> None:
                self._subscribe("all/events/some-topic", self._handle_event)

            def _cleanup(self) -> None:
                pass

            def _handle_event(self, topic: str, payload: bytes) -> None:
                # Process the event
                pass
        ```
    """

    def __init__(self, config: ServiceConfig) -> None:
        """Initialize the base service.

        Args:
            config: Service configuration
        """
        self._config = config
        self._logger = get_logger(f"{__name__}.{config.service_name}")

        # Setup logging
        setup_logging(
            LogConfig(
                level=config.log_level,
                colored_console=True,
            )
        )

        # MQTT client
        self._mqtt_client: Optional[MQTTClient] = None
        self._mqtt_config = config.to_mqtt_config()

        # Lifecycle state
        self._running = False
        self._initialized = False
        self._lock = threading.RLock()

        # Health check thread
        self._health_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        # Track subscriptions for cleanup
        self._subscriptions: list[tuple[str, Callable[[str, bytes], None]]] = []

    @property
    def service_name(self) -> str:
        """Get the service name."""
        return self._config.service_name

    @property
    def machine_id(self) -> str:
        """Get the machine ID."""
        return self._config.machine_id

    @property
    def is_running(self) -> bool:
        """Check if the service is running."""
        return self._running

    @property
    def is_initialized(self) -> bool:
        """Check if the service is initialized."""
        return self._initialized

    def initialize(self) -> None:
        """Initialize the service.

        Creates MQTT client and calls subclass _setup().
        """
        with self._lock:
            if self._initialized:
                self._logger.warning("Service already initialized")
                return

            self._logger.info(f"Initializing {self.service_name}...")

            # Create and initialize MQTT client
            self._mqtt_client = MQTTClient(self._mqtt_config)
            self._mqtt_client.initialize()

            # Call subclass setup
            self._setup()

            self._initialized = True
            self._logger.info(f"{self.service_name} initialized")

    def start(self) -> None:
        """Start the service.

        Connects to MQTT broker and starts health check thread.
        """
        with self._lock:
            if self._running:
                raise RuntimeError(f"{self.service_name} is already running")

            if not self._initialized:
                raise RuntimeError(f"{self.service_name} not initialized. Call initialize() first.")

            self._logger.info(f"Starting {self.service_name}...")

            # Start MQTT client (we know it's not None since we checked _initialized)
            assert self._mqtt_client is not None
            self._mqtt_client.start()

            # Wait for connection
            for _ in range(50):  # Up to 5 seconds
                if self._mqtt_client.is_connected():
                    break
                time.sleep(0.1)

            if not self._mqtt_client.is_connected():
                self._logger.warning("MQTT broker not connected yet, will retry")

            self._running = True
            self._stop_event.clear()

            # Start health check thread
            if self._config.health_check_interval > 0:
                self._health_thread = threading.Thread(
                    target=self._health_check_loop,
                    name=f"{self.service_name}-Health",
                    daemon=True,
                )
                self._health_thread.start()

            # Publish startup health
            self._publish_health("started")

            self._logger.info(f"{self.service_name} started")

    def stop(self) -> None:
        """Stop the service gracefully."""
        with self._lock:
            if not self._running:
                return

            self._logger.info(f"Stopping {self.service_name}...")

            self._running = False
            self._stop_event.set()

            # Publish shutdown health
            self._publish_health("stopping")

            # Call subclass cleanup
            self._cleanup()

            # Stop health check thread
            if self._health_thread and self._health_thread.is_alive():
                self._health_thread.join(timeout=2.0)

            # Stop MQTT client
            if self._mqtt_client:
                self._mqtt_client.stop()

            self._initialized = False
            self._logger.info(f"{self.service_name} stopped")

    def run_forever(self) -> None:
        """Run the service until interrupted.

        Handles SIGINT and SIGTERM for graceful shutdown.
        """
        self.initialize()
        self.start()

        def signal_handler(signum: int, frame: Any) -> None:
            self._logger.info(f"Received signal {signum}, shutting down...")
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

    @abstractmethod
    def _setup(self) -> None:
        """Set up the service (called during initialization).

        Subclasses should set up MQTT subscriptions here.
        """
        pass

    @abstractmethod
    def _cleanup(self) -> None:
        """Clean up service resources (called during shutdown).

        Subclasses should release any resources here.
        """
        pass

    def _subscribe(self, topic: str, callback: Callable[[str, bytes], None]) -> None:
        """Subscribe to an MQTT topic.

        Args:
            topic: MQTT topic pattern (supports wildcards)
            callback: Function to call when message received (topic, payload)
        """
        if self._mqtt_client is None:
            raise RuntimeError("MQTT client not initialized")

        self._mqtt_client.subscribe(topic, callback)
        self._subscriptions.append((topic, callback))
        self._logger.debug(f"Subscribed to: {topic}")

    def _unsubscribe(self, topic: str, callback: Callable[[str, bytes], None]) -> None:
        """Unsubscribe from an MQTT topic.

        Args:
            topic: MQTT topic to unsubscribe from
            callback: The callback to remove
        """
        if self._mqtt_client is None:
            return

        self._mqtt_client.unsubscribe(topic, callback)
        if (topic, callback) in self._subscriptions:
            self._subscriptions.remove((topic, callback))
        self._logger.debug(f"Unsubscribed from: {topic}")

    def _publish(self, topic: str, payload: bytes, retain: bool = False) -> bool:
        """Publish a message to an MQTT topic.

        Args:
            topic: MQTT topic to publish to
            payload: Message payload as bytes
            retain: Whether to retain the message on the broker

        Returns:
            True if message was queued for sending
        """
        if self._mqtt_client is None or not self._running:
            self._logger.warning("Cannot publish: service not running")
            return False

        return self._mqtt_client.publish(topic, payload, retain=retain)

    def _publish_health(self, status: str) -> None:
        """Publish a health check message.

        Args:
            status: Current service status
        """
        import json

        health_data = {
            "service": self.service_name,
            "machine_id": self.machine_id,
            "status": status,
            "timestamp": time.time(),
        }

        topic = f"all/system/health/{self.service_name}"
        payload = json.dumps(health_data).encode("utf-8")

        self._publish(topic, payload, retain=True)

    def _health_check_loop(self) -> None:
        """Background thread for periodic health checks."""
        while not self._stop_event.wait(self._config.health_check_interval):
            if self._running:
                self._publish_health("healthy")
