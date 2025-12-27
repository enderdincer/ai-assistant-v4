"""Heartbeat Service implementation.

Sends periodic heartbeat messages to MQTT to signal that the system is alive.
The heartbeat interval is configurable via the HEARTBEAT_INTERVAL_MS environment variable.
"""

import json
import os
import threading
import time
from dataclasses import dataclass

from ai_assistant.shared.logging import get_logger, LogLevel
from ai_assistant.shared.services import BaseService, ServiceConfig

logger = get_logger(__name__)

# Default heartbeat interval in milliseconds
DEFAULT_HEARTBEAT_INTERVAL_MS = 5000


@dataclass
class HeartbeatServiceConfig(ServiceConfig):
    """Configuration for Heartbeat Service.

    Attributes:
        heartbeat_interval_ms: Interval between heartbeats in milliseconds
    """

    heartbeat_interval_ms: int = DEFAULT_HEARTBEAT_INTERVAL_MS

    @classmethod
    def from_env(cls) -> "HeartbeatServiceConfig":
        """Create configuration from environment variables."""
        return cls(
            service_name="heartbeat-service",
            heartbeat_interval_ms=int(
                os.getenv("HEARTBEAT_INTERVAL_MS", str(DEFAULT_HEARTBEAT_INTERVAL_MS))
            ),
            log_level=LogLevel.DEBUG
            if os.getenv("DEBUG", "").lower() in ("1", "true", "yes")
            else LogLevel.INFO,
        )


class HeartbeatService(BaseService):
    """Service that sends periodic heartbeat messages.

    This service publishes heartbeat messages at a configurable interval
    to the all/events/heartbeat topic.
    """

    # Topic for heartbeat messages
    HEARTBEAT_TOPIC = "all/events/heartbeat"

    def __init__(self, config: HeartbeatServiceConfig) -> None:
        """Initialize the heartbeat service.

        Args:
            config: Service configuration
        """
        super().__init__(config)
        self._heartbeat_config = config
        self._heartbeat_thread: threading.Thread | None = None
        self._heartbeat_stop_event = threading.Event()

    def _setup(self) -> None:
        """Initialize the heartbeat service."""
        self._logger.info(
            f"Heartbeat service configured with interval: "
            f"{self._heartbeat_config.heartbeat_interval_ms}ms"
        )

    def _cleanup(self) -> None:
        """Clean up heartbeat service resources."""
        self._stop_heartbeat_thread()
        self._logger.info("Heartbeat service cleaned up")

    def start(self) -> None:
        """Start the heartbeat service."""
        super().start()
        self._start_heartbeat_thread()

    def stop(self) -> None:
        """Stop the heartbeat service."""
        self._stop_heartbeat_thread()
        super().stop()

    def _start_heartbeat_thread(self) -> None:
        """Start the background thread that sends heartbeats."""
        self._heartbeat_stop_event.clear()
        self._heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop,
            name=f"{self.service_name}-Heartbeat",
            daemon=True,
        )
        self._heartbeat_thread.start()
        self._logger.info("Heartbeat thread started")

    def _stop_heartbeat_thread(self) -> None:
        """Stop the heartbeat thread."""
        self._heartbeat_stop_event.set()
        if self._heartbeat_thread and self._heartbeat_thread.is_alive():
            self._heartbeat_thread.join(timeout=2.0)
        self._heartbeat_thread = None
        self._logger.debug("Heartbeat thread stopped")

    def _heartbeat_loop(self) -> None:
        """Background thread loop that sends periodic heartbeats."""
        interval_seconds = self._heartbeat_config.heartbeat_interval_ms / 1000.0

        while not self._heartbeat_stop_event.wait(interval_seconds):
            if self._running:
                self._send_heartbeat()

    def _send_heartbeat(self) -> None:
        """Send a single heartbeat message."""
        heartbeat_data = {
            "service": self.service_name,
            "machine_id": self.machine_id,
            "timestamp": time.time(),
            "interval_ms": self._heartbeat_config.heartbeat_interval_ms,
        }

        payload = json.dumps(heartbeat_data).encode("utf-8")

        if self._publish(self.HEARTBEAT_TOPIC, payload):
            self._logger.debug(f"Heartbeat sent to {self.HEARTBEAT_TOPIC}")
        else:
            self._logger.warning("Failed to send heartbeat")
