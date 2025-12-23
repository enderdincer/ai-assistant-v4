"""MQTT configuration settings."""

import os
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional


class MQTTQoS(IntEnum):
    """MQTT Quality of Service levels."""

    AT_MOST_ONCE = 0  # Fire and forget (fastest, may lose messages)
    AT_LEAST_ONCE = 1  # Guaranteed delivery (may have duplicates)
    EXACTLY_ONCE = 2  # Guaranteed exactly once (slowest)


@dataclass
class MQTTConfig:
    """Configuration for MQTT client connection.

    Attributes:
        host: MQTT broker hostname
        port: MQTT broker port
        client_id: Unique client identifier (auto-generated if None)
        username: Optional username for authentication
        password: Optional password for authentication
        keepalive: Keepalive interval in seconds
        clean_session: Whether to use clean session
        qos: Default QoS level for publishing
        reconnect_on_failure: Whether to auto-reconnect on connection loss
        reconnect_delay_min: Minimum delay between reconnection attempts (seconds)
        reconnect_delay_max: Maximum delay between reconnection attempts (seconds)
        connect_timeout: Connection timeout in seconds
    """

    host: str = field(default_factory=lambda: os.getenv("MQTT_HOST", "localhost"))
    port: int = field(default_factory=lambda: int(os.getenv("MQTT_PORT", "1883")))
    client_id: Optional[str] = None
    username: Optional[str] = field(default_factory=lambda: os.getenv("MQTT_USERNAME"))
    password: Optional[str] = field(default_factory=lambda: os.getenv("MQTT_PASSWORD"))
    keepalive: int = 60
    clean_session: bool = True
    qos: MQTTQoS = MQTTQoS.AT_LEAST_ONCE
    reconnect_on_failure: bool = True
    reconnect_delay_min: float = 1.0
    reconnect_delay_max: float = 120.0
    connect_timeout: float = 10.0

    @classmethod
    def from_env(cls) -> "MQTTConfig":
        """Create configuration from environment variables.

        Environment variables:
            MQTT_HOST: Broker hostname (default: localhost)
            MQTT_PORT: Broker port (default: 1883)
            MQTT_CLIENT_ID: Client ID (default: auto-generated)
            MQTT_USERNAME: Authentication username
            MQTT_PASSWORD: Authentication password
            MQTT_KEEPALIVE: Keepalive interval (default: 60)
            MQTT_QOS: Default QoS level (default: 1)

        Returns:
            MQTTConfig: Configuration instance
        """
        return cls(
            host=os.getenv("MQTT_HOST", "localhost"),
            port=int(os.getenv("MQTT_PORT", "1883")),
            client_id=os.getenv("MQTT_CLIENT_ID"),
            username=os.getenv("MQTT_USERNAME"),
            password=os.getenv("MQTT_PASSWORD"),
            keepalive=int(os.getenv("MQTT_KEEPALIVE", "60")),
            qos=MQTTQoS(int(os.getenv("MQTT_QOS", "1"))),
        )
