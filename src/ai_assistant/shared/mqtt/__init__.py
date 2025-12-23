"""MQTT integration for distributed event communication.

This module provides MQTT-based event routing, allowing the AI assistant
components to communicate via a shared MQTT broker.

Topic Hierarchy:
    all/                          # Root namespace
    ├── raw/                      # Raw sensory data
    │   ├── audio                 # Audio chunks
    │   ├── video                 # Camera frames
    │   └── text                  # Raw text
    ├── events/                   # Processed events
    │   ├── audio-transcribed     # STT results
    │   ├── vision-described      # VLM descriptions
    │   └── text-input            # User text input
    └── actions/                  # Action requests
        ├── speech                # TTS requests
        └── notification          # Notifications

Example Usage:
    from ai_assistant.shared.mqtt import MQTTEventBus, MQTTConfig

    # Create event bus
    config = MQTTConfig(host="localhost", port=1883)
    bus = MQTTEventBus(config)

    # Start the bus
    bus.initialize()
    bus.start()

    # Subscribe to events
    def handle_transcription(event):
        print(f"Transcribed: {event.data['text']}")

    bus.subscribe("audio.transcription", handle_transcription)

    # Publish events
    from ai_assistant.shared.events import AudioTranscriptionEvent
    event = AudioTranscriptionEvent.create(...)
    bus.publish(event)

    # Stop when done
    bus.stop()
"""

from ai_assistant.shared.mqtt.config import MQTTConfig, MQTTQoS
from ai_assistant.shared.mqtt.topics import (
    TopicCategory,
    TopicDefinition,
    Topics,
    event_type_to_topic,
    topic_to_event_type,
    get_subscription_pattern,
)
from ai_assistant.shared.mqtt.serialization import (
    JSONSerializer,
    ActionMessage,
    ActionMessages,
)
from ai_assistant.shared.mqtt.client import MQTTClient
from ai_assistant.shared.mqtt.event_bus import MQTTEventBus

__all__ = [
    # Configuration
    "MQTTConfig",
    "MQTTQoS",
    # Topics
    "TopicCategory",
    "TopicDefinition",
    "Topics",
    "event_type_to_topic",
    "topic_to_event_type",
    "get_subscription_pattern",
    # Serialization
    "JSONSerializer",
    "ActionMessage",
    "ActionMessages",
    # Client and Event Bus
    "MQTTClient",
    "MQTTEventBus",
]
