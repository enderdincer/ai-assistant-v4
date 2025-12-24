"""MQTT topic hierarchy and mapping utilities.

Topic Structure:
    all/                              # Root namespace
    ├── raw/                          # Raw sensory data from input sources
    │   ├── audio/                    # Audio chunks from microphone
    │   │   └── {machine_id}          # e.g., all/raw/audio/macbook-pro-1
    │   ├── video/                    # Camera frames
    │   │   └── {machine_id}
    │   └── text                      # Raw text input
    ├── events/                       # Processed/enriched events
    │   ├── audio-transcribed         # Speech-to-text results
    │   ├── vision-described          # Vision model descriptions
    │   ├── text-input                # User text input
    │   ├── assistant-response        # Assistant text responses
    │   └── speaker-activity          # TTS start/end notifications
    ├── actions/                      # Action requests for actuators
    │   ├── speech                    # TTS requests (voice, text)
    │   ├── display                   # Display/UI actions
    │   └── notification              # Notification actions
    ├── memory/                       # Memory service communication
    │   ├── query                     # Context retrieval requests
    │   ├── response                  # Context retrieval responses
    │   ├── store                     # Store conversation/facts requests
    │   └── facts                     # Extracted facts
    └── system/                       # System/control messages
        ├── health/                   # Service health checks
        │   └── {service_name}
        ├── config                    # Configuration updates
        └── logs                      # Distributed logging

Mapping from current event types to MQTT topics:
    - camera.frame        -> all/raw/video/{machine_id}
    - audio.sample        -> all/raw/audio/{machine_id}
    - text.input          -> all/events/text-input
    - audio.transcription -> all/events/audio-transcribed
    - vision.description  -> all/events/vision-described
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class TopicCategory(Enum):
    """Category of MQTT topics."""

    RAW = "raw"
    EVENTS = "events"
    ACTIONS = "actions"
    MEMORY = "memory"
    SYSTEM = "system"


@dataclass(frozen=True)
class TopicDefinition:
    """Definition of an MQTT topic with metadata."""

    category: TopicCategory
    name: str
    description: str
    event_type: Optional[str] = None  # Corresponding internal event type
    parameterized: bool = False  # Whether topic includes a parameter like {machine_id}

    @property
    def topic(self) -> str:
        """Get the full MQTT topic path (without parameters)."""
        return f"all/{self.category.value}/{self.name}"

    def with_param(self, param: str) -> str:
        """Get the full MQTT topic path with a parameter value.

        Args:
            param: Parameter value (e.g., machine_id)

        Returns:
            Full topic path with parameter
        """
        if self.parameterized:
            return f"all/{self.category.value}/{self.name}/{param}"
        return self.topic

    @property
    def subscription_pattern(self) -> str:
        """Get subscription pattern for this topic (with wildcards if parameterized)."""
        if self.parameterized:
            return f"all/{self.category.value}/{self.name}/#"
        return self.topic


# Standard topic definitions
class Topics:
    """Standard MQTT topic definitions for the AI assistant."""

    # =========================================================================
    # Raw sensory data topics
    # =========================================================================

    RAW_AUDIO = TopicDefinition(
        category=TopicCategory.RAW,
        name="audio",
        description="Raw audio chunks from microphone input (per machine)",
        event_type="audio.sample",
        parameterized=True,
    )

    RAW_VIDEO = TopicDefinition(
        category=TopicCategory.RAW,
        name="video",
        description="Raw video frames from camera input (per machine)",
        event_type="camera.frame",
        parameterized=True,
    )

    RAW_TEXT = TopicDefinition(
        category=TopicCategory.RAW,
        name="text",
        description="Raw text input stream",
        event_type=None,
    )

    # =========================================================================
    # Processed event topics
    # =========================================================================

    EVENT_AUDIO_TRANSCRIBED = TopicDefinition(
        category=TopicCategory.EVENTS,
        name="audio-transcribed",
        description="Speech-to-text transcription results",
        event_type="audio.transcription",
    )

    EVENT_VISION_DESCRIBED = TopicDefinition(
        category=TopicCategory.EVENTS,
        name="vision-described",
        description="Vision model descriptions of camera frames",
        event_type="vision.description",
    )

    EVENT_TEXT_INPUT = TopicDefinition(
        category=TopicCategory.EVENTS,
        name="text-input",
        description="User text input events",
        event_type="text.input",
    )

    EVENT_ASSISTANT_RESPONSE = TopicDefinition(
        category=TopicCategory.EVENTS,
        name="assistant-response",
        description="Assistant text responses",
        event_type="assistant.response",
    )

    EVENT_SPEAKER_ACTIVITY = TopicDefinition(
        category=TopicCategory.EVENTS,
        name="speaker-activity",
        description="TTS start/end notifications for echo prevention",
        event_type="speaker.activity",
    )

    # =========================================================================
    # Action topics
    # =========================================================================

    ACTION_SPEECH = TopicDefinition(
        category=TopicCategory.ACTIONS,
        name="speech",
        description="Text-to-speech requests",
        event_type=None,
    )

    ACTION_DISPLAY = TopicDefinition(
        category=TopicCategory.ACTIONS,
        name="display",
        description="Display/UI action requests",
        event_type=None,
    )

    ACTION_NOTIFICATION = TopicDefinition(
        category=TopicCategory.ACTIONS,
        name="notification",
        description="Notification action requests",
        event_type=None,
    )

    # =========================================================================
    # Memory topics
    # =========================================================================

    MEMORY_QUERY = TopicDefinition(
        category=TopicCategory.MEMORY,
        name="query",
        description="Context retrieval requests to memory service",
        event_type=None,
    )

    MEMORY_RESPONSE = TopicDefinition(
        category=TopicCategory.MEMORY,
        name="response",
        description="Context retrieval responses from memory service",
        event_type=None,
    )

    MEMORY_STORE = TopicDefinition(
        category=TopicCategory.MEMORY,
        name="store",
        description="Requests to store conversation/facts",
        event_type=None,
    )

    MEMORY_FACTS = TopicDefinition(
        category=TopicCategory.MEMORY,
        name="facts",
        description="Extracted facts from extraction service",
        event_type=None,
    )

    # =========================================================================
    # System topics
    # =========================================================================

    SYSTEM_HEALTH = TopicDefinition(
        category=TopicCategory.SYSTEM,
        name="health",
        description="Service health check messages",
        event_type=None,
        parameterized=True,  # all/system/health/{service_name}
    )

    SYSTEM_CONFIG = TopicDefinition(
        category=TopicCategory.SYSTEM,
        name="config",
        description="Configuration update messages",
        event_type=None,
    )

    SYSTEM_LOGS = TopicDefinition(
        category=TopicCategory.SYSTEM,
        name="logs",
        description="Distributed logging messages",
        event_type=None,
    )


# =========================================================================
# Mapping utilities
# =========================================================================

# Mapping from internal event types to MQTT topics (for non-parameterized topics)
EVENT_TYPE_TO_TOPIC: dict[str, str] = {
    "text.input": Topics.EVENT_TEXT_INPUT.topic,
    "audio.transcription": Topics.EVENT_AUDIO_TRANSCRIBED.topic,
    "vision.description": Topics.EVENT_VISION_DESCRIBED.topic,
    "assistant.response": Topics.EVENT_ASSISTANT_RESPONSE.topic,
    "speaker.activity": Topics.EVENT_SPEAKER_ACTIVITY.topic,
}

# Mapping from MQTT topics to internal event types
TOPIC_TO_EVENT_TYPE: dict[str, str] = {v: k for k, v in EVENT_TYPE_TO_TOPIC.items()}


def event_type_to_topic(event_type: str, machine_id: Optional[str] = None) -> str:
    """Convert an internal event type to an MQTT topic.

    Args:
        event_type: Internal event type (e.g., 'camera.frame')
        machine_id: Machine ID for parameterized topics (e.g., 'macbook-pro-1')

    Returns:
        MQTT topic path (e.g., 'all/raw/video/macbook-pro-1')

    Raises:
        ValueError: If event type has no topic mapping
    """
    # Handle parameterized topics
    if event_type == "audio.sample":
        if machine_id:
            return Topics.RAW_AUDIO.with_param(machine_id)
        return Topics.RAW_AUDIO.subscription_pattern

    if event_type == "camera.frame":
        if machine_id:
            return Topics.RAW_VIDEO.with_param(machine_id)
        return Topics.RAW_VIDEO.subscription_pattern

    # Handle standard topics
    topic = EVENT_TYPE_TO_TOPIC.get(event_type)
    if topic is not None:
        return topic

    # For unknown event types, create a dynamic topic under events/
    parts = event_type.split(".")
    if len(parts) == 2:
        return f"all/events/{parts[0]}-{parts[1]}"

    raise ValueError(f"Cannot map event type to topic: {event_type}")


def topic_to_event_type(topic: str) -> Optional[str]:
    """Convert an MQTT topic to an internal event type.

    Args:
        topic: MQTT topic path (e.g., 'all/raw/video/machine-1')

    Returns:
        Internal event type (e.g., 'camera.frame') or None if unknown
    """
    # Check parameterized topics first
    if topic.startswith("all/raw/audio/"):
        return "audio.sample"
    if topic.startswith("all/raw/video/"):
        return "camera.frame"
    if topic.startswith("all/system/health/"):
        return "system.health"

    return TOPIC_TO_EVENT_TYPE.get(topic)


def get_subscription_pattern(category: Optional[TopicCategory] = None) -> str:
    """Get an MQTT subscription pattern for a category.

    Args:
        category: Optional category to filter by. If None, subscribes to all.

    Returns:
        MQTT subscription pattern with wildcards

    Examples:
        get_subscription_pattern() -> 'all/#'
        get_subscription_pattern(TopicCategory.RAW) -> 'all/raw/#'
        get_subscription_pattern(TopicCategory.EVENTS) -> 'all/events/#'
    """
    if category is None:
        return "all/#"
    return f"all/{category.value}/#"


def extract_machine_id(topic: str) -> Optional[str]:
    """Extract machine ID from a parameterized topic.

    Args:
        topic: Full MQTT topic path (e.g., 'all/raw/audio/macbook-pro-1')

    Returns:
        Machine ID if topic is parameterized, None otherwise
    """
    parts = topic.split("/")
    if len(parts) >= 4:
        if parts[0] == "all" and parts[1] in ("raw", "system"):
            return parts[3] if len(parts) > 3 else None
    return None
