"""MQTT topic hierarchy and mapping utilities.

Topic Structure:
    all/                          # Root namespace
    ├── raw/                      # Raw sensory data from input sources
    │   ├── audio                 # Audio chunks from microphone
    │   ├── video                 # Camera frames
    │   └── text                  # Raw text input
    ├── events/                   # Processed/enriched events from processors
    │   ├── audio-transcribed     # Speech-to-text results
    │   ├── vision-described      # Vision model descriptions
    │   └── text-input            # Processed text input
    └── actions/                  # Action requests for actuators
        ├── speech                # TTS requests (voice, text)
        ├── display               # Display/UI actions
        └── notification          # Notification actions

Mapping from current event types to MQTT topics:
    - camera.frame      -> all/raw/video
    - audio.sample      -> all/raw/audio
    - text.input        -> all/events/text-input (user input is already "processed")
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


@dataclass(frozen=True)
class TopicDefinition:
    """Definition of an MQTT topic with metadata."""

    category: TopicCategory
    name: str
    description: str
    event_type: Optional[str] = None  # Corresponding internal event type

    @property
    def topic(self) -> str:
        """Get the full MQTT topic path."""
        return f"all/{self.category.value}/{self.name}"


# Standard topic definitions
class Topics:
    """Standard MQTT topic definitions for the AI assistant."""

    # Raw sensory data topics
    RAW_AUDIO = TopicDefinition(
        category=TopicCategory.RAW,
        name="audio",
        description="Raw audio chunks from microphone input",
        event_type="audio.sample",
    )

    RAW_VIDEO = TopicDefinition(
        category=TopicCategory.RAW,
        name="video",
        description="Raw video frames from camera input",
        event_type="camera.frame",
    )

    RAW_TEXT = TopicDefinition(
        category=TopicCategory.RAW,
        name="text",
        description="Raw text input stream",
        event_type=None,  # No direct mapping
    )

    # Processed event topics
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

    # Action topics
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


# Mapping from internal event types to MQTT topics
EVENT_TYPE_TO_TOPIC: dict[str, str] = {
    "audio.sample": Topics.RAW_AUDIO.topic,
    "camera.frame": Topics.RAW_VIDEO.topic,
    "text.input": Topics.EVENT_TEXT_INPUT.topic,
    "audio.transcription": Topics.EVENT_AUDIO_TRANSCRIBED.topic,
    "vision.description": Topics.EVENT_VISION_DESCRIBED.topic,
}

# Mapping from MQTT topics to internal event types
TOPIC_TO_EVENT_TYPE: dict[str, str] = {v: k for k, v in EVENT_TYPE_TO_TOPIC.items()}


def event_type_to_topic(event_type: str) -> str:
    """Convert an internal event type to an MQTT topic.

    Args:
        event_type: Internal event type (e.g., 'camera.frame')

    Returns:
        MQTT topic path (e.g., 'all/raw/video')

    Raises:
        ValueError: If event type has no topic mapping
    """
    topic = EVENT_TYPE_TO_TOPIC.get(event_type)
    if topic is None:
        # For unknown event types, create a dynamic topic under events/
        # Format: all/events/{domain}/{action} from {domain}.{action}
        parts = event_type.split(".")
        if len(parts) == 2:
            return f"all/events/{parts[0]}-{parts[1]}"
        raise ValueError(f"Cannot map event type to topic: {event_type}")
    return topic


def topic_to_event_type(topic: str) -> Optional[str]:
    """Convert an MQTT topic to an internal event type.

    Args:
        topic: MQTT topic path (e.g., 'all/raw/video')

    Returns:
        Internal event type (e.g., 'camera.frame') or None if unknown
    """
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
