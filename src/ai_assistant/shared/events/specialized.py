"""Specialized event types for different input sources."""

from dataclasses import dataclass
from typing import Any
import numpy as np
import numpy.typing as npt
from ai_assistant.shared.events.event import Event
from ai_assistant.shared.interfaces import EventPriority


@dataclass(frozen=True)
class CameraFrameEvent(Event):
    """Event containing a camera frame."""

    @staticmethod
    def create(
        source: str,
        frame: npt.NDArray[Any],
        frame_number: int,
        priority: EventPriority = EventPriority.NORMAL,
    ) -> "CameraFrameEvent":
        """Create a camera frame event.

        Args:
            source: Camera source identifier
            frame: The camera frame (numpy array)
            frame_number: Sequential frame number
            priority: Event priority

        Returns:
            CameraFrameEvent: The created event
        """
        data = {
            "frame": frame,
            "frame_number": frame_number,
            "shape": frame.shape,
            "dtype": str(frame.dtype),
        }
        return CameraFrameEvent(
            event_type="camera.frame",
            source=source,
            data=data,
            priority=priority,
        )


@dataclass(frozen=True)
class TextInputEvent(Event):
    """Event containing text input."""

    @staticmethod
    def create(
        source: str,
        text: str,
        priority: EventPriority = EventPriority.HIGH,
    ) -> "TextInputEvent":
        """Create a text input event.

        Args:
            source: Text source identifier
            text: The text content
            priority: Event priority (default HIGH for user input)

        Returns:
            TextInputEvent: The created event
        """
        data = {
            "text": text,
            "length": len(text),
        }
        return TextInputEvent(
            event_type="text.input",
            source=source,
            data=data,
            priority=priority,
        )


@dataclass(frozen=True)
class AudioSampleEvent(Event):
    """Event containing an audio sample."""

    @staticmethod
    def create(
        source: str,
        samples: npt.NDArray[Any],
        sample_rate: int,
        priority: EventPriority = EventPriority.NORMAL,
    ) -> "AudioSampleEvent":
        """Create an audio sample event.

        Args:
            source: Audio source identifier
            samples: Audio samples (numpy array)
            sample_rate: Sample rate in Hz
            priority: Event priority

        Returns:
            AudioSampleEvent: The created event
        """
        data = {
            "samples": samples,
            "sample_rate": sample_rate,
            "duration": len(samples) / sample_rate,
            "shape": samples.shape,
        }
        return AudioSampleEvent(
            event_type="audio.sample",
            source=source,
            data=data,
            priority=priority,
        )
