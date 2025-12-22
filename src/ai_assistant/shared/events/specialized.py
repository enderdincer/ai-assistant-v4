"""Specialized event types for different input sources."""

from dataclasses import dataclass
from typing import Any, Optional
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


@dataclass(frozen=True)
class AudioTranscriptionEvent(Event):
    """Event containing audio transcription result from STT processor."""

    @staticmethod
    def create(
        source: str,
        text: str,
        language: str,
        confidence: float,
        audio_duration: float,
        model_name: str,
        source_event_id: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
        priority: EventPriority = EventPriority.HIGH,
    ) -> "AudioTranscriptionEvent":
        """Create an audio transcription event.

        Args:
            source: Processor identifier that created this transcription
            text: The transcribed text
            language: Language code (e.g., 'en', 'es', 'fr')
            confidence: Confidence score (0.0 to 1.0)
            audio_duration: Duration of audio in seconds
            model_name: Name of the STT model used
            source_event_id: Optional ID of source AudioSampleEvent for tracing
            metadata: Optional additional metadata
            priority: Event priority (default HIGH for user input)

        Returns:
            AudioTranscriptionEvent: The created event
        """
        data = {
            "text": text,
            "language": language,
            "confidence": confidence,
            "audio_duration": audio_duration,
            "model_name": model_name,
            "source_event_id": source_event_id,
            "text_length": len(text),
            "metadata": metadata or {},
        }
        return AudioTranscriptionEvent(
            event_type="audio.transcription",
            source=source,
            data=data,
            priority=priority,
        )


@dataclass(frozen=True)
class VisionDescriptionEvent(Event):
    """Event containing visual analysis of a camera frame.

    Contains ONLY description and metadata - NO image data.
    Scene details (objects, people, activities) come from the description text.
    """

    @staticmethod
    def create(
        source: str,
        description: str,
        frame_number: int,
        frame_timestamp: float,
        processing_time: float,
        change_score: float,
        model_name: str,
        source_event_id: Optional[str] = None,
        priority: EventPriority = EventPriority.NORMAL,
    ) -> "VisionDescriptionEvent":
        """Create a vision description event.

        Args:
            source: Processor identifier that created this description
            description: Text description of the image (includes objects, people, activities)
            frame_number: Frame number from camera source
            frame_timestamp: Unix timestamp when frame was captured
            processing_time: Time taken to process in seconds
            change_score: How different from previous frame (0.0-1.0, 0=identical, 1=completely different)
            model_name: Name of the VLM model used
            source_event_id: Optional ID of source CameraFrameEvent for tracing
            priority: Event priority

        Returns:
            VisionDescriptionEvent: The created event
        """
        data = {
            "description": description,
            "frame_number": frame_number,
            "frame_timestamp": frame_timestamp,
            "processing_time": processing_time,
            "change_score": change_score,
            "model_name": model_name,
            "source_event_id": source_event_id,
        }
        return VisionDescriptionEvent(
            event_type="vision.description",
            source=source,
            data=data,
            priority=priority,
        )
