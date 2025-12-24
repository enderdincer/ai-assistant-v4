"""Audio-related message schemas."""

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt

from ai_assistant.shared.messages.base import BaseMessage


@dataclass
class AudioSampleMessage(BaseMessage):
    """Message containing raw audio samples.

    Published to: all/raw/audio/{machine_id}

    Attributes:
        samples: Audio samples as float32 numpy array
        sample_rate: Sample rate in Hz (e.g., 16000)
        channels: Number of audio channels (usually 1 for mono)
        machine_id: ID of the machine that captured this audio
        chunk_index: Sequential chunk number for ordering
    """

    samples: npt.NDArray[np.float32] = field(default_factory=lambda: np.array([], dtype=np.float32))
    sample_rate: int = 16000
    channels: int = 1
    machine_id: str = ""
    chunk_index: int = 0

    @classmethod
    def create(
        cls,
        samples: npt.NDArray[np.float32],
        sample_rate: int,
        machine_id: str,
        chunk_index: int = 0,
        channels: int = 1,
    ) -> "AudioSampleMessage":
        """Create an audio sample message.

        Args:
            samples: Audio samples as float32 array
            sample_rate: Sample rate in Hz
            machine_id: ID of the source machine
            chunk_index: Sequential chunk number
            channels: Number of audio channels

        Returns:
            AudioSampleMessage instance
        """
        return cls(
            samples=samples,
            sample_rate=sample_rate,
            channels=channels,
            machine_id=machine_id,
            chunk_index=chunk_index,
            source=f"audio-collector-{machine_id}",
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AudioSampleMessage":
        """Create from dictionary, handling numpy array conversion."""
        samples = data.get("samples", np.array([], dtype=np.float32))
        if not isinstance(samples, np.ndarray):
            samples = np.array(samples, dtype=np.float32)

        return cls(
            message_id=data.get("message_id", ""),
            timestamp=data.get("timestamp", 0.0),
            source=data.get("source", ""),
            samples=samples,
            sample_rate=data.get("sample_rate", 16000),
            channels=data.get("channels", 1),
            machine_id=data.get("machine_id", ""),
            chunk_index=data.get("chunk_index", 0),
        )
