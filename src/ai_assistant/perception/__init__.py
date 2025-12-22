"""Perception module for the AI assistant.

The perception module handles all input sources (cameras, text, audio)
and provides a unified event-driven interface for processing sensory data.
"""

from ai_assistant.perception.core import PerceptionConfig, PerceptionSystem
from ai_assistant.perception.input_sources import (
    BaseInputSource,
    CameraInputSource,
    TextInputSource,
    AudioInputSource,
)

__all__ = [
    # Core system
    "PerceptionConfig",
    "PerceptionSystem",
    # Input sources
    "BaseInputSource",
    "CameraInputSource",
    "TextInputSource",
    "AudioInputSource",
]
