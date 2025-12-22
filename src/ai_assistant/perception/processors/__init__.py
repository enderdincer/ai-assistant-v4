"""Sensory processors for the perception system."""

from ai_assistant.perception.processors.base import BaseProcessor
from ai_assistant.perception.processors.manager import ProcessorManager
from ai_assistant.perception.processors.stt_processor import STTProcessor

__all__ = [
    "BaseProcessor",
    "ProcessorManager",
    "STTProcessor",
]
