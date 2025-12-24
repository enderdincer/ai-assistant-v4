"""Transcription Service.

Provides Speech-to-Text functionality via MQTT.
"""

from services.transcription.service import TranscriptionService, TranscriptionServiceConfig

__all__ = ["TranscriptionService", "TranscriptionServiceConfig"]
