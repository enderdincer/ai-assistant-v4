"""Audio Collector Service.

Captures audio from the microphone and publishes to MQTT.
"""

from services.audio_collector.service import AudioCollectorService, AudioCollectorConfig

__all__ = ["AudioCollectorService", "AudioCollectorConfig"]
