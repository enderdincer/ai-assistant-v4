"""AI Assistant Microservices.

This package contains independent microservices that communicate via MQTT.

Services:
    - audio_collector: Captures audio from microphone, publishes audio samples
    - transcription: Converts speech to text using STT
    - speech: Text-to-speech synthesis and playback
    - assistant: LLM-powered conversational AI (the brain)
    - text_interaction: CLI interface for text-based interaction
    - memory: Persistent storage for conversations and facts
    - extraction: Extracts facts from conversations using LLM

To run a service:
    python -m services.<service_name>

Example:
    python -m services.assistant
    python -m services.transcription
"""

__all__ = [
    "audio_collector",
    "transcription",
    "speech",
    "assistant",
    "text_interaction",
    "memory",
    "extraction",
]
