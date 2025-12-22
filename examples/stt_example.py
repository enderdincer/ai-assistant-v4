"""Example demonstrating audio STT processor.

This example shows how to:
1. Create a perception system with audio input
2. Add an STT processor to transcribe audio
3. Subscribe to transcription events
4. Process transcribed text
"""

from ai_assistant.perception.core import PerceptionSystem, PerceptionConfig
from ai_assistant.shared.interfaces import IEvent


def handle_transcription(event: IEvent) -> None:
    """Handle transcription events.

    Args:
        event: AudioTranscriptionEvent containing transcription
    """
    text = event.data["text"]
    confidence = event.data["confidence"]
    language = event.data["language"]
    duration = event.data["audio_duration"]
    model = event.data["model_name"]

    print(f"\n[Transcription]")
    print(f"  Text: {text}")
    print(f"  Language: {language}")
    print(f"  Confidence: {confidence:.2f}")
    print(f"  Audio Duration: {duration:.2f}s")
    print(f"  Model: {model}")


def main() -> None:
    """Run the STT example."""
    print("=== Audio STT Processor Example ===\n")

    # Method 1: Create config with audio and STT
    config = PerceptionConfig.with_audio_and_stt(
        audio_source_id="microphone",
        stt_processor_id="stt_en",
        model_name="nvidia/canary-1b",
        device="cpu",
        buffer_duration=3.0,
        language="en",
    )

    # Method 2: Add processor dynamically (commented out)
    # config = PerceptionConfig.default()
    # system = PerceptionSystem(config)
    # system.add_audio_input("microphone")
    # system.add_stt_processor("stt_en", {
    #     "model_name": "nvidia/canary-1b",
    #     "device": "cpu",
    #     "buffer_duration": 3.0,
    #     "language": "en",
    # })

    # Create system with configuration
    system = PerceptionSystem(config)

    # Subscribe to transcription events
    system.subscribe("audio.transcription", handle_transcription)

    # Start the system
    print("Starting perception system...")
    system.start()

    print("\nSystem Status:")
    status = system.get_status()
    print(f"  Input Sources: {len(status['input_sources'])}")
    print(f"  Processors: {len(status['processors']['processors'])}")
    print(f"  Event Bus Running: {status['event_bus']['running']}")

    print("\nListening for audio and transcribing...")
    print("(This is a demo - audio input is simulated)")
    print("\nPress Ctrl+C to stop\n")

    try:
        # In a real scenario, audio would be captured from microphone
        # For this demo, we just wait
        import time

        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n\nStopping...")
    finally:
        system.stop()
        print("System stopped")


if __name__ == "__main__":
    main()
