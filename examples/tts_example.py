"""Example: Basic TTS usage with Kokoro.

This example demonstrates:
1. Setting up the Kokoro TTS engine
2. Synthesizing speech from text
3. Saving audio to a file
4. Using different voices

Before running this example:
1. Download the model files:
   - kokoro-v1.0.onnx (from releases)
   - voices-v1.0.bin (from releases)
2. Place them in the same directory as this script or update the paths below
3. Install dependencies: pip install ai-assistant[tts]
"""

from pathlib import Path

from ai_assistant.actions.tts import KokoroTTS, KokoroTTSConfig


def main() -> None:
    """Run the TTS example."""
    # Setup paths (adjust these to match your setup)
    model_path = Path("kokoro-v1.0.onnx")
    voices_path = Path("voices-v1.0.bin")

    # Check if model files exist
    if not model_path.exists():
        print(f"Error: Model file not found: {model_path}")
        print("Please download kokoro-v1.0.onnx from the releases page")
        return

    if not voices_path.exists():
        print(f"Error: Voices file not found: {voices_path}")
        print("Please download voices-v1.0.bin from the releases page")
        return

    # Create configuration
    config = KokoroTTSConfig(
        model_path=model_path,
        voices_path=voices_path,
        voice="af_bella",  # Female voice
        speed=1.0,  # Normal speed
        lang="en-us",
    )

    # Initialize TTS engine
    print("Initializing Kokoro TTS engine...")
    tts = KokoroTTS(config)
    tts.initialize()

    # List available voices
    voices = tts.get_available_voices()
    print(f"\nAvailable voices ({len(voices)}):")
    for voice in sorted(voices):
        print(f"  - {voice}")

    # Synthesize speech with default voice
    text = "Hello! This is a demonstration of the Kokoro text-to-speech system."
    print(f"\nSynthesizing with voice '{config.voice}'...")
    audio = tts.synthesize(text)
    print(f"Generated {len(audio)} audio samples")

    # Save to file
    output_path = Path("output_bella.wav")
    tts.save_to_file(audio, output_path)
    print(f"Audio saved to: {output_path}")

    # Try a different voice
    print("\nSynthesizing with voice 'am_adam'...")
    audio2 = tts.synthesize(text, voice="am_adam")
    output_path2 = Path("output_adam.wav")
    tts.save_to_file(audio2, output_path2)
    print(f"Audio saved to: {output_path2}")

    # Cleanup
    tts.cleanup()
    print("\nDone!")


def example_context_manager() -> None:
    """Example using context manager pattern."""
    config = KokoroTTSConfig(
        model_path="kokoro-v1.0.onnx",
        voices_path="voices-v1.0.bin",
        voice="af_sarah",
    )

    # Using context manager automatically handles initialization and cleanup
    with KokoroTTS(config) as tts:
        audio = tts.synthesize("This is using a context manager!")
        tts.save_to_file(audio, "output_context.wav")


if __name__ == "__main__":
    main()
    # example_context_manager()
