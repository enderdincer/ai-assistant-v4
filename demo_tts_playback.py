"""Simple TTS playback demo.

This script demonstrates the audio playback functionality of the TTS module.
"""

from pathlib import Path

from ai_assistant.actions.tts import KokoroTTS, KokoroTTSConfig


def main() -> None:
    """Run a simple TTS demo with audio playback."""
    print("=" * 80)
    print("KOKORO TTS - AUDIO PLAYBACK DEMO")
    print("=" * 80)

    # Check model files
    model_path = Path("kokoro-v1.0.onnx")
    voices_path = Path("voices-v1.0.bin")

    if not model_path.exists() or not voices_path.exists():
        print("\n❌ Model files not found!")
        print("Please run: python test_tts_voices.py")
        return

    # Initialize TTS
    print("\nInitializing TTS engine...")
    config = KokoroTTSConfig(
        model_path=model_path,
        voices_path=voices_path,
        voice="af_bella",
    )

    tts = KokoroTTS(config)
    tts.initialize()
    print("✅ TTS engine ready!\n")

    # Demo different methods
    print("=" * 80)
    print("METHOD 1: Using speak() - synthesizes and plays automatically")
    print("=" * 80)
    print("🔊 Speaking: 'Hello! This is Bella speaking.'")
    tts.speak("Hello! This is Bella speaking.")
    print("✅ Complete!\n")

    print("=" * 80)
    print("METHOD 2: Using synthesize() + play() separately")
    print("=" * 80)
    print("🔊 Speaking: 'This is Adam with a separate play call.'")
    audio = tts.synthesize("This is Adam with a separate play call.", voice="am_adam")
    tts.play(audio)
    print("✅ Complete!\n")

    print("=" * 80)
    print("METHOD 3: speak() with auto-save")
    print("=" * 80)
    print("🔊 Speaking and saving: 'This will be saved and played.'")
    tts.speak("This will be saved and played.", voice="bf_emma", save_to="demo_output.wav")
    print("✅ Complete! Saved to demo_output.wav\n")

    # Cleanup
    tts.cleanup()

    print("=" * 80)
    print("DEMO COMPLETE!")
    print("=" * 80)
    print("\nThe TTS module now:")
    print("  ✓ Plays audio directly (no need to open files)")
    print("  ✓ Uses speak() for quick text-to-speech")
    print("  ✓ Optional file saving with save_to parameter")
    print("\nTry the interactive mode: python test_tts_voices.py (option 5)")


if __name__ == "__main__":
    main()
