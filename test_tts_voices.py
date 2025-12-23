"""Interactive TTS testing script.

This script allows you to:
1. Test different sentences with various voices
2. Compare voice outputs
3. Adjust speed settings
4. Generate multiple audio files quickly

Before running:
1. Download kokoro-v1.0.onnx and voices-v1.0.bin
2. Install dependencies: pip install -e ".[tts]"
3. Place model files in the same directory or update paths below
"""

from pathlib import Path
from typing import Optional

from ai_assistant.actions.tts import KokoroTTS, KokoroTTSConfig


# Configuration
MODEL_PATH = Path("kokoro-v1.0.onnx")
VOICES_PATH = Path("voices-v1.0.bin")
OUTPUT_DIR = Path("tts_outputs")


# Test sentences - add your own!
TEST_SENTENCES = [
    "Hello! Welcome to the Kokoro text-to-speech system.",
    "The quick brown fox jumps over the lazy dog.",
    "How are you doing today? I hope you're having a wonderful day!",
    "This is a test of the emergency broadcast system.",
    "Python is an amazing programming language for artificial intelligence.",
    "Can you hear the difference between the various voices?",
]


# Voice groups for easy testing
VOICE_GROUPS = {
    "american_female": ["af_bella", "af_sarah", "af_nicole"],
    "american_male": ["am_adam", "am_michael"],
    "british_female": ["bf_emma", "bf_isabella"],
    "british_male": ["bm_george", "bm_lewis"],
}


def setup_output_dir() -> None:
    """Create output directory if it doesn't exist."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {OUTPUT_DIR.absolute()}")


def check_model_files() -> bool:
    """Check if required model files exist."""
    if not MODEL_PATH.exists():
        print(f"❌ Error: Model file not found: {MODEL_PATH}")
        print("   Download from: https://github.com/thewh1teagle/kokoro-onnx/releases")
        return False

    if not VOICES_PATH.exists():
        print(f"❌ Error: Voices file not found: {VOICES_PATH}")
        print("   Download from: https://github.com/thewh1teagle/kokoro-onnx/releases")
        return False

    print("✓ Model files found")
    return True


def test_single_voice(
    tts: KokoroTTS,
    text: str,
    voice: str,
    speed: float = 1.0,
    play_audio: bool = True,
    save: bool = False,
) -> None:
    """Test a single voice with given text.

    Args:
        tts: Initialized TTS engine
        text: Text to synthesize
        voice: Voice name
        speed: Speech speed
        play_audio: Whether to play audio (default: True)
        save: Whether to save audio file (default: False)
    """
    try:
        print(f"  🗣️  {voice}: ", end="", flush=True)

        # Synthesize
        audio = tts.synthesize(text, voice=voice)
        print(f"✓ ({len(audio)} samples)", end="")

        # Play audio
        if play_audio:
            print(" 🔊 Playing...", end="", flush=True)
            tts.play(audio, blocking=True)
            print(" ✓", end="")

        # Save if requested
        if save:
            filename = f"{voice}_speed{speed}.wav"
            output_path = OUTPUT_DIR / filename
            tts.save_to_file(audio, output_path)
            print(f" → {filename}", end="")

        print()  # New line

    except Exception as e:
        print(f"❌ Error: {e}")


def test_all_voices_with_sentence(tts: KokoroTTS, text: str, speed: float = 1.0) -> None:
    """Test all available voices with a single sentence.

    Args:
        tts: Initialized TTS engine
        text: Text to synthesize
        speed: Speech speed
    """
    voices = tts.get_available_voices()

    print(f"\n{'=' * 80}")
    print(f"Testing: '{text[:60]}...'")
    print(f"Speed: {speed}x | Voices: {len(voices)}")
    print(f"{'=' * 80}")

    for voice in sorted(voices):
        test_single_voice(tts, text, voice, speed=speed)


def test_voice_group(
    tts: KokoroTTS, text: str, group_name: str, voices: list[str], speed: float = 1.0
) -> None:
    """Test a group of voices with the same text.

    Args:
        tts: Initialized TTS engine
        text: Text to synthesize
        group_name: Name of the voice group
        voices: List of voice names
        speed: Speech speed
    """
    print(f"\n{'=' * 80}")
    print(f"Testing {group_name.replace('_', ' ').title()}")
    print(f"Text: '{text[:60]}...'")
    print(f"Speed: {speed}x")
    print(f"{'=' * 80}")

    for voice in voices:
        test_single_voice(tts, text, voice, speed=speed)


def test_speed_variations(tts: KokoroTTS, text: str, voice: str) -> None:
    """Test different speed settings with a single voice.

    Args:
        tts: Initialized TTS engine
        text: Text to synthesize
        voice: Voice name
    """
    speeds = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]

    print(f"\n{'=' * 80}")
    print(f"Testing Speed Variations: {voice}")
    print(f"Text: '{text[:60]}...'")
    print(f"{'=' * 80}")

    for speed in speeds:
        print(f"Speed {speed}x: ", end="", flush=True)

        # Update config speed
        config = tts.config
        config.speed = speed

        audio = tts.synthesize(text, voice=voice)

        filename = f"{voice}_speed{speed}.wav"
        output_path = OUTPUT_DIR / filename
        tts.save_to_file(audio, output_path)

        print(f"✓ → {filename}")


def interactive_mode(tts: KokoroTTS) -> None:
    """Interactive mode for testing custom sentences.

    Args:
        tts: Initialized TTS engine
    """
    voices = tts.get_available_voices()

    print(f"\n{'=' * 80}")
    print("INTERACTIVE MODE")
    print(f"{'=' * 80}")
    print("\nAvailable voices:")
    for i, voice in enumerate(sorted(voices), 1):
        print(f"  {i:2d}. {voice}")

    print("\nCommands:")
    print("  - Type text to synthesize and play")
    print("  - Type 'v:voice_name' to change voice (e.g., v:af_bella)")
    print("  - Type 's:1.5' to change speed (e.g., s:1.5)")
    print("  - Type 'save' to toggle auto-save (current: OFF)")
    print("  - Type 'list' to show available voices")
    print("  - Type 'quit' or 'exit' to exit")

    current_voice = "af_bella"
    current_speed = 1.0
    auto_save = False

    print(
        f"\nCurrent settings: voice={current_voice}, speed={current_speed}x, auto_save={auto_save}"
    )

    while True:
        try:
            user_input = input("\n> ").strip()

            if not user_input:
                continue

            if user_input.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break

            if user_input.lower() == "list":
                print("\nAvailable voices:")
                for voice in sorted(voices):
                    marker = " ← current" if voice == current_voice else ""
                    print(f"  {voice}{marker}")
                continue

            # Change voice
            if user_input.startswith("v:"):
                new_voice = user_input[2:].strip()
                if new_voice in voices:
                    current_voice = new_voice
                    print(f"✓ Voice changed to: {current_voice}")
                else:
                    print(f"❌ Voice '{new_voice}' not found. Type 'list' to see options.")
                continue

            # Change speed
            if user_input.startswith("s:"):
                try:
                    new_speed = float(user_input[2:].strip())
                    if 0.1 <= new_speed <= 3.0:
                        current_speed = new_speed
                        tts.config.speed = current_speed
                        print(f"✓ Speed changed to: {current_speed}x")
                    else:
                        print("❌ Speed must be between 0.1 and 3.0")
                except ValueError:
                    print("❌ Invalid speed value")
                continue

            # Toggle auto-save
            if user_input.lower() == "save":
                auto_save = not auto_save
                print(f"✓ Auto-save {'enabled' if auto_save else 'disabled'}")
                continue

            # Synthesize and play text
            print(f"🗣️  {current_voice} at {current_speed}x: ", end="", flush=True)

            audio = tts.synthesize(user_input, voice=current_voice)
            print(f"✓ ({len(audio)} samples) 🔊 Playing...", end="", flush=True)

            # Play audio
            tts.play(audio, blocking=True)
            print(" ✓", end="")

            # Save if auto-save enabled
            if auto_save:
                filename = f"interactive_{current_voice}.wav"
                output_path = OUTPUT_DIR / filename
                tts.save_to_file(audio, output_path)
                print(f" → {filename}", end="")

            print()  # New line

        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")


def main() -> None:
    """Main testing function with menu."""
    print("=" * 80)
    print("KOKORO TTS TESTING SCRIPT")
    print("=" * 80)

    # Setup
    setup_output_dir()

    if not check_model_files():
        return

    # Initialize TTS
    print("\nInitializing TTS engine...")
    config = KokoroTTSConfig(
        model_path=MODEL_PATH,
        voices_path=VOICES_PATH,
        voice="af_bella",
        speed=1.0,
    )

    tts = KokoroTTS(config)
    tts.initialize()

    voices = tts.get_available_voices()
    print(f"✓ Loaded {len(voices)} voices")

    # Menu
    while True:
        print(f"\n{'=' * 80}")
        print("TEST OPTIONS")
        print(f"{'=' * 80}")
        print("1. Test all voices with first sentence (plays audio)")
        print("2. Test all sentences with one voice (plays audio)")
        print("3. Test voice groups (plays audio)")
        print("4. Test speed variations (saves to file)")
        print("5. Interactive mode - RECOMMENDED (plays audio)")
        print("6. Generate comparison samples (saves to file)")
        print("0. Exit")
        print()

        choice = input("Select option (0-6): ").strip()

        try:
            if choice == "0":
                print("Cleaning up...")
                tts.cleanup()
                print("Goodbye!")
                break

            elif choice == "1":
                test_all_voices_with_sentence(tts, TEST_SENTENCES[0])

            elif choice == "2":
                voice = input(f"Enter voice name (default: af_bella): ").strip() or "af_bella"
                if voice not in voices:
                    print(f"❌ Voice not found. Available voices: {', '.join(voices[:5])}...")
                    continue

                print(f"\nTesting {len(TEST_SENTENCES)} sentences with {voice}:")
                for i, text in enumerate(TEST_SENTENCES, 1):
                    print(f"\n[{i}/{len(TEST_SENTENCES)}] ", end="")
                    test_single_voice(tts, text, voice)

            elif choice == "3":
                print("\nAvailable groups:")
                for i, (group_name, group_voices) in enumerate(VOICE_GROUPS.items(), 1):
                    print(
                        f"  {i}. {group_name.replace('_', ' ').title()} "
                        f"({len(group_voices)} voices)"
                    )

                group_choice = input("Select group (1-4): ").strip()
                group_idx = int(group_choice) - 1

                if 0 <= group_idx < len(VOICE_GROUPS):
                    group_name = list(VOICE_GROUPS.keys())[group_idx]
                    group_voices = VOICE_GROUPS[group_name]
                    test_voice_group(tts, TEST_SENTENCES[0], group_name, group_voices)
                else:
                    print("❌ Invalid group selection")

            elif choice == "4":
                voice = input("Enter voice name (default: af_bella): ").strip() or "af_bella"
                if voice not in voices:
                    print(f"❌ Voice not found")
                    continue

                test_speed_variations(tts, TEST_SENTENCES[0], voice)

            elif choice == "5":
                interactive_mode(tts)

            elif choice == "6":
                print("\nGenerating comparison samples (saving to files)...")
                # Generate samples with popular voices (save only, no play)
                comparison_voices = ["af_bella", "am_adam", "bf_emma", "bm_george"]
                text = TEST_SENTENCES[0]

                for voice in comparison_voices:
                    if voice in voices:
                        test_single_voice(tts, text, voice, play_audio=False, save=True)

                print("\n✓ Comparison samples saved to tts_outputs/!")

            else:
                print("❌ Invalid option")

        except KeyboardInterrupt:
            print("\n\nOperation cancelled")
            continue
        except Exception as e:
            print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()
