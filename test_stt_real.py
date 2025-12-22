#!/usr/bin/env python3
"""Real STT processor test with actual audio input and transcription.

This test uses real audio libraries and the NVIDIA Parakeet TDT model
via sherpa-onnx. No mocking - everything is production code.

Requirements:
    pip install sherpa-onnx sounddevice soundfile numpy

Usage:
    python test_stt_real.py

The system will:
    1. Initialize the STT model (downloads ~460MB on first run)
    2. Start capturing audio from your microphone
    3. Transcribe speech every 2 seconds
    4. Display transcriptions in real-time
    5. Press Ctrl+C to stop
"""

import time
import numpy as np
from ai_assistant.perception import PerceptionSystem, PerceptionConfig
from ai_assistant.shared.logging import LogLevel, get_logger
from ai_assistant.shared.interfaces import IEvent

logger = get_logger(__name__)

# Transcription results
transcriptions = []


def handle_transcription(event: IEvent) -> None:
    """Handle real transcription events from STT processor."""
    text = event.data.get("text", "")
    confidence = event.data.get("confidence", 0.0)
    language = event.data.get("language", "unknown")
    duration = event.data.get("audio_duration", 0.0)
    model = event.data.get("model_name", "unknown")

    transcriptions.append(
        {
            "text": text,
            "confidence": confidence,
            "language": language,
            "duration": duration,
            "timestamp": time.time(),
        }
    )

    logger.info("\n" + "=" * 70)
    logger.info("NEW TRANSCRIPTION")
    logger.info("=" * 70)
    logger.info(f"Text:       {text}")
    logger.info(f"Language:   {language}")
    logger.info(f"Confidence: {confidence:.3f}")
    logger.info(f"Duration:   {duration:.2f}s")
    logger.info(f"Model:      {model}")
    logger.info("=" * 70 + "\n")


def check_dependencies() -> bool:
    """Check if all required dependencies are installed."""
    missing = []

    try:
        import sherpa_onnx

        logger.info(f"  sherpa-onnx: installed")
    except ImportError:
        missing.append("sherpa-onnx")

    try:
        import sounddevice

        logger.info(f"  sounddevice: {sounddevice.__version__}")
    except ImportError:
        missing.append("sounddevice")

    try:
        import soundfile

        logger.info(f"  soundfile: {soundfile.__version__}")
    except ImportError:
        missing.append("soundfile")

    try:
        import numpy

        logger.info(f"  numpy: {numpy.__version__}")
    except ImportError:
        missing.append("numpy")

    if missing:
        logger.error(f"\nMissing required dependencies: {', '.join(missing)}")
        logger.error("Install with: pip install sherpa-onnx sounddevice soundfile numpy")
        return False

    return True


def create_real_audio_source(system: PerceptionSystem) -> None:
    """Create a real audio input source using sounddevice.

    This will capture audio from your default microphone.
    """
    logger.info("\nSetting up real microphone input...")

    # Add real audio input with proper configuration
    system.add_audio_input(
        "microphone",
        {
            "sample_rate": 16000,  # 16kHz required by most STT models
            "chunk_size": 1024,  # Process in small chunks
            "channels": 1,  # Mono audio
        },
    )

    logger.info("  Microphone configured")
    logger.info("  Sample rate: 16000 Hz")
    logger.info("  Channels: 1 (mono)")
    logger.info("  Chunk size: 1024 samples")


def main():
    """Main test function for real STT processing."""
    logger.info("=" * 70)
    logger.info("REAL STT PROCESSOR TEST")
    logger.info("=" * 70)
    logger.info("\nThis test uses:")
    logger.info("  - Real audio input from microphone (sounddevice)")
    logger.info("  - NVIDIA Parakeet TDT 0.6B V2 model (sherpa-onnx)")
    logger.info("  - Real transcription processing")
    logger.info("\nNo mocking - production code only!\n")

    # Check dependencies
    logger.info("Checking dependencies...")
    if not check_dependencies():
        logger.error("\nCannot run test without required dependencies")
        return 1

    logger.info("\n" + "=" * 70 + "\n")

    # Configure perception system
    logger.info("Configuring perception system...")
    config = PerceptionConfig.default()
    config.log_level = LogLevel.INFO
    config.max_queue_size = 1000

    # Create system
    system = PerceptionSystem(config)

    # Subscribe to transcription events
    system.subscribe("audio.transcription", handle_transcription)

    # Start system
    logger.info("Starting perception system...")
    system.start()
    logger.info("System started\n")

    # Add real audio input
    create_real_audio_source(system)

    # Add STT processor with real model
    logger.info("\nLoading STT model with Voice Activity Detection...")
    logger.info("  Model: NVIDIA Parakeet TDT 0.6B V2")
    logger.info("  Backend: sherpa-onnx (int8 optimized)")
    logger.info("  VAD: Silero VAD for speech detection")
    logger.info("  Language: English")
    logger.info("\nPlease wait while models are loaded...")
    logger.info("  (First run will download ~460MB STT model + ~2MB VAD model)")

    try:
        start_load = time.time()

        system.add_stt_processor(
            "stt_en",
            {
                "model_name": "csukuangfj/sherpa-onnx-nemo-parakeet-tdt-0.6b-v2",
                "sample_rate": 16000,
                "language": "en",
                "min_confidence": 0.0,
                # VAD settings
                "vad_threshold": 0.5,  # Speech detection sensitivity (0-1)
                "vad_min_silence_duration": 0.5,  # Silence duration to end speech (seconds)
                "vad_min_speech_duration": 0.25,  # Minimum speech to transcribe (seconds)
                "vad_max_speech_duration": 30.0,  # Max speech before forced split (seconds)
            },
        )

        load_time = time.time() - start_load
        logger.info(f"Models loaded in {load_time:.1f}s\n")

    except Exception as e:
        logger.error(f"\nFailed to load STT model: {e}")
        logger.error("Make sure you have enough memory and correct dependencies")
        system.stop()
        return 1

    # Show system status
    status = system.get_status()
    sources = system.list_input_sources()
    processors = system.list_processors()

    logger.info("=" * 70)
    logger.info("SYSTEM STATUS")
    logger.info("=" * 70)
    logger.info(f"Input Sources:  {list(sources.keys())}")
    logger.info(f"Processors:     {list(processors.keys())}")
    logger.info(f"Event Bus:      {'Running' if status['event_bus']['running'] else 'Stopped'}")
    logger.info(f"Queue Size:     {status['event_bus']['queue_size']}")
    logger.info("=" * 70)

    # Instructions
    logger.info("\n" + "=" * 70)
    logger.info("READY TO TRANSCRIBE")
    logger.info("=" * 70)
    logger.info("\nThe system is now listening to your microphone!")
    logger.info("Using Voice Activity Detection (VAD) for natural speech boundaries.\n")
    logger.info("How it works:")
    logger.info("  - Speak naturally into your microphone")
    logger.info("  - Transcription triggers when you pause (0.5s silence)")
    logger.info("  - Complete sentences will be captured together")
    logger.info("  - Press Ctrl+C to stop\n")
    logger.info("Example phrases to try:")
    logger.info("  - 'Hello, this is a test of the speech recognition system'")
    logger.info("  - 'The quick brown fox jumps over the lazy dog'")
    logger.info("  - 'Can you hear me clearly?'")
    logger.info("\n" + "=" * 70 + "\n")

    # Run and collect transcriptions
    start_time = time.time()
    last_status_time = start_time

    try:
        while True:
            time.sleep(0.5)
            current_time = time.time()

            # Show periodic status every 10 seconds
            if current_time - last_status_time >= 10:
                elapsed = int(current_time - start_time)
                logger.info(f"\nRunning for {elapsed}s | Transcriptions: {len(transcriptions)}\n")
                last_status_time = current_time

    except KeyboardInterrupt:
        logger.info("\n\nStopping...")

    # Stop system
    logger.info("\nShutting down...")
    system.stop()

    # Print summary
    logger.info("\n" + "=" * 70)
    logger.info("TEST SUMMARY")
    logger.info("=" * 70)
    logger.info(f"Total transcriptions: {len(transcriptions)}")
    logger.info(f"Test duration: {time.time() - start_time:.1f}s")

    if transcriptions:
        logger.info("\nAll transcriptions:")
        for i, trans in enumerate(transcriptions, 1):
            logger.info(f"\n  #{i}:")
            logger.info(f"    Text: {trans['text']}")
            logger.info(f"    Confidence: {trans['confidence']:.3f}")
            logger.info(f"    Duration: {trans['duration']:.2f}s")

        # Calculate average confidence
        avg_conf = sum(t["confidence"] for t in transcriptions) / len(transcriptions)
        logger.info(f"\n  Average confidence: {avg_conf:.3f}")
    else:
        logger.warning("\nNo transcriptions were generated")
        logger.warning("  Possible reasons:")
        logger.warning("    - Microphone not working or muted")
        logger.warning("    - Audio too quiet (speak louder)")
        logger.warning("    - No speech detected (try speaking)")

    logger.info("\n" + "=" * 70)
    logger.info("Test complete!")
    logger.info("=" * 70 + "\n")

    return 0


if __name__ == "__main__":
    exit(main())
