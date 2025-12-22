#!/usr/bin/env python3
"""Custom test script for the perception system."""

import time
from ai_assistant.perception import PerceptionSystem, PerceptionConfig
from ai_assistant.shared.logging import LogLevel, get_logger
from ai_assistant.shared.interfaces import IEvent

logger = get_logger(__name__)

# Event counters
counters = {"text": 0, "audio": 0, "camera": 0, "transcription": 0}


def handle_audio(event: IEvent) -> None:
    """Handle audio samples."""
    counters["audio"] += 1
    if counters["audio"] % 20 == 0:
        logger.info(f"🎤 Audio: {counters['audio']} samples")


def handle_transcription(event: IEvent) -> None:
    """Handle audio transcriptions from STT processor."""
    counters["transcription"] += 1
    text = event.data.get("text", "")
    confidence = event.data.get("confidence", 0.0)
    language = event.data.get("language", "unknown")
    duration = event.data.get("audio_duration", 0.0)

    logger.info(
        f"🎙️  Transcription #{counters['transcription']}: '{text}' "
        f"(lang: {language}, conf: {confidence:.2f}, duration: {duration:.1f}s)"
    )


def main():
    """Main test function."""
    logger.info("=== Custom Perception System Test ===\n")

    # Configure system
    config = PerceptionConfig.default()
    config.log_level = LogLevel.INFO
    config.max_queue_size = 500

    # Create and configure system
    system = PerceptionSystem(config)

    system.subscribe("audio.sample", handle_audio)
    system.subscribe("audio.transcription", handle_transcription)

    # Start system
    system.start()
    logger.info("✓ System started\n")

    # Add input sources
    logger.info("Adding input sources...")

    # Try to add camera (will fail gracefully if no camera)
    try:
        system.add_camera_input("camera_0", {"device_id": 0, "fps": 15})
        logger.info("✓ Camera added")
    except Exception as e:
        logger.warning(f"⚠️  Camera not available: {e}")

    # Add text input
    system.add_text_input("console", {"interval": 0.5})
    logger.info("✓ Text input added")

    # Add audio input
    system.add_audio_input("microphone", {"sample_rate": 16000})
    logger.info("✓ Audio input added")

    # Try to add STT processor (will fail gracefully if dependencies not installed)
    stt_enabled = False
    try:
        # First check if dependencies are available
        import torch
        import transformers

        logger.info("\nAdding STT processor...")
        system.add_stt_processor(
            "stt_en",
            {
                "model_name": "nvidia/canary-1b",
                "device": "cpu",
                "buffer_duration": 3.0,
                "language": "en",
                "min_confidence": 0.3,
            },
        )
        logger.info("✓ STT processor added")
        logger.info("  Note: Model will be downloaded on first run (~1.2GB)")
        stt_enabled = True
    except ImportError as e:
        logger.warning(f"\n⚠️  STT dependencies not installed: {e}")
        logger.warning('  Install with: pip install -e ".[stt]"')
        logger.warning("  Continuing without STT processor...\n")
    except RuntimeError as e:
        logger.warning(f"\n⚠️  Failed to initialize STT processor: {e}")
        logger.warning("  Continuing without STT processor...\n")
    except Exception as e:
        logger.warning(f"\n⚠️  Unexpected error adding STT processor: {e}")
        logger.warning("  Continuing without STT processor...\n")

    # Show active sources and processors
    sources = system.list_input_sources()
    processors = system.list_processors()
    logger.info(f"\n📊 Active sources: {sources}")
    logger.info(f"📊 Active processors: {processors}")
    logger.info("\nType 'quit' to stop\n")

    # Run for a while
    try:
        start_time = time.time()
        while time.time() - start_time < 15:  # Run for 15 seconds
            time.sleep(0.5)

            # Print status every 3 seconds
            if int(time.time() - start_time) % 3 == 0:
                status = system.get_status()
                queue_size = status["event_bus"]["queue_size"]
                logger.info(f"⏱️  Running... Queue: {queue_size}, Events: {sum(counters.values())}")

    except KeyboardInterrupt:
        logger.info("\n⚠️  Interrupted by user")

    # Stop system
    logger.info("\n🛑 Stopping system...")
    system.stop()

    # Print summary
    logger.info("\n=== Test Summary ===")
    logger.info(f"Camera frames: {counters['camera']}")
    logger.info(f"Text inputs: {counters['text']}")
    logger.info(f"Audio samples: {counters['audio']}")
    logger.info(f"Transcriptions: {counters['transcription']}")
    logger.info(f"Total events: {sum(counters.values())}")
    logger.info("\n✅ Test complete!")


if __name__ == "__main__":
    main()
