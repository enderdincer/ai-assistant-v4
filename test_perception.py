#!/usr/bin/env python3
"""Custom test script for the perception system."""

import time
from ai_assistant.perception import PerceptionSystem, PerceptionConfig
from ai_assistant.shared.logging import LogLevel, get_logger
from ai_assistant.shared.interfaces import IEvent

logger = get_logger(__name__)

# Event counters
counters = {"text": 0, "audio": 0, "camera": 0}


def handle_camera(event: IEvent):
    """Handle camera frames."""
    counters["camera"] += 1
    if counters["camera"] % 10 == 0:
        logger.info(f"📷 Camera: {counters['camera']} frames")


def handle_text(event: IEvent):
    """Handle text input."""
    counters["text"] += 1
    text = event.data.get("text", "")
    logger.info(f"💬 Text #{counters['text']}: '{text}'")

    # Stop on 'quit'
    if text.lower().strip() == "quit":
        return True


def handle_audio(event: IEvent):
    """Handle audio samples."""
    counters["audio"] += 1
    if counters["audio"] % 20 == 0:
        logger.info(f"🎤 Audio: {counters['audio']} samples")


def main():
    """Main test function."""
    logger.info("=== Custom Perception System Test ===\n")

    # Configure system
    config = PerceptionConfig.default()
    config.log_level = LogLevel.INFO
    config.max_queue_size = 500

    # Create and configure system
    system = PerceptionSystem(config)

    # Subscribe to events
    system.subscribe("camera.frame", handle_camera)
    system.subscribe("text.input", handle_text)
    system.subscribe("audio.sample", handle_audio)

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

    # Show active sources
    sources = system.list_input_sources()
    logger.info(f"\n📊 Active sources: {sources}\n")
    logger.info("Type 'quit' to stop\n")

    # Run for a while
    try:
        start_time = time.time()
        while time.time() - start_time < 10:  # Run for 10 seconds
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
    logger.info(f"Total events: {sum(counters.values())}")
    logger.info("\n✅ Test complete!")


if __name__ == "__main__":
    main()
