#!/usr/bin/env python3
"""Basic perception system example.

This example demonstrates:
- Creating a perception system with configuration
- Subscribing to events from different input sources
- Running the system with multiple input sources
- Graceful shutdown
"""

import time
import signal
import sys
from typing import Any
from ai_assistant.perception import PerceptionSystem, PerceptionConfig
from ai_assistant.shared.logging import LogLevel, get_logger
from ai_assistant.shared.interfaces import IEvent

logger = get_logger(__name__)

# Global system reference for signal handler
system: PerceptionSystem | None = None


def signal_handler(signum: int, frame: Any) -> None:
    """Handle shutdown signals gracefully."""
    logger.info(f"Received signal {signum}, shutting down...")
    if system:
        system.stop()
    sys.exit(0)


def handle_camera_frame(event: IEvent) -> None:
    """Handle camera frame events."""
    data = event.data
    frame_num = data["frame_number"]
    shape = data["shape"]

    # Log every 30th frame to avoid spam
    if frame_num % 30 == 0:
        logger.info(
            f"Camera frame #{frame_num} from {event.source}: "
            f"shape={shape}, priority={event.priority.name}"
        )


def handle_text_input(event: IEvent) -> None:
    """Handle text input events."""
    text = event.data["text"]
    logger.info(f"Text input from {event.source}: '{text}'")

    # Check for special commands
    if text.lower() == "quit" or text.lower() == "exit":
        logger.info("Quit command received")
        if system:
            system.stop()
        sys.exit(0)
    elif text.lower() == "status":
        if system:
            status = system.get_status()
            logger.info(f"System status: {status}")


def handle_audio_sample(event: IEvent) -> None:
    """Handle audio sample events."""
    data = event.data
    duration = data["duration"]
    sample_rate = data["sample_rate"]

    # Log periodically
    logger.debug(
        f"Audio sample from {event.source}: duration={duration:.3f}s, rate={sample_rate}Hz"
    )


def main() -> None:
    """Main function."""
    global system

    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    logger.info("=== Basic Perception System Example ===")

    # Create configuration
    # Note: Remove camera if you don't have one
    config = PerceptionConfig.with_camera(device_id=0, fps=30)
    config.log_level = LogLevel.INFO

    # Create system
    system = PerceptionSystem(config)

    # Subscribe to events
    system.subscribe("camera.frame", handle_camera_frame)
    system.subscribe("text.input", handle_text_input)
    system.subscribe("audio.sample", handle_audio_sample)

    # Start system
    system.start()

    # Add text input (console)
    system.add_text_input("console", config={"prompt": "> "})

    # Optionally add audio input (stub implementation)
    # system.add_audio_input('mic_0', config={'sample_rate': 16000})

    logger.info("System started. Type 'status' for status, 'quit' to exit.")
    logger.info(f"Active input sources: {system.list_input_sources()}")

    try:
        # Keep running until interrupted
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")
    finally:
        logger.info("Shutting down...")
        system.stop()
        logger.info("Shutdown complete")


if __name__ == "__main__":
    main()
