#!/usr/bin/env python3
"""Multi-input perception example.

This example demonstrates:
- Running multiple input sources simultaneously (camera, text, audio)
- Handling different event types with different priorities
- Event statistics and monitoring
"""

import time
import signal
import sys
from typing import Any, Dict
from ai_assistant.perception import PerceptionSystem, PerceptionConfig
from ai_assistant.shared.logging import LogLevel, get_logger
from ai_assistant.shared.interfaces import IEvent

logger = get_logger(__name__)

# Global system reference for signal handler
system: PerceptionSystem | None = None

# Event counters
event_stats: Dict[str, int] = {
    "camera.frame": 0,
    "text.input": 0,
    "audio.sample": 0,
}


def signal_handler(signum: int, frame: Any) -> None:
    """Handle shutdown signals gracefully."""
    logger.info(f"Received signal {signum}, shutting down...")
    print_statistics()
    if system:
        system.stop()
    sys.exit(0)


def print_statistics() -> None:
    """Print event statistics."""
    logger.info("=== Event Statistics ===")
    for event_type, count in event_stats.items():
        logger.info(f"  {event_type}: {count} events")


def handle_camera_frame(event: IEvent) -> None:
    """Handle camera frame events."""
    event_stats["camera.frame"] += 1
    data = event.data
    frame_num = data["frame_number"]
    shape = data["shape"]

    # Log every 60th frame to reduce spam
    if frame_num % 60 == 0:
        logger.info(
            f"📷 Camera frame #{frame_num} from {event.source}: "
            f"shape={shape}, priority={event.priority.name}"
        )


def handle_text_input(event: IEvent) -> None:
    """Handle text input events."""
    event_stats["text.input"] += 1
    text = event.data["text"]
    logger.info(f"💬 Text input from {event.source}: '{text}'")

    # Check for special commands
    text_lower = text.lower().strip()
    if text_lower in ("quit", "exit"):
        logger.info("Quit command received")
        print_statistics()
        if system:
            system.stop()
        sys.exit(0)
    elif text_lower == "status":
        if system:
            status = system.get_status()
            logger.info(f"System status: {status}")
            print_statistics()
    elif text_lower == "stats":
        print_statistics()


def handle_audio_sample(event: IEvent) -> None:
    """Handle audio sample events."""
    event_stats["audio.sample"] += 1
    data = event.data
    duration = data["duration"]
    sample_rate = data["sample_rate"]

    # Log every 50th sample to reduce spam
    if event_stats["audio.sample"] % 50 == 0:
        logger.info(
            f"🎤 Audio sample #{event_stats['audio.sample']} from {event.source}: "
            f"duration={duration:.3f}s, rate={sample_rate}Hz"
        )


def main() -> None:
    """Main function."""
    global system

    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    logger.info("=== Multi-Input Perception System Example ===")
    logger.info("This example demonstrates multiple input sources working together")

    # Create configuration without pre-configured camera
    # We'll add sources dynamically to show the flexibility
    config = PerceptionConfig.default()
    config.log_level = LogLevel.INFO
    config.max_queue_size = 2000  # Larger queue for multiple inputs

    # Create system
    system = PerceptionSystem(config)

    # Subscribe to all event types
    system.subscribe("camera.frame", handle_camera_frame)
    system.subscribe("text.input", handle_text_input)
    system.subscribe("audio.sample", handle_audio_sample)

    # Start system
    system.start()

    logger.info("Adding input sources dynamically...")

    # Add camera input (comment out if no camera available)
    try:
        system.add_camera_input("camera_0", config={"device_id": 0, "fps": 30})
        logger.info("✓ Camera input added")
    except Exception as e:
        logger.warning(f"✗ Could not add camera: {e}")
        logger.info("  (Continuing without camera)")

    # Add text input (console)
    system.add_text_input("console", config={"prompt": "> "})
    logger.info("✓ Text input added")

    # Add audio input (stub implementation for demo)
    system.add_audio_input("mic_0", config={"sample_rate": 16000, "chunk_size": 1024})
    logger.info("✓ Audio input added")

    logger.info("\n=== System Ready ===")
    logger.info("Commands:")
    logger.info("  'status' - Show system status")
    logger.info("  'stats'  - Show event statistics")
    logger.info("  'quit'   - Exit the system")
    logger.info(f"\nActive input sources: {system.list_input_sources()}")

    # Print statistics periodically
    last_stats_time = time.time()
    stats_interval = 10.0  # Print stats every 10 seconds

    try:
        # Keep running until interrupted
        while True:
            time.sleep(1)

            # Print periodic statistics
            current_time = time.time()
            if current_time - last_stats_time >= stats_interval:
                print_statistics()
                last_stats_time = current_time

    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")
    finally:
        logger.info("Shutting down...")
        print_statistics()
        system.stop()
        logger.info("Shutdown complete")


if __name__ == "__main__":
    main()
