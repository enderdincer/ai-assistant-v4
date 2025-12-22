#!/usr/bin/env python3
"""Dynamic source management example.

This example demonstrates:
- Adding and removing input sources at runtime
- Dynamic source management based on user commands
- Monitoring active sources
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

# Track source counters for unique IDs
source_counters = {
    "camera": 0,
    "text": 0,
    "audio": 0,
}


def signal_handler(signum: int, frame: Any) -> None:
    """Handle shutdown signals gracefully."""
    logger.info(f"Received signal {signum}, shutting down...")
    if system:
        system.stop()
    sys.exit(0)


def handle_event(event: IEvent) -> None:
    """Generic event handler that logs all events."""
    logger.info(
        f"📥 Event: type={event.event_type}, source={event.source}, "
        f"priority={event.priority.name}, timestamp={event.timestamp}"
    )


def process_command(command: str) -> None:
    """Process user commands for dynamic source management."""
    global system

    if not system:
        return

    parts = command.strip().lower().split()
    if not parts:
        return

    cmd = parts[0]

    if cmd == "quit" or cmd == "exit":
        logger.info("Quit command received")
        system.stop()
        sys.exit(0)

    elif cmd == "list":
        sources = system.list_input_sources()
        logger.info(f"Active sources ({len(sources)}): {sources}")

    elif cmd == "status":
        status = system.get_status()
        logger.info(f"System status: {status}")

    elif cmd == "add":
        if len(parts) < 2:
            logger.warning("Usage: add <camera|text|audio> [config...]")
            return

        source_type = parts[1]

        if source_type == "camera":
            device_id = int(parts[2]) if len(parts) > 2 else 0
            source_id = f"camera_{source_counters['camera']}"
            source_counters["camera"] += 1

            try:
                system.add_camera_input(source_id, config={"device_id": device_id, "fps": 30})
                logger.info(f"✓ Added camera '{source_id}' (device {device_id})")
            except Exception as e:
                logger.error(f"✗ Failed to add camera: {e}")

        elif source_type == "text":
            source_id = f"text_{source_counters['text']}"
            source_counters["text"] += 1

            system.add_text_input(source_id, config={"prompt": f"[{source_id}]> "})
            logger.info(f"✓ Added text input '{source_id}'")

        elif source_type == "audio":
            source_id = f"audio_{source_counters['audio']}"
            source_counters["audio"] += 1

            system.add_audio_input(source_id, config={"sample_rate": 16000})
            logger.info(f"✓ Added audio input '{source_id}'")

        else:
            logger.warning(f"Unknown source type: {source_type}")

    elif cmd == "remove":
        if len(parts) < 2:
            logger.warning("Usage: remove <source_id>")
            return

        source_id = parts[1]
        system.remove_input_source(source_id)
        logger.info(f"✓ Removed source '{source_id}'")

    elif cmd == "help":
        print_help()

    else:
        logger.warning(f"Unknown command: {cmd}. Type 'help' for available commands.")


def print_help() -> None:
    """Print available commands."""
    logger.info("=== Available Commands ===")
    logger.info("  list                     - List all active input sources")
    logger.info("  status                   - Show system status")
    logger.info("  add camera [device_id]   - Add a camera source")
    logger.info("  add text                 - Add a text input source")
    logger.info("  add audio                - Add an audio input source")
    logger.info("  remove <source_id>       - Remove a specific source")
    logger.info("  help                     - Show this help message")
    logger.info("  quit/exit                - Exit the system")


def handle_text_input(event: IEvent) -> None:
    """Handle text input events as commands."""
    text = event.data["text"].strip()

    # Only process commands from the control console
    if event.source == "control":
        process_command(text)
    else:
        # Log other text inputs normally
        logger.info(f"💬 Text from {event.source}: '{text}'")


def main() -> None:
    """Main function."""
    global system

    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    logger.info("=== Dynamic Source Management Example ===")
    logger.info("This example demonstrates adding and removing input sources at runtime")

    # Create configuration with minimal setup
    config = PerceptionConfig.default()
    config.log_level = LogLevel.INFO

    # Create system
    system = PerceptionSystem(config)

    # Subscribe to events
    system.subscribe("camera.frame", handle_event)
    system.subscribe("text.input", handle_text_input)
    system.subscribe("audio.sample", handle_event)

    # Start system
    system.start()

    # Add a control console for commands
    system.add_text_input("control", config={"prompt": "cmd> "})
    logger.info("✓ Control console added")

    logger.info("\n=== System Ready ===")
    print_help()
    logger.info("\nStarting with minimal configuration. Use commands to add/remove sources.")
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
