#!/usr/bin/env python3
"""Real vision processor test with camera input and VLM description.

This test uses real camera input via OpenCV and the Qwen3-VL model
via HuggingFace Transformers. No mocking - everything is production code.

Requirements:
    pip install -e ".[vision]"
    # or manually:
    pip install torch transformers accelerate qwen-vl-utils opencv-python pillow

Usage:
    python test_vision_real.py

The system will:
    1. Initialize the Qwen3-VL model (downloads ~4GB on first run)
    2. Start capturing video from your webcam
    3. Display the camera feed in a window
    4. Generate descriptions when scene changes (>8% threshold)
    5. Print descriptions in real-time
    6. Press 'q' in the camera window to stop
"""

import time
import cv2
from datetime import datetime
from ai_assistant.perception import PerceptionSystem, PerceptionConfig
from ai_assistant.shared.logging import LogLevel, get_logger
from ai_assistant.shared.interfaces import IEvent

logger = get_logger(__name__)

# Vision results
descriptions = []

# Latest frame for display (updated by camera callback)
latest_frame = None
latest_description = ""


def handle_camera_frame(event: IEvent) -> None:
    """Handle camera frames for display."""
    global latest_frame
    frame = event.data.get("frame")
    if frame is not None:
        latest_frame = frame.copy()


def handle_vision_description(event: IEvent) -> None:
    """Handle vision description events from the VLM processor."""
    global latest_description

    description = event.data.get("description", "")
    frame_number = event.data.get("frame_number", 0)
    frame_timestamp = event.data.get("frame_timestamp", 0.0)
    processing_time = event.data.get("processing_time", 0.0)
    change_score = event.data.get("change_score", 0.0)
    model_name = event.data.get("model_name", "unknown")

    # Format timestamp
    if frame_timestamp > 0:
        ts_str = datetime.fromtimestamp(frame_timestamp).strftime("%H:%M:%S")
    else:
        ts_str = "N/A"

    descriptions.append(
        {
            "description": description,
            "frame_number": frame_number,
            "timestamp": frame_timestamp,
            "processing_time": processing_time,
            "change_score": change_score,
        }
    )

    # Update latest description for overlay
    latest_description = description

    # Print to console
    logger.info("\n" + "=" * 70)
    logger.info(
        f"Frame #{frame_number} | Change: {change_score:.3f} | Time: {processing_time:.2f}s"
    )
    logger.info(f"Timestamp: {ts_str}")
    logger.info("=" * 70)
    logger.info(description)
    logger.info("=" * 70 + "\n")


def check_dependencies() -> bool:
    """Check if all required dependencies are installed."""
    missing = []

    try:
        import torch

        logger.info(f"  torch: {torch.__version__}")
        if torch.backends.mps.is_available():
            logger.info("    MPS (Metal) available: Yes")
        elif torch.cuda.is_available():
            logger.info(f"    CUDA available: Yes ({torch.cuda.get_device_name(0)})")
        else:
            logger.info("    Running on CPU")
    except ImportError:
        missing.append("torch")

    try:
        import transformers

        logger.info(f"  transformers: {transformers.__version__}")
    except ImportError:
        missing.append("transformers")

    try:
        import accelerate

        logger.info(f"  accelerate: {accelerate.__version__}")
    except ImportError:
        missing.append("accelerate")

    try:
        import qwen_vl_utils

        logger.info("  qwen-vl-utils: installed")
    except ImportError:
        missing.append("qwen-vl-utils")

    try:
        import cv2

        logger.info(f"  opencv-python: {cv2.__version__}")
    except ImportError:
        missing.append("opencv-python")

    try:
        from PIL import Image

        import PIL

        logger.info(f"  pillow: {PIL.__version__}")
    except ImportError:
        missing.append("pillow")

    if missing:
        logger.error(f"\nMissing required dependencies: {', '.join(missing)}")
        logger.error('Install with: pip install -e ".[vision]"')
        return False

    return True


def display_camera_feed() -> bool:
    """Display camera feed with optional description overlay.

    Returns:
        False if 'q' was pressed to quit, True otherwise
    """
    global latest_frame, latest_description

    if latest_frame is None:
        return True

    # Create a copy for display
    display_frame = latest_frame.copy()

    # Add description overlay if we have one
    if latest_description:
        # Wrap text for display
        max_width = 60
        words = latest_description.split()
        lines = []
        current_line = ""

        for word in words:
            if len(current_line) + len(word) + 1 <= max_width:
                current_line += (" " if current_line else "") + word
            else:
                if current_line:
                    lines.append(current_line)
                current_line = word

        if current_line:
            lines.append(current_line)

        # Draw semi-transparent background
        overlay = display_frame.copy()
        h = display_frame.shape[0]
        text_height = 25 * min(len(lines), 4) + 20
        cv2.rectangle(overlay, (0, h - text_height), (display_frame.shape[1], h), (0, 0, 0), -1)
        display_frame = cv2.addWeighted(overlay, 0.6, display_frame, 0.4, 0)

        # Draw text
        for i, line in enumerate(lines[:4]):  # Max 4 lines
            y = h - text_height + 25 + i * 25
            cv2.putText(
                display_frame,
                line,
                (10, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

    # Show frame
    cv2.imshow("Vision Test - Press 'q' to quit", display_frame)

    # Check for 'q' key
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        return False

    return True


def main():
    """Main test function for real vision processing."""
    global latest_frame

    logger.info("=" * 70)
    logger.info("REAL VISION PROCESSOR TEST")
    logger.info("=" * 70)
    logger.info("\nThis test uses:")
    logger.info("  - Real camera input from webcam (OpenCV)")
    logger.info("  - Qwen3-VL-2B-Instruct model (HuggingFace Transformers)")
    logger.info("  - Change detection with Jaccard distance")
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
    config.max_queue_size = 1000  # Larger queue to handle frame bursts

    # Create system
    system = PerceptionSystem(config)

    # Subscribe to events
    system.subscribe("camera.frame", handle_camera_frame)
    system.subscribe("vision.description", handle_vision_description)

    # Start system
    logger.info("Starting perception system...")
    system.start()
    logger.info("System started\n")

    # Add camera input
    logger.info("Setting up camera input...")
    try:
        system.add_camera_input(
            "webcam",
            {
                "device_id": 0,  # Default webcam
                "fps": 30,
                "width": 416,  # Lower resolution for faster VLM processing
                "height": 312,  # Maintains 4:3 aspect ratio
            },
        )
        logger.info("  Camera configured: 416x312 @ 30 FPS")
    except Exception as e:
        logger.error(f"\nFailed to initialize camera: {e}")
        logger.error("Make sure your webcam is connected and not in use by another app")
        system.stop()
        return 1

    # Add vision processor
    logger.info("\nLoading vision model...")
    logger.info("  Model: Qwen/Qwen3-VL-2B-Instruct")
    logger.info("  Frame skip: 30 (process 1 FPS at 30 FPS camera)")
    logger.info("  Change threshold: 0.08 (8%)")
    logger.info("\nPlease wait while the model is loaded...")
    logger.info("  (First run will download ~4GB model)")

    try:
        start_load = time.time()

        system.add_vision_processor(
            "vision_main",
            {
                "model_name": "Qwen/Qwen3-VL-2B-Instruct",
                "device": "auto",  # Auto-detect MPS/CUDA/CPU
                "quantization": "none",  # Use fp16 on MPS, int8 requires CUDA
                "frame_skip": 30,  # Process every 30th frame = 1 FPS
                "change_threshold": 0.08,  # 8% change threshold
                "prompt": "Describe what you see in this image, including people, objects, and activities.",
                "max_tokens": 50,  # Reduced for faster inference (~1.2-1.5s vs 3-5s)
            },
        )

        load_time = time.time() - start_load
        logger.info(f"Model loaded in {load_time:.1f}s\n")

    except Exception as e:
        logger.error(f"\nFailed to load vision model: {e}")
        logger.error("Make sure you have enough memory and correct dependencies")
        system.stop()
        cv2.destroyAllWindows()
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
    logger.info("READY TO ANALYZE")
    logger.info("=" * 70)
    logger.info("\nThe system is now processing your webcam feed!")
    logger.info("\nHow it works:")
    logger.info("  - Camera captures at 30 FPS")
    logger.info("  - VLM processes every 30th frame (1 per second)")
    logger.info("  - Description emitted when scene changes >8%")
    logger.info("  - Description overlaid on camera window")
    logger.info("\nControls:")
    logger.info("  - Press 'q' in the camera window to quit")
    logger.info("\n" + "=" * 70 + "\n")

    # Run and display camera feed
    start_time = time.time()
    last_status_time = start_time

    try:
        while True:
            # Display camera feed (returns False if 'q' pressed)
            if not display_camera_feed():
                logger.info("\n'q' pressed, stopping...")
                break

            current_time = time.time()

            # Show periodic status every 15 seconds
            if current_time - last_status_time >= 15:
                elapsed = int(current_time - start_time)
                logger.info(f"\nRunning for {elapsed}s | Descriptions: {len(descriptions)}\n")
                last_status_time = current_time

            # Small sleep to prevent CPU spinning
            time.sleep(0.01)

    except KeyboardInterrupt:
        logger.info("\n\nCtrl+C pressed, stopping...")

    # Cleanup
    cv2.destroyAllWindows()

    # Stop system
    logger.info("\nShutting down...")
    system.stop()

    # Print summary
    logger.info("\n" + "=" * 70)
    logger.info("TEST SUMMARY")
    logger.info("=" * 70)
    logger.info(f"Total descriptions: {len(descriptions)}")
    logger.info(f"Test duration: {time.time() - start_time:.1f}s")

    if descriptions:
        logger.info("\nAll descriptions:")
        for i, desc in enumerate(descriptions, 1):
            logger.info(f"\n  #{i} (frame {desc['frame_number']}):")
            logger.info(f"    Change: {desc['change_score']:.3f}")
            logger.info(f"    Time: {desc['processing_time']:.2f}s")
            logger.info(f"    Description: {desc['description'][:100]}...")

        # Calculate average processing time
        avg_time = sum(d["processing_time"] for d in descriptions) / len(descriptions)
        logger.info(f"\n  Average processing time: {avg_time:.2f}s")
    else:
        logger.warning("\nNo descriptions were generated")
        logger.warning("  Possible reasons:")
        logger.warning("    - Camera not working")
        logger.warning("    - Scene not changing enough (try moving)")
        logger.warning("    - Model loading failed")

    logger.info("\n" + "=" * 70)
    logger.info("Test complete!")
    logger.info("=" * 70 + "\n")

    return 0


if __name__ == "__main__":
    exit(main())
