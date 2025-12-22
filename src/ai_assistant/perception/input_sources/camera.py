"""Camera input source implementation."""

import time
from typing import Any, Optional
import cv2
import numpy as np
import numpy.typing as npt
from ai_assistant.shared.interfaces import IEventBus, EventPriority
from ai_assistant.shared.events import CameraFrameEvent
from ai_assistant.shared.logging import get_logger
from ai_assistant.perception.input_sources.base import BaseInputSource

logger = get_logger(__name__)


class CameraInputSource(BaseInputSource):
    """Camera input source that captures video frames.

    Configuration:
        - device_id: Camera device ID (default: 0)
        - fps: Target frames per second (default: 30)
        - width: Frame width (default: 640)
        - height: Frame height (default: 480)
        - priority: Event priority (default: NORMAL)
    """

    def __init__(
        self,
        source_id: str,
        event_bus: IEventBus,
        config: Optional[dict[str, Any]] = None,
    ) -> None:
        """Initialize camera input source.

        Args:
            source_id: Unique identifier for this camera
            event_bus: Event bus for publishing frames
            config: Camera configuration
        """
        super().__init__(source_id, "camera", event_bus, config)

        self._device_id = self._config.get("device_id", 0)
        self._target_fps = self._config.get("fps", 30)
        self._width = self._config.get("width", 640)
        self._height = self._config.get("height", 480)
        self._priority = self._config.get("priority", EventPriority.NORMAL)

        self._capture: Optional[cv2.VideoCapture] = None
        self._frame_count = 0
        self._frame_delay = 1.0 / self._target_fps

    def _initialize_source(self) -> None:
        """Initialize the camera device."""
        logger.info(f"Opening camera device {self._device_id}")

        self._capture = cv2.VideoCapture(self._device_id)

        if not self._capture.isOpened():
            raise RuntimeError(f"Failed to open camera device {self._device_id}")

        # Set camera properties
        self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
        self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
        self._capture.set(cv2.CAP_PROP_FPS, self._target_fps)

        # Read actual properties
        actual_width = int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = self._capture.get(cv2.CAP_PROP_FPS)

        logger.info(
            f"Camera {self._source_id} initialized: "
            f"{actual_width}x{actual_height} @ {actual_fps:.1f} FPS"
        )

    def _cleanup_source(self) -> None:
        """Release the camera device."""
        if self._capture is not None:
            self._capture.release()
            self._capture = None
            logger.info(f"Camera {self._source_id} released")

    def _capture_and_publish(self) -> None:
        """Capture a frame and publish it as an event."""
        if self._capture is None:
            return

        start_time = time.time()

        # Capture frame
        ret, frame = self._capture.read()

        if not ret or frame is None:
            logger.warning(f"Failed to capture frame from {self._source_id}")
            time.sleep(self._frame_delay)
            return

        self._frame_count += 1

        # Create and publish event
        event = CameraFrameEvent.create(
            source=self._source_id,
            frame=frame,
            frame_number=self._frame_count,
            priority=self._priority,
        )

        try:
            self._event_bus.publish(event)
            logger.debug(
                f"Published frame {self._frame_count} from {self._source_id} (shape: {frame.shape})"
            )
        except Exception as e:
            logger.error(f"Failed to publish frame: {e}")

        # Maintain target FPS
        elapsed = time.time() - start_time
        sleep_time = max(0, self._frame_delay - elapsed)
        if sleep_time > 0:
            time.sleep(sleep_time)
