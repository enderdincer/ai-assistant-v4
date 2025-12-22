"""Base class for input sources."""

import threading
from abc import ABC, abstractmethod
from typing import Any, Optional
from ai_assistant.shared.interfaces import IEventBus
from ai_assistant.shared.logging import get_logger, log_context

logger = get_logger(__name__)


class BaseInputSource(ABC):
    """Abstract base class for input sources.

    This class provides common functionality for all input sources including:
    - Lifecycle management (initialize, start, stop)
    - Threading support
    - Event bus integration
    - Configuration management
    """

    def __init__(
        self,
        source_id: str,
        source_type: str,
        event_bus: IEventBus,
        config: Optional[dict[str, Any]] = None,
    ) -> None:
        """Initialize the input source.

        Args:
            source_id: Unique identifier for this source
            source_type: Type of input source (e.g., 'camera', 'audio')
            event_bus: Event bus for publishing events
            config: Optional configuration dictionary
        """
        self._source_id = source_id
        self._source_type = source_type
        self._event_bus = event_bus
        self._config = config or {}
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    @property
    def source_id(self) -> str:
        """Get the source identifier."""
        return self._source_id

    @property
    def source_type(self) -> str:
        """Get the source type."""
        return self._source_type

    def get_config(self) -> dict[str, Any]:
        """Get the source configuration."""
        return self._config.copy()

    def initialize(self) -> None:
        """Initialize the input source.

        Override this method to perform initialization tasks like
        opening devices, loading resources, etc.
        """
        with log_context(source_id=self._source_id, source_type=self._source_type):
            logger.info(f"Initializing {self._source_type} source: {self._source_id}")
            self._initialize_source()

    def start(self) -> None:
        """Start the input source.

        This creates and starts a thread that runs the capture loop.
        """
        if self._running:
            raise RuntimeError(f"Source {self._source_id} is already running")

        with log_context(source_id=self._source_id):
            logger.info(f"Starting {self._source_type} source: {self._source_id}")

            self._running = True
            self._stop_event.clear()

            self._thread = threading.Thread(
                target=self._run_capture_loop,
                name=f"{self._source_type}-{self._source_id}",
                daemon=False,
            )
            self._thread.start()

            logger.info(f"Source {self._source_id} started in thread {self._thread.name}")

    def stop(self) -> None:
        """Stop the input source gracefully."""
        if not self._running:
            return

        with log_context(source_id=self._source_id):
            logger.info(f"Stopping {self._source_type} source: {self._source_id}")

            self._running = False
            self._stop_event.set()

            if self._thread and self._thread.is_alive():
                self._thread.join(timeout=5.0)
                if self._thread.is_alive():
                    logger.warning(f"Source {self._source_id} thread did not stop cleanly")

            self._cleanup_source()
            logger.info(f"Source {self._source_id} stopped")

    def is_running(self) -> bool:
        """Check if the source is running."""
        return self._running

    def _run_capture_loop(self) -> None:
        """Run the capture loop in the thread."""
        with log_context(source_id=self._source_id, source_type=self._source_type):
            logger.info(f"Capture loop started for {self._source_id}")

            try:
                while self._running and not self._stop_event.is_set():
                    self._capture_and_publish()
            except Exception as e:
                logger.error(f"Error in capture loop for {self._source_id}: {e}", exc_info=True)
                self._running = False
            finally:
                logger.info(f"Capture loop ended for {self._source_id}")

    @abstractmethod
    def _initialize_source(self) -> None:
        """Initialize the specific input source.

        Override this method to perform source-specific initialization.
        """
        pass

    @abstractmethod
    def _cleanup_source(self) -> None:
        """Clean up the specific input source.

        Override this method to perform source-specific cleanup.
        """
        pass

    @abstractmethod
    def _capture_and_publish(self) -> None:
        """Capture data and publish an event.

        Override this method to implement the specific capture logic.
        This method is called repeatedly in the capture loop.
        """
        pass
