"""Base class for sensory processors."""

import asyncio
import threading
from abc import ABC, abstractmethod
from typing import Any, Optional

from ai_assistant.shared.interfaces import IEventBus, IEvent, IProcessor
from ai_assistant.shared.logging import get_logger, log_context

logger = get_logger(__name__)


class BaseProcessor(IProcessor, ABC):
    """Abstract base class for sensory processors.

    This class provides common functionality for all processors including:
    - Automatic event subscription/unsubscription
    - Lifecycle management (initialize, start, stop)
    - Thread-safe error handling
    - Event processing and publishing
    - Configuration management

    Subclasses need only implement the _process_event() method to define
    their specific processing logic.
    """

    def __init__(
        self,
        processor_id: str,
        processor_type: str,
        event_bus: IEventBus,
        input_event_types: list[str],
        output_event_types: list[str],
        config: Optional[dict[str, Any]] = None,
    ) -> None:
        """Initialize the processor.

        Args:
            processor_id: Unique identifier for this processor
            processor_type: Type of processor (e.g., 'stt', 'object_detection')
            event_bus: Event bus for subscribing and publishing events
            input_event_types: Event types to subscribe to
            output_event_types: Event types this processor produces
            config: Optional configuration dictionary
        """
        self._processor_id = processor_id
        self._processor_type = processor_type
        self._event_bus = event_bus
        self._input_event_types = input_event_types
        self._output_event_types = output_event_types
        self._config = config or {}
        self._running = False
        self._initialized = False
        self._lock = threading.Lock()
        self._logger = get_logger(f"{__name__}.{processor_id}")

    @property
    def processor_id(self) -> str:
        """Get unique identifier for this processor."""
        return self._processor_id

    @property
    def processor_type(self) -> str:
        """Get the type/category of this processor."""
        return self._processor_type

    @property
    def input_event_types(self) -> list[str]:
        """Get event types this processor subscribes to."""
        return self._input_event_types.copy()

    @property
    def output_event_types(self) -> list[str]:
        """Get event types this processor produces."""
        return self._output_event_types.copy()

    def get_config(self) -> dict[str, Any]:
        """Get the processor configuration."""
        return self._config.copy()

    def initialize(self) -> None:
        """Initialize the processor.

        Override _initialize_processor() to add custom initialization logic.

        Raises:
            RuntimeError: If initialization fails
        """
        with self._lock:
            if self._initialized:
                raise RuntimeError(f"Processor {self._processor_id} is already initialized")

            with log_context(processor_id=self._processor_id, processor_type=self._processor_type):
                self._logger.info(
                    f"Initializing {self._processor_type} processor: {self._processor_id}"
                )
                try:
                    self._validate_config()
                    self._initialize_processor()
                    self._initialized = True
                    self._logger.info(f"Processor {self._processor_id} initialized successfully")
                except Exception as e:
                    self._logger.error(f"Failed to initialize processor {self._processor_id}: {e}")
                    raise RuntimeError(f"Processor initialization failed: {e}") from e

    def start(self) -> None:
        """Start the processor and subscribe to events.

        Raises:
            RuntimeError: If processor is not initialized or already running
        """
        with self._lock:
            if not self._initialized:
                raise RuntimeError(f"Processor {self._processor_id} is not initialized")

            if self._running:
                raise RuntimeError(f"Processor {self._processor_id} is already running")

            with log_context(processor_id=self._processor_id):
                self._logger.info(
                    f"Starting {self._processor_type} processor: {self._processor_id}"
                )

                # Subscribe to input events
                for event_type in self._input_event_types:
                    self._event_bus.subscribe(event_type, self._handle_event)
                    self._logger.debug(f"Subscribed to event type: {event_type}")

                self._running = True
                self._logger.info(f"Processor {self._processor_id} started successfully")

    def stop(self) -> None:
        """Stop the processor and unsubscribe from events."""
        with self._lock:
            if not self._running:
                return

            with log_context(processor_id=self._processor_id):
                self._logger.info(
                    f"Stopping {self._processor_type} processor: {self._processor_id}"
                )

                # Unsubscribe from input events
                for event_type in self._input_event_types:
                    try:
                        self._event_bus.unsubscribe(event_type, self._handle_event)
                        self._logger.debug(f"Unsubscribed from event type: {event_type}")
                    except Exception as e:
                        self._logger.warning(f"Failed to unsubscribe from {event_type}: {e}")

                self._running = False

                # Allow subclasses to cleanup
                try:
                    self._cleanup_processor()
                except Exception as e:
                    self._logger.error(f"Error during processor cleanup: {e}")

                self._logger.info(f"Processor {self._processor_id} stopped successfully")

    def is_running(self) -> bool:
        """Check if the processor is currently running."""
        return self._running

    def process(self, event: IEvent) -> list[IEvent]:
        """Process an event synchronously.

        This is implemented by calling the abstract _process_event() method.
        Subclasses should not override this method.

        Args:
            event: Input event to process

        Returns:
            list[IEvent]: List of output events (may be empty)
        """
        return self._process_event(event)

    async def process_async(self, event: IEvent) -> list[IEvent]:
        """Process an event asynchronously.

        Default implementation runs sync process() in thread pool.
        Override _process_event_async() for true async processing.

        Args:
            event: Input event to process

        Returns:
            list[IEvent]: List of output events (may be empty)
        """
        # Check if subclass provides async implementation
        if hasattr(self, "_process_event_async"):
            return await self._process_event_async(event)

        # Fall back to running sync process in thread pool
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._process_event, event)

    def _handle_event(self, event: IEvent) -> None:
        """Internal event handler with error handling.

        This method is called by the event bus when an event is published.
        It wraps the user's processing logic with error handling.

        Args:
            event: Event to process
        """
        if not self._running:
            return

        try:
            with log_context(
                processor_id=self._processor_id,
                event_type=event.event_type,
                source=event.source,
            ):
                self._logger.debug(f"Processing event: {event.event_type} from {event.source}")

                # Process the event
                output_events = self._process_event(event)

                # Publish output events
                for output_event in output_events:
                    self._event_bus.publish(output_event)
                    self._logger.debug(
                        f"Published {output_event.event_type} event from processor {self._processor_id}"
                    )

                self._logger.debug(f"Processed event produced {len(output_events)} output event(s)")

        except Exception as e:
            self._logger.error(
                f"Error processing event {event.event_type} from {event.source}: {e}", exc_info=True
            )
            # Don't re-raise - we don't want to crash the event bus

    def _validate_config(self) -> None:
        """Validate processor configuration.

        Override this method to validate configuration parameters.
        Raise ValueError if configuration is invalid.

        Raises:
            ValueError: If configuration is invalid
        """
        pass  # Default: no validation required

    def _initialize_processor(self) -> None:
        """Initialize processor-specific resources.

        Override this method to perform initialization tasks like
        loading models, creating buffers, etc.

        Raises:
            Exception: If initialization fails
        """
        pass  # Default: no initialization required

    def _cleanup_processor(self) -> None:
        """Cleanup processor-specific resources.

        Override this method to perform cleanup tasks like
        releasing resources, closing connections, etc.
        """
        pass  # Default: no cleanup required

    @abstractmethod
    def _process_event(self, event: IEvent) -> list[IEvent]:
        """Process an event and return output events.

        This is the main processing logic that subclasses must implement.

        Args:
            event: Input event to process

        Returns:
            list[IEvent]: List of output events (may be empty)

        Raises:
            Exception: If processing fails
        """
        pass
