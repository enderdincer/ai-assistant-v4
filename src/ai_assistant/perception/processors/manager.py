"""Manager for coordinating multiple sensory processors."""

import threading
from typing import Any, Optional

from ai_assistant.shared.interfaces import IProcessor
from ai_assistant.shared.logging import get_logger, log_context

logger = get_logger(__name__)


class ProcessorManager:
    """Manages lifecycle of sensory processors.

    The ProcessorManager provides centralized control over processor
    registration, initialization, and lifecycle management. It ensures
    processors are started and stopped in proper order and provides
    visibility into processor status.
    """

    def __init__(self) -> None:
        """Initialize the processor manager."""
        self._processors: dict[str, IProcessor] = {}
        self._lock = threading.Lock()
        self._initialized = False
        self._running = False

    def register_processor(self, processor: IProcessor) -> None:
        """Register a processor.

        Args:
            processor: Processor instance to register

        Raises:
            ValueError: If processor_id is already registered
        """
        processor_id = processor.processor_id

        with self._lock:
            if processor_id in self._processors:
                raise ValueError(f"Processor '{processor_id}' is already registered")

            self._processors[processor_id] = processor

            with log_context(processor_id=processor_id, processor_type=processor.processor_type):
                logger.info(
                    f"Registered processor: {processor_id} (type: {processor.processor_type})"
                )

    def unregister_processor(self, processor_id: str) -> None:
        """Unregister a processor.

        The processor must be stopped before unregistering.

        Args:
            processor_id: ID of processor to unregister

        Raises:
            ValueError: If processor_id is not found
            RuntimeError: If processor is still running
        """
        with self._lock:
            if processor_id not in self._processors:
                raise ValueError(f"Processor '{processor_id}' is not registered")

            processor = self._processors[processor_id]

            if processor.is_running():
                raise RuntimeError(
                    f"Cannot unregister running processor '{processor_id}'. Stop it first."
                )

            del self._processors[processor_id]

            with log_context(processor_id=processor_id):
                logger.info(f"Unregistered processor: {processor_id}")

    def get_processor(self, processor_id: str) -> Optional[IProcessor]:
        """Get a processor by ID.

        Args:
            processor_id: ID of processor to retrieve

        Returns:
            Optional[IProcessor]: Processor instance or None if not found
        """
        return self._processors.get(processor_id)

    def list_processors(self) -> dict[str, str]:
        """List all registered processors.

        Returns:
            dict[str, str]: Mapping of processor_id to processor_type
        """
        return {
            processor_id: processor.processor_type
            for processor_id, processor in self._processors.items()
        }

    def initialize_all(self) -> None:
        """Initialize all registered processors.

        If any processor fails to initialize, an error is logged but other
        processors will still be attempted.

        Raises:
            RuntimeError: If manager is already initialized
        """
        with self._lock:
            if self._initialized:
                raise RuntimeError("Processor manager is already initialized")

            logger.info(f"Initializing {len(self._processors)} processor(s)")

            success_count = 0
            failure_count = 0

            for processor_id, processor in self._processors.items():
                try:
                    with log_context(processor_id=processor_id):
                        processor.initialize()
                        success_count += 1
                except Exception as e:
                    failure_count += 1
                    logger.error(
                        f"Failed to initialize processor '{processor_id}': {e}", exc_info=True
                    )

            self._initialized = True
            logger.info(
                f"Processor initialization complete: "
                f"{success_count} succeeded, {failure_count} failed"
            )

    def start_all(self) -> None:
        """Start all registered processors.

        Processors must be initialized before starting. If any processor fails
        to start, an error is logged but other processors will still be attempted.

        Raises:
            RuntimeError: If processors are not initialized or already running
        """
        with self._lock:
            if not self._initialized:
                raise RuntimeError("Processors must be initialized before starting")

            if self._running:
                raise RuntimeError("Processors are already running")

            logger.info(f"Starting {len(self._processors)} processor(s)")

            success_count = 0
            failure_count = 0

            for processor_id, processor in self._processors.items():
                try:
                    with log_context(processor_id=processor_id):
                        processor.start()
                        success_count += 1
                except Exception as e:
                    failure_count += 1
                    logger.error(f"Failed to start processor '{processor_id}': {e}", exc_info=True)

            self._running = True
            logger.info(
                f"Processor startup complete: {success_count} succeeded, {failure_count} failed"
            )

    def stop_all(self) -> None:
        """Stop all processors in reverse order of registration.

        Stopping in reverse order helps ensure cleanup happens in proper
        dependency order.
        """
        with self._lock:
            if not self._running:
                return

            logger.info(f"Stopping {len(self._processors)} processor(s)")

            # Stop in reverse order
            for processor_id, processor in reversed(list(self._processors.items())):
                try:
                    with log_context(processor_id=processor_id):
                        processor.stop()
                except Exception as e:
                    logger.error(f"Failed to stop processor '{processor_id}': {e}", exc_info=True)

            self._running = False
            logger.info("All processors stopped")

    def start_processor(self, processor_id: str) -> None:
        """Start a specific processor.

        Args:
            processor_id: ID of processor to start

        Raises:
            ValueError: If processor_id is not found
            RuntimeError: If processor is not initialized
        """
        processor = self._processors.get(processor_id)

        if processor is None:
            raise ValueError(f"Processor '{processor_id}' is not registered")

        with log_context(processor_id=processor_id):
            logger.info(f"Starting processor: {processor_id}")
            processor.start()

    def stop_processor(self, processor_id: str) -> None:
        """Stop a specific processor.

        Args:
            processor_id: ID of processor to stop

        Raises:
            ValueError: If processor_id is not found
        """
        processor = self._processors.get(processor_id)

        if processor is None:
            raise ValueError(f"Processor '{processor_id}' is not registered")

        with log_context(processor_id=processor_id):
            logger.info(f"Stopping processor: {processor_id}")
            processor.stop()

    def get_status(self) -> dict[str, Any]:
        """Get status of all processors.

        Returns:
            dict[str, Any]: Status information including:
                - processor_count: Total number of registered processors
                - initialized: Whether all processors are initialized
                - running: Whether processors are running
                - processors: Dict of processor details
        """
        return {
            "processor_count": len(self._processors),
            "initialized": self._initialized,
            "running": self._running,
            "processors": {
                pid: {
                    "type": p.processor_type,
                    "running": p.is_running(),
                    "input_types": p.input_event_types,
                    "output_types": p.output_event_types,
                }
                for pid, p in self._processors.items()
            },
        }

    def is_running(self) -> bool:
        """Check if manager has running processors.

        Returns:
            bool: True if any processor is running
        """
        return self._running

    def __len__(self) -> int:
        """Get number of registered processors."""
        return len(self._processors)

    def __contains__(self, processor_id: str) -> bool:
        """Check if processor is registered."""
        return processor_id in self._processors
