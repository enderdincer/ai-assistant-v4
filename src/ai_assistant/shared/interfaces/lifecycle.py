"""Lifecycle management interface for all components."""

from typing import Protocol, runtime_checkable


@runtime_checkable
class ILifecycle(Protocol):
    """Interface for components with lifecycle management.

    All components that need initialization, startup, shutdown
    should implement this interface.
    """

    def initialize(self) -> None:
        """Initialize the component.

        This is called once before start(). Use this for one-time setup
        like loading configuration, creating resources, etc.

        Raises:
            RuntimeError: If initialization fails
        """
        ...

    def start(self) -> None:
        """Start the component.

        This begins the component's active operation (e.g., starting threads,
        beginning to process events, etc.).

        Raises:
            RuntimeError: If component is already started or not initialized
        """
        ...

    def stop(self) -> None:
        """Stop the component gracefully.

        This signals the component to stop its operations. Should be idempotent.
        After calling stop(), the component should eventually reach a stopped state.

        Raises:
            RuntimeError: If component is not started
        """
        ...

    def is_running(self) -> bool:
        """Check if the component is currently running.

        Returns:
            bool: True if component is running, False otherwise
        """
        ...
