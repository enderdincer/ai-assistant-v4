"""Interface for input sources that generate events."""

from typing import Any, Protocol, runtime_checkable
from ai_assistant.shared.interfaces.lifecycle import ILifecycle


@runtime_checkable
class IInputSource(ILifecycle, Protocol):
    """Interface for input sources that generate events.

    Input sources are components that capture data from external sources
    (cameras, microphones, text streams, etc.) and publish events.
    """

    @property
    def source_id(self) -> str:
        """Get the unique identifier for this input source.

        Returns:
            str: Unique source identifier
        """
        ...

    @property
    def source_type(self) -> str:
        """Get the type of this input source.

        Returns:
            str: Source type (e.g., 'camera', 'audio', 'text')
        """
        ...

    def get_config(self) -> dict[str, Any]:
        """Get the configuration of this input source.

        Returns:
            dict: Configuration parameters
        """
        ...
