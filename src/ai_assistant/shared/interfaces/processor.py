"""Processor interface for sensory data processing."""

from typing import Protocol, runtime_checkable, Awaitable

from ai_assistant.shared.interfaces.lifecycle import ILifecycle
from ai_assistant.shared.interfaces.event import IEvent


@runtime_checkable
class IProcessor(ILifecycle, Protocol):
    """Interface for sensory processors.

    Processors are components that subscribe to raw sensor events,
    process them, and publish derived events. For example, an STT
    processor subscribes to audio events and publishes transcription events.

    A single input source can have multiple processors attached to it,
    each producing different types of events from the same input.
    """

    @property
    def processor_id(self) -> str:
        """Get unique identifier for this processor.

        Returns:
            str: Unique processor ID
        """
        ...

    @property
    def processor_type(self) -> str:
        """Get the type/category of this processor.

        Returns:
            str: Processor type (e.g., 'stt', 'object_detection', 'frequency_detector')
        """
        ...

    @property
    def input_event_types(self) -> list[str]:
        """Get event types this processor subscribes to.

        Returns:
            list[str]: List of input event types (e.g., ['audio.sample'])
        """
        ...

    @property
    def output_event_types(self) -> list[str]:
        """Get event types this processor produces.

        Returns:
            list[str]: List of output event types (e.g., ['audio.transcription'])
        """
        ...

    def process(self, event: IEvent) -> list[IEvent]:
        """Process an event synchronously.

        Takes a raw sensor event and produces zero or more processed events.
        Returning an empty list effectively filters out the event.

        Args:
            event: Input event to process

        Returns:
            list[IEvent]: List of output events (may be empty)

        Raises:
            Exception: If processing fails
        """
        ...

    async def process_async(self, event: IEvent) -> list[IEvent]:
        """Process an event asynchronously.

        Async variant of process() for processors that need to perform
        async operations (e.g., calling async APIs, using async I/O).

        Args:
            event: Input event to process

        Returns:
            list[IEvent]: List of output events (may be empty)

        Raises:
            Exception: If processing fails
        """
        ...
