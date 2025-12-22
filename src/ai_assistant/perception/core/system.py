"""Main perception system facade."""

from typing import Dict, Optional, Any
from ai_assistant.shared.logging import (
    setup_logging,
    LogConfig,
    get_logger,
)
from ai_assistant.shared.events import EventBus
from ai_assistant.shared.threading import ThreadManager
from ai_assistant.shared.interfaces import IEvent, EventHandler
from ai_assistant.perception.input_sources import (
    BaseInputSource,
    CameraInputSource,
    TextInputSource,
    AudioInputSource,
)
from ai_assistant.perception.core.config import PerceptionConfig

logger = get_logger(__name__)


class PerceptionSystem:
    """High-level facade for the perception system.

    This class provides a simple API for managing the entire perception system,
    including the event bus, thread manager, and all input sources.

    Example:
        ```python
        # Create and configure system
        config = PerceptionConfig.with_camera(device_id=0, fps=30)
        system = PerceptionSystem(config)

        # Subscribe to events
        system.subscribe('camera.frame', handle_frame)

        # Start the system
        system.start()

        # Add another input source dynamically
        system.add_text_input('console')

        # Stop the system
        system.stop()
        ```
    """

    def __init__(self, config: Optional[PerceptionConfig] = None) -> None:
        """Initialize the perception system.

        Args:
            config: Configuration for the system (uses default if None)
        """
        self._config = config or PerceptionConfig.default()

        # Setup logging
        log_config = LogConfig(
            level=self._config.log_level,
            log_file=self._config.log_file,
            console_output=True,
            colored_console=self._config.colored_console,
        )
        setup_logging(log_config)

        logger.info("Initializing perception system")

        # Create core components
        self._event_bus = EventBus(max_queue_size=self._config.max_queue_size)
        self._thread_manager = ThreadManager()
        self._input_sources: Dict[str, BaseInputSource] = {}

        # Register event bus with thread manager
        self._thread_manager.register_component("event_bus", self._event_bus)

        # Create thread pool for async operations
        self._thread_pool = self._thread_manager.create_thread_pool(
            "perception-pool",
            max_workers=self._config.max_worker_threads,
        )

        logger.info("Perception system initialized")

    def start(self) -> None:
        """Start the perception system.

        This starts the event bus, thread manager, and all configured input sources.
        """
        logger.info("Starting perception system")

        # Initialize and start all components
        self._thread_manager.initialize_all()
        self._thread_manager.start_all()

        # Create and start input sources from configuration
        for source_config in self._config.input_sources:
            source_type = source_config["type"]
            source_id = source_config["source_id"]
            config = source_config.get("config", {})

            if source_type == "camera":
                self.add_camera_input(source_id, config)
            elif source_type == "text":
                self.add_text_input(source_id, config)
            elif source_type == "audio":
                self.add_audio_input(source_id, config)
            else:
                logger.warning(f"Unknown input source type: {source_type}")

        logger.info("Perception system started")

    def stop(self) -> None:
        """Stop the perception system gracefully.

        This stops all input sources and shuts down the thread manager.
        """
        logger.info("Stopping perception system")

        # Stop all input sources
        for source_id, source in list(self._input_sources.items()):
            self.remove_input_source(source_id)

        # Stop thread manager (which stops event bus and pools)
        self._thread_manager.stop_all()

        logger.info("Perception system stopped")

    def subscribe(self, event_type: str, handler: EventHandler) -> None:
        """Subscribe to events of a specific type.

        Args:
            event_type: The type of events to subscribe to
            handler: Callback function to handle events
        """
        self._event_bus.subscribe(event_type, handler)

    def unsubscribe(self, event_type: str, handler: EventHandler) -> None:
        """Unsubscribe from events of a specific type.

        Args:
            event_type: The type of events to unsubscribe from
            handler: The handler to remove
        """
        self._event_bus.unsubscribe(event_type, handler)

    def add_camera_input(
        self, source_id: str, config: Optional[Dict[str, Any]] = None
    ) -> CameraInputSource:
        """Add a camera input source dynamically.

        Args:
            source_id: Unique identifier for the camera
            config: Camera configuration

        Returns:
            CameraInputSource: The created camera source
        """
        if source_id in self._input_sources:
            raise ValueError(f"Input source '{source_id}' already exists")

        logger.info(f"Adding camera input: {source_id}")

        camera = CameraInputSource(source_id, self._event_bus, config)
        camera.initialize()
        camera.start()

        self._input_sources[source_id] = camera
        self._thread_manager.register_component(f"input_{source_id}", camera)

        return camera

    def add_text_input(
        self, source_id: str, config: Optional[Dict[str, Any]] = None
    ) -> TextInputSource:
        """Add a text input source dynamically.

        Args:
            source_id: Unique identifier for the text source
            config: Text input configuration

        Returns:
            TextInputSource: The created text source
        """
        if source_id in self._input_sources:
            raise ValueError(f"Input source '{source_id}' already exists")

        logger.info(f"Adding text input: {source_id}")

        text = TextInputSource(source_id, self._event_bus, config)
        text.initialize()
        text.start()

        self._input_sources[source_id] = text
        self._thread_manager.register_component(f"input_{source_id}", text)

        return text

    def add_audio_input(
        self, source_id: str, config: Optional[Dict[str, Any]] = None
    ) -> AudioInputSource:
        """Add an audio input source dynamically.

        Args:
            source_id: Unique identifier for the audio source
            config: Audio input configuration

        Returns:
            AudioInputSource: The created audio source
        """
        if source_id in self._input_sources:
            raise ValueError(f"Input source '{source_id}' already exists")

        logger.info(f"Adding audio input: {source_id}")

        audio = AudioInputSource(source_id, self._event_bus, config)
        audio.initialize()
        audio.start()

        self._input_sources[source_id] = audio
        self._thread_manager.register_component(f"input_{source_id}", audio)

        return audio

    def remove_input_source(self, source_id: str) -> None:
        """Remove an input source.

        Args:
            source_id: Identifier of the source to remove
        """
        if source_id not in self._input_sources:
            logger.warning(f"Input source '{source_id}' not found")
            return

        logger.info(f"Removing input source: {source_id}")

        source = self._input_sources.pop(source_id)
        source.stop()
        self._thread_manager.unregister_component(f"input_{source_id}")

    def get_input_source(self, source_id: str) -> Optional[BaseInputSource]:
        """Get an input source by ID.

        Args:
            source_id: Identifier of the source

        Returns:
            Optional[BaseInputSource]: The input source or None if not found
        """
        return self._input_sources.get(source_id)

    def list_input_sources(self) -> Dict[str, str]:
        """List all active input sources.

        Returns:
            Dict[str, str]: Mapping of source IDs to source types
        """
        return {source_id: source.source_type for source_id, source in self._input_sources.items()}

    def get_status(self) -> Dict[str, Any]:
        """Get the status of the perception system.

        Returns:
            Dict: Status information including all components
        """
        return {
            "running": self._thread_manager.is_running(),
            "event_bus": {
                "running": self._event_bus.is_running(),
                "queue_size": self._event_bus.get_queue_size(),
            },
            "input_sources": {
                source_id: {
                    "type": source.source_type,
                    "running": source.is_running(),
                }
                for source_id, source in self._input_sources.items()
            },
            "thread_manager": self._thread_manager.get_status(),
        }
