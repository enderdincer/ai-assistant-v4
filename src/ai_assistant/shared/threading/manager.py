"""Central thread manager for the application."""

import threading
from typing import Any, Dict, List, Optional
from ai_assistant.shared.interfaces import ILifecycle
from ai_assistant.shared.threading.thread_info import ThreadInfo
from ai_assistant.shared.threading.thread_pool import ManagedThreadPool
from ai_assistant.shared.logging import get_logger

logger = get_logger(__name__)


class ThreadManager:
    """Central manager for all threads and lifecycle-managed components.

    This manager coordinates:
    - Component lifecycle (initialize, start, stop)
    - Thread pool management
    - Thread monitoring and health checks
    - Graceful shutdown coordination
    """

    def __init__(self) -> None:
        """Initialize the thread manager."""
        self._components: Dict[str, ILifecycle] = {}
        self._thread_pools: Dict[str, ManagedThreadPool] = {}
        self._threads: Dict[str, ThreadInfo] = {}
        self._lock = threading.RLock()
        self._running = False

    def register_component(self, name: str, component: ILifecycle) -> None:
        """Register a component for lifecycle management.

        Args:
            name: Unique name for the component
            component: The component to manage

        Raises:
            ValueError: If a component with this name already exists
        """
        with self._lock:
            if name in self._components:
                raise ValueError(f"Component '{name}' is already registered")

            self._components[name] = component
            logger.info(f"Registered component: {name}")

    def unregister_component(self, name: str) -> None:
        """Unregister a component.

        Args:
            name: Name of the component to unregister
        """
        with self._lock:
            if name in self._components:
                component = self._components.pop(name)
                if component.is_running():
                    component.stop()
                logger.info(f"Unregistered component: {name}")

    def create_thread_pool(
        self,
        name: str,
        max_workers: Optional[int] = None,
    ) -> ManagedThreadPool:
        """Create a managed thread pool.

        Args:
            name: Unique name for the thread pool
            max_workers: Maximum number of worker threads

        Returns:
            ManagedThreadPool: The created thread pool

        Raises:
            ValueError: If a thread pool with this name already exists
        """
        with self._lock:
            if name in self._thread_pools:
                raise ValueError(f"Thread pool '{name}' already exists")

            pool = ManagedThreadPool(name=name, max_workers=max_workers)
            self._thread_pools[name] = pool
            logger.info(f"Created thread pool: {name}")
            return pool

    def get_thread_pool(self, name: str) -> Optional[ManagedThreadPool]:
        """Get a thread pool by name.

        Args:
            name: Name of the thread pool

        Returns:
            Optional[ManagedThreadPool]: The thread pool or None if not found
        """
        return self._thread_pools.get(name)

    def track_thread(self, thread: threading.Thread, component_name: str) -> None:
        """Track a thread for monitoring.

        Args:
            thread: The thread to track
            component_name: Name of the component using this thread
        """
        with self._lock:
            info = ThreadInfo.from_thread(thread, component_name)
            self._threads[thread.name] = info
            logger.debug(f"Tracking thread: {thread.name} for component {component_name}")

    def initialize_all(self) -> None:
        """Initialize all registered components."""
        logger.info("Initializing all components...")

        with self._lock:
            for name, component in self._components.items():
                try:
                    logger.info(f"Initializing component: {name}")
                    component.initialize()
                except Exception as e:
                    logger.error(f"Failed to initialize component '{name}': {e}", exc_info=True)
                    raise

    def start_all(self) -> None:
        """Start all registered components and thread pools."""
        if self._running:
            raise RuntimeError("ThreadManager is already running")

        logger.info("Starting all components...")

        with self._lock:
            # Start thread pools first
            for name, pool in self._thread_pools.items():
                try:
                    logger.info(f"Starting thread pool: {name}")
                    pool.start()
                except Exception as e:
                    logger.error(f"Failed to start thread pool '{name}': {e}", exc_info=True)
                    # Stop any already started pools
                    self._stop_pools()
                    raise

            # Then start components
            for name, component in self._components.items():
                try:
                    logger.info(f"Starting component: {name}")
                    component.start()
                except Exception as e:
                    logger.error(f"Failed to start component '{name}': {e}", exc_info=True)
                    # Stop any already started components and pools
                    self._stop_components()
                    self._stop_pools()
                    raise

            self._running = True

        logger.info("All components started successfully")

    def stop_all(self, timeout: float = 5.0) -> None:
        """Stop all components and thread pools gracefully.

        Args:
            timeout: Maximum time to wait for each component to stop
        """
        if not self._running:
            return

        logger.info("Stopping all components...")

        with self._lock:
            self._running = False
            self._stop_components()
            self._stop_pools()

        logger.info("All components stopped")

    def _stop_components(self) -> None:
        """Stop all components (internal method)."""
        for name, component in list(self._components.items()):
            try:
                if component.is_running():
                    logger.info(f"Stopping component: {name}")
                    component.stop()
            except Exception as e:
                logger.error(f"Error stopping component '{name}': {e}", exc_info=True)

    def _stop_pools(self) -> None:
        """Stop all thread pools (internal method)."""
        for name, pool in list(self._thread_pools.items()):
            try:
                if pool.is_running():
                    logger.info(f"Stopping thread pool: {name}")
                    pool.stop(wait=True, timeout=5.0)
            except Exception as e:
                logger.error(f"Error stopping thread pool '{name}': {e}", exc_info=True)

    def is_running(self) -> bool:
        """Check if the thread manager is running.

        Returns:
            bool: True if running
        """
        return self._running

    def get_status(self) -> dict[str, Any]:
        """Get status of all components and threads.

        Returns:
            dict: Status information
        """
        with self._lock:
            return {
                "running": self._running,
                "components": {name: comp.is_running() for name, comp in self._components.items()},
                "thread_pools": {
                    name: pool.get_stats() for name, pool in self._thread_pools.items()
                },
                "tracked_threads": {
                    name: {
                        "component": info.component_name,
                        "alive": info.is_alive,
                        "thread_id": info.thread_id,
                    }
                    for name, info in self._threads.items()
                },
            }
