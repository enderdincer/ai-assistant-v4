"""Thread pool management using ThreadPoolExecutor."""

import threading
from concurrent.futures import ThreadPoolExecutor, Future
from typing import Callable, Optional, Any
from ai_assistant.shared.logging import get_logger

logger = get_logger(__name__)


class ManagedThreadPool:
    """Managed thread pool with lifecycle control.

    This wraps ThreadPoolExecutor with additional monitoring and
    lifecycle management capabilities.
    """

    def __init__(
        self,
        name: str,
        max_workers: Optional[int] = None,
        thread_name_prefix: str = "",
    ) -> None:
        """Initialize the thread pool.

        Args:
            name: Name for this thread pool
            max_workers: Maximum number of worker threads (None = CPU count * 5)
            thread_name_prefix: Prefix for worker thread names
        """
        self.name = name
        self._max_workers = max_workers
        self._thread_name_prefix = thread_name_prefix or f"{name}-worker"
        self._executor: Optional[ThreadPoolExecutor] = None
        self._lock = threading.Lock()
        self._running = False
        self._submitted_tasks = 0
        self._completed_tasks = 0

    def start(self) -> None:
        """Start the thread pool."""
        with self._lock:
            if self._running:
                raise RuntimeError(f"Thread pool '{self.name}' is already running")

            self._executor = ThreadPoolExecutor(
                max_workers=self._max_workers,
                thread_name_prefix=self._thread_name_prefix,
            )
            self._running = True
            logger.info(
                f"Thread pool '{self.name}' started with "
                f"{self._max_workers or 'default'} max workers"
            )

    def stop(self, wait: bool = True, timeout: Optional[float] = None) -> None:
        """Stop the thread pool.

        Args:
            wait: Whether to wait for running tasks to complete
            timeout: Maximum time to wait in seconds
        """
        with self._lock:
            if not self._running:
                return

            self._running = False

            if self._executor:
                self._executor.shutdown(wait=wait, cancel_futures=not wait)
                self._executor = None

            logger.info(
                f"Thread pool '{self.name}' stopped "
                f"(submitted={self._submitted_tasks}, completed={self._completed_tasks})"
            )

    def submit(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Future[Any]:
        """Submit a task to the thread pool.

        Args:
            fn: The function to execute
            *args: Positional arguments for the function
            **kwargs: Keyword arguments for the function

        Returns:
            Future: Future representing the task

        Raises:
            RuntimeError: If thread pool is not running
        """
        with self._lock:
            if not self._running or self._executor is None:
                raise RuntimeError(f"Thread pool '{self.name}' is not running")

            future = self._executor.submit(self._wrap_task(fn), *args, **kwargs)
            self._submitted_tasks += 1
            return future

    def _wrap_task(self, fn: Callable[..., Any]) -> Callable[..., Any]:
        """Wrap a task function with error handling and completion tracking.

        Args:
            fn: The function to wrap

        Returns:
            Callable: Wrapped function
        """

        def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                result = fn(*args, **kwargs)
                with self._lock:
                    self._completed_tasks += 1
                return result
            except Exception as e:
                logger.error(f"Error in thread pool task: {e}", exc_info=True)
                with self._lock:
                    self._completed_tasks += 1
                raise

        return wrapper

    def is_running(self) -> bool:
        """Check if the thread pool is running.

        Returns:
            bool: True if running
        """
        return self._running

    def get_stats(self) -> dict[str, Any]:
        """Get thread pool statistics.

        Returns:
            dict: Statistics about the thread pool
        """
        with self._lock:
            return {
                "name": self.name,
                "running": self._running,
                "max_workers": self._max_workers,
                "submitted_tasks": self._submitted_tasks,
                "completed_tasks": self._completed_tasks,
                "pending_tasks": self._submitted_tasks - self._completed_tasks,
            }
