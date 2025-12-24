"""Memory Service entry point.

Run with: python -m services.memory
"""

from services.memory.service import MemoryService, MemoryServiceConfig


def main() -> None:
    """Run the memory service."""
    config = MemoryServiceConfig.from_env()
    service = MemoryService(config)
    service.run_forever()


if __name__ == "__main__":
    main()
