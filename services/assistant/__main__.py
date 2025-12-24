"""Assistant Service entry point.

Run with: python -m services.assistant
"""

from services.assistant.service import AssistantService, AssistantServiceConfig


def main() -> None:
    """Run the assistant service."""
    config = AssistantServiceConfig.from_env()
    service = AssistantService(config)
    service.run_forever()


if __name__ == "__main__":
    main()
