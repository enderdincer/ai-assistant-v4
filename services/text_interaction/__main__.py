"""Text Interaction Service entry point.

Run with: python -m services.text_interaction
"""

from services.text_interaction.service import TextInteractionService, TextInteractionServiceConfig


def main() -> None:
    """Run the text interaction service."""
    config = TextInteractionServiceConfig.from_env()
    service = TextInteractionService(config)
    service.run_forever()


if __name__ == "__main__":
    main()
