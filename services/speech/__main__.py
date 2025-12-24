"""Speech Service entry point.

Run with: python -m services.speech
"""

from services.speech.service import SpeechService, SpeechServiceConfig


def main() -> None:
    """Run the speech service."""
    config = SpeechServiceConfig.from_env()
    service = SpeechService(config)
    service.run_forever()


if __name__ == "__main__":
    main()
