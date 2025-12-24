"""Transcription Service entry point.

Run with: python -m services.transcription
"""

from services.transcription.service import TranscriptionService, TranscriptionServiceConfig


def main() -> None:
    """Run the transcription service."""
    config = TranscriptionServiceConfig.from_env()
    service = TranscriptionService(config)
    service.run_forever()


if __name__ == "__main__":
    main()
