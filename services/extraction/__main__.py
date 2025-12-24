"""Extraction Service entry point.

Run with: python -m services.extraction
"""

from services.extraction.service import ExtractionService, ExtractionServiceConfig


def main() -> None:
    """Run the extraction service."""
    config = ExtractionServiceConfig.from_env()
    service = ExtractionService(config)
    service.run_forever()


if __name__ == "__main__":
    main()
