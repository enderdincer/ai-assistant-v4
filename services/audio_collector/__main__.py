"""Audio Collector Service entry point.

Run with: python -m services.audio_collector
"""

from services.audio_collector.service import AudioCollectorService, AudioCollectorConfig


def main() -> None:
    """Run the audio collector service."""
    config = AudioCollectorConfig.from_env()
    service = AudioCollectorService(config)
    service.run_forever()


if __name__ == "__main__":
    main()
