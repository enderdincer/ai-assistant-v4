"""Heartbeat Service entry point.

Run with: python -m services.heartbeat
"""

from services.heartbeat.service import HeartbeatService, HeartbeatServiceConfig


def main() -> None:
    """Run the heartbeat service."""
    config = HeartbeatServiceConfig.from_env()
    service = HeartbeatService(config)
    service.run_forever()


if __name__ == "__main__":
    main()
