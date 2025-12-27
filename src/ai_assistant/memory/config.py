import os
from dataclasses import dataclass, field


@dataclass
class MemoryConfig:
    """Configuration for the memory system.

    The memory system uses:
    - PostgreSQL for structured data (messages, facts, events)
    - Qdrant for vector similarity search
    - Local sentence-transformers for embeddings (no external API needed)
    """

    # PostgreSQL settings
    postgres_host: str = field(default_factory=lambda: os.getenv("POSTGRES_HOST", "localhost"))
    postgres_port: int = field(default_factory=lambda: int(os.getenv("POSTGRES_PORT", "5432")))
    postgres_user: str = field(default_factory=lambda: os.getenv("POSTGRES_USER", "ai_assistant"))
    postgres_password: str = field(
        default_factory=lambda: os.getenv("POSTGRES_PASSWORD", "ai_assistant_secret")
    )
    postgres_db: str = field(default_factory=lambda: os.getenv("POSTGRES_DB", "ai_assistant"))
    postgres_pool_min: int = field(default_factory=lambda: int(os.getenv("POSTGRES_POOL_MIN", "2")))
    postgres_pool_max: int = field(
        default_factory=lambda: int(os.getenv("POSTGRES_POOL_MAX", "10"))
    )

    # Qdrant settings
    qdrant_host: str = field(default_factory=lambda: os.getenv("QDRANT_HOST", "localhost"))
    qdrant_port: int = field(default_factory=lambda: int(os.getenv("QDRANT_HTTP_PORT", "6333")))

    # Embedding settings (local sentence-transformers model)
    # Default: all-mpnet-base-v2 (768 dimensions, ~420MB, best quality)
    embedding_model: str = field(
        default_factory=lambda: os.getenv("EMBEDDING_MODEL", "all-mpnet-base-v2")
    )
    embedding_dimension: int = field(
        default_factory=lambda: int(os.getenv("EMBEDDING_DIMENSION", "768"))
    )

    @classmethod
    def from_env(cls) -> "MemoryConfig":
        return cls()

    @property
    def postgres_dsn(self) -> str:
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )
