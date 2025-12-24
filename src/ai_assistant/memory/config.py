import os
from dataclasses import dataclass, field


@dataclass
class MemoryConfig:
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

    qdrant_host: str = field(default_factory=lambda: os.getenv("QDRANT_HOST", "localhost"))
    qdrant_port: int = field(default_factory=lambda: int(os.getenv("QDRANT_HTTP_PORT", "6333")))

    embedding_model: str = field(
        default_factory=lambda: os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
    )
    embedding_dimension: int = field(
        default_factory=lambda: int(os.getenv("EMBEDDING_DIMENSION", "768"))
    )

    ollama_host: str = field(
        default_factory=lambda: os.getenv("OLLAMA_HOST", "http://localhost:11434")
    )

    compaction_threshold: int = field(
        default_factory=lambda: int(os.getenv("MEMORY_COMPACTION_THRESHOLD", "50"))
    )
    compaction_keep_recent: int = field(
        default_factory=lambda: int(os.getenv("MEMORY_COMPACTION_KEEP_RECENT", "10"))
    )
    compaction_model: str = field(default_factory=lambda: os.getenv("MEMORY_COMPACTION_MODEL", ""))

    @classmethod
    def from_env(cls) -> "MemoryConfig":
        return cls()

    @property
    def postgres_dsn(self) -> str:
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )
