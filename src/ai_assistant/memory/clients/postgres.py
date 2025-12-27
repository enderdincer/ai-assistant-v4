from contextlib import contextmanager
from typing import Any, Generator, Optional

from ai_assistant.memory.config import MemoryConfig
from ai_assistant.memory.exceptions import MemoryConnectionError, MemoryQueryError

try:
    import psycopg
    from psycopg_pool import ConnectionPool
except ImportError:
    psycopg = None  # type: ignore
    ConnectionPool = None  # type: ignore


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS facts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    category VARCHAR(50) NOT NULL,
    subject VARCHAR(255) NOT NULL,
    attribute VARCHAR(255) NOT NULL,
    value TEXT NOT NULL,
    confidence FLOAT DEFAULT 1.0,
    source VARCHAR(50) NOT NULL,
    is_immutable BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(category, subject, attribute)
);

CREATE TABLE IF NOT EXISTS event_log (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    event_type VARCHAR(50) NOT NULL,
    session_id UUID NOT NULL,
    data JSONB NOT NULL,
    metadata JSONB DEFAULT '{}',
    timestamp TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_event_log_session ON event_log(session_id);
CREATE INDEX IF NOT EXISTS idx_event_log_timestamp ON event_log(timestamp);
CREATE INDEX IF NOT EXISTS idx_event_log_type ON event_log(event_type);

CREATE TABLE IF NOT EXISTS messages (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID NOT NULL,
    role VARCHAR(20) NOT NULL,
    content TEXT NOT NULL,
    metadata JSONB DEFAULT '{}',
    timestamp TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id);
CREATE INDEX IF NOT EXISTS idx_messages_timestamp ON messages(timestamp);

CREATE TABLE IF NOT EXISTS conversation_summaries (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID NOT NULL,
    summary TEXT NOT NULL,
    message_count INT NOT NULL,
    first_message_id UUID NOT NULL,
    last_message_id UUID NOT NULL,
    first_timestamp TIMESTAMPTZ NOT NULL,
    last_timestamp TIMESTAMPTZ NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_summaries_session ON conversation_summaries(session_id);
CREATE INDEX IF NOT EXISTS idx_summaries_timestamp ON conversation_summaries(created_at);
"""


class PostgresClient:
    def __init__(self, config: Optional[MemoryConfig] = None):
        if psycopg is None:
            raise ImportError(
                "psycopg is not installed. Install with: pip install 'psycopg[binary,pool]'"
            )
        self._config = config or MemoryConfig.from_env()
        self._pool: Optional[ConnectionPool] = None

    def initialize(self) -> None:
        try:
            self._pool = ConnectionPool(
                self._config.postgres_dsn,
                min_size=self._config.postgres_pool_min,
                max_size=self._config.postgres_pool_max,
            )
            self._create_schema()
        except psycopg.Error as e:
            raise MemoryConnectionError(f"Failed to connect to PostgreSQL: {e}") from e

    def _create_schema(self) -> None:
        with self.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(SCHEMA_SQL)
            conn.commit()

    @contextmanager
    def connection(self) -> Generator[Any, None, None]:
        if self._pool is None:
            raise MemoryConnectionError("PostgresClient not initialized")
        with self._pool.connection() as conn:
            yield conn

    def execute(
        self,
        query: str,
        params: Optional[tuple[Any, ...]] = None,
    ) -> None:
        try:
            with self.connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(query, params)
                conn.commit()
        except psycopg.Error as e:
            raise MemoryQueryError(f"Query execution failed: {e}") from e

    def execute_returning(
        self,
        query: str,
        params: Optional[tuple[Any, ...]] = None,
    ) -> Any:
        try:
            with self.connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(query, params)
                    result = cur.fetchone()
                conn.commit()
                return result
        except psycopg.Error as e:
            raise MemoryQueryError(f"Query execution failed: {e}") from e

    def fetch_one(
        self,
        query: str,
        params: Optional[tuple[Any, ...]] = None,
    ) -> Optional[tuple[Any, ...]]:
        try:
            with self.connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(query, params)
                    result: Optional[tuple[Any, ...]] = cur.fetchone()
                    return result
        except psycopg.Error as e:
            raise MemoryQueryError(f"Query execution failed: {e}") from e

    def fetch_all(
        self,
        query: str,
        params: Optional[tuple[Any, ...]] = None,
    ) -> list[tuple[Any, ...]]:
        try:
            with self.connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(query, params)
                    result: list[tuple[Any, ...]] = cur.fetchall()
                    return result
        except psycopg.Error as e:
            raise MemoryQueryError(f"Query execution failed: {e}") from e

    def close(self) -> None:
        if self._pool is not None:
            self._pool.close()
            self._pool = None

    def __enter__(self) -> "PostgresClient":
        self.initialize()
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
