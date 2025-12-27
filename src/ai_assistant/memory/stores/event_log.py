import json
from datetime import datetime
from typing import Any, Optional

from ai_assistant.memory.clients.postgres import PostgresClient
from ai_assistant.memory.models import EventLogEntry


class EventLog:
    def __init__(self, postgres: PostgresClient):
        self._postgres = postgres

    def log_event(
        self,
        event_type: str,
        session_id: str,
        data: dict[str, Any],
        metadata: Optional[dict[str, Any]] = None,
    ) -> str:
        entry = EventLogEntry(
            event_type=event_type,
            session_id=session_id,
            data=data,
            metadata=metadata or {},
        )

        query = """
            INSERT INTO event_log (id, event_type, session_id, data, metadata, timestamp)
            VALUES (%s, %s, %s, %s, %s, %s)
            RETURNING id
        """
        result = self._postgres.execute_returning(
            query,
            (
                entry.id,
                entry.event_type,
                entry.session_id,
                json.dumps(entry.data),
                json.dumps(entry.metadata),
                entry.timestamp,
            ),
        )
        return str(result[0]) if result else entry.id

    def get_events(
        self,
        session_id: Optional[str] = None,
        event_type: Optional[str] = None,
        since: Optional[datetime] = None,
        limit: int = 100,
    ) -> list[EventLogEntry]:
        conditions: list[str] = []
        params: list[Any] = []

        if session_id:
            conditions.append("session_id = %s")
            params.append(session_id)
        if event_type:
            conditions.append("event_type = %s")
            params.append(event_type)
        if since:
            conditions.append("timestamp >= %s")
            params.append(since)

        where_clause = " AND ".join(conditions) if conditions else "TRUE"
        query = f"""
            SELECT id, event_type, session_id, data, metadata, timestamp
            FROM event_log
            WHERE {where_clause}
            ORDER BY timestamp DESC
            LIMIT %s
        """
        params.append(limit)

        rows = self._postgres.fetch_all(query, tuple(params))
        return [self._row_to_entry(row) for row in rows]

    def get_recent_events(self, limit: int = 50) -> list[EventLogEntry]:
        return self.get_events(limit=limit)

    def get_session_events(self, session_id: str) -> list[EventLogEntry]:
        return self.get_events(session_id=session_id, limit=1000)

    def _row_to_entry(self, row: tuple[Any, ...]) -> EventLogEntry:
        return EventLogEntry(
            id=str(row[0]),
            event_type=row[1],
            session_id=str(row[2]),
            data=row[3] if isinstance(row[3], dict) else json.loads(row[3]),
            metadata=row[4] if isinstance(row[4], dict) else json.loads(row[4]),
            timestamp=row[5],
        )
