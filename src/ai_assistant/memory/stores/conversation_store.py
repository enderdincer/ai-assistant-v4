import json
from typing import Any, Optional

from ai_assistant.memory.clients.embeddings import EmbeddingService
from ai_assistant.memory.clients.postgres import PostgresClient
from ai_assistant.memory.clients.qdrant import QdrantClient
from ai_assistant.memory.models import ConversationSummary, Message


class ConversationStore:
    def __init__(
        self,
        postgres: PostgresClient,
        qdrant: QdrantClient,
        embeddings: EmbeddingService,
    ):
        self._postgres = postgres
        self._qdrant = qdrant
        self._embeddings = embeddings

    def store_message(
        self,
        session_id: str,
        role: str,
        content: str,
        metadata: Optional[dict[str, Any]] = None,
    ) -> str:
        message = Message(
            session_id=session_id,
            role=role,
            content=content,
            metadata=metadata or {},
        )

        query = """
            INSERT INTO messages (id, session_id, role, content, metadata, timestamp)
            VALUES (%s, %s, %s, %s, %s, %s)
            RETURNING id
        """
        result = self._postgres.execute_returning(
            query,
            (
                message.id,
                message.session_id,
                message.role,
                message.content,
                json.dumps(message.metadata),
                message.timestamp,
            ),
        )
        message_id = str(result[0]) if result else message.id

        embedding = self._embeddings.embed(content)
        self._qdrant.upsert(
            collection=QdrantClient.COLLECTION_CONVERSATIONS,
            point_id=message_id,
            vector=embedding,
            payload={
                "message_id": message_id,
                "session_id": session_id,
                "role": role,
                "content": content[:500],
            },
        )

        return message_id

    def get_session(self, session_id: str) -> list[Message]:
        query = """
            SELECT id, session_id, role, content, metadata, timestamp
            FROM messages
            WHERE session_id = %s
            ORDER BY timestamp ASC
        """
        rows = self._postgres.fetch_all(query, (session_id,))
        return [self._row_to_message(row) for row in rows]

    def search_similar(
        self,
        query: str,
        limit: int = 5,
        exclude_session: Optional[str] = None,
    ) -> list[Message]:
        embedding = self._embeddings.embed(query)

        filter_conditions = None
        if exclude_session:
            pass

        results = self._qdrant.search(
            collection=QdrantClient.COLLECTION_CONVERSATIONS,
            query_vector=embedding,
            limit=limit * 2,
            filter_conditions=filter_conditions,
        )

        messages: list[Message] = []
        seen_sessions: set[str] = set()
        if exclude_session:
            seen_sessions.add(exclude_session)

        for result in results:
            message_id = result["payload"].get("message_id")
            session_id = result["payload"].get("session_id")

            if session_id in seen_sessions:
                continue

            if message_id:
                message = self.get_message(message_id)
                if message:
                    messages.append(message)
                    seen_sessions.add(session_id)

            if len(messages) >= limit:
                break

        return messages

    def get_message(self, message_id: str) -> Optional[Message]:
        query = """
            SELECT id, session_id, role, content, metadata, timestamp
            FROM messages WHERE id = %s
        """
        row = self._postgres.fetch_one(query, (message_id,))
        if row is None:
            return None
        return self._row_to_message(row)

    def get_recent_sessions(self, limit: int = 10) -> list[str]:
        query = """
            SELECT DISTINCT session_id
            FROM messages
            ORDER BY MAX(timestamp) DESC
            LIMIT %s
        """
        rows = self._postgres.fetch_all(
            """
            SELECT session_id FROM (
                SELECT session_id, MAX(timestamp) as last_msg
                FROM messages
                GROUP BY session_id
                ORDER BY last_msg DESC
                LIMIT %s
            ) sub
            """,
            (limit,),
        )
        return [str(row[0]) for row in rows]

    def get_session_summary(self, session_id: str) -> dict[str, Any]:
        messages = self.get_session(session_id)
        if not messages:
            return {"session_id": session_id, "message_count": 0}

        return {
            "session_id": session_id,
            "message_count": len(messages),
            "first_message": messages[0].timestamp.isoformat(),
            "last_message": messages[-1].timestamp.isoformat(),
            "preview": messages[0].content[:100] if messages else "",
        }

    def _row_to_message(self, row: tuple[Any, ...]) -> Message:
        return Message(
            id=str(row[0]),
            session_id=str(row[1]),
            role=row[2],
            content=row[3],
            metadata=row[4] if isinstance(row[4], dict) else json.loads(row[4]),
            timestamp=row[5],
        )

    def get_session_message_count(self, session_id: str) -> int:
        query = "SELECT COUNT(*) FROM messages WHERE session_id = %s"
        result = self._postgres.fetch_one(query, (session_id,))
        return int(result[0]) if result else 0

    def get_compactable_messages(
        self,
        session_id: str,
        keep_recent: int = 10,
    ) -> list[Message]:
        query = """
            SELECT id, session_id, role, content, metadata, timestamp
            FROM messages
            WHERE session_id = %s
            ORDER BY timestamp ASC
            LIMIT (SELECT GREATEST(0, COUNT(*) - %s) FROM messages WHERE session_id = %s)
        """
        rows = self._postgres.fetch_all(query, (session_id, keep_recent, session_id))
        return [self._row_to_message(row) for row in rows]

    def store_summary(self, summary: ConversationSummary) -> str:
        query = """
            INSERT INTO conversation_summaries 
            (id, session_id, summary, message_count, first_message_id, last_message_id, first_timestamp, last_timestamp, created_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id
        """
        result = self._postgres.execute_returning(
            query,
            (
                summary.id,
                summary.session_id,
                summary.summary,
                summary.message_count,
                summary.first_message_id,
                summary.last_message_id,
                summary.first_timestamp,
                summary.last_timestamp,
                summary.created_at,
            ),
        )
        summary_id = str(result[0]) if result else summary.id

        embedding = self._embeddings.embed(summary.summary)
        self._qdrant.upsert(
            collection=QdrantClient.COLLECTION_CONVERSATIONS,
            point_id=f"summary_{summary_id}",
            vector=embedding,
            payload={
                "summary_id": summary_id,
                "session_id": summary.session_id,
                "role": "summary",
                "content": summary.summary[:500],
                "is_summary": True,
            },
        )

        return summary_id

    def delete_messages(self, message_ids: list[str]) -> int:
        if not message_ids:
            return 0

        placeholders = ",".join(["%s"] * len(message_ids))
        query = f"DELETE FROM messages WHERE id IN ({placeholders}) RETURNING id"

        deleted_count = 0
        for msg_id in message_ids:
            result = self._postgres.execute_returning(
                "DELETE FROM messages WHERE id = %s RETURNING id",
                (msg_id,),
            )
            if result:
                deleted_count += 1
                try:
                    self._qdrant.delete(QdrantClient.COLLECTION_CONVERSATIONS, msg_id)
                except Exception:
                    pass

        return deleted_count

    def get_session_summaries(self, session_id: str) -> list[ConversationSummary]:
        query = """
            SELECT id, session_id, summary, message_count, first_message_id, last_message_id, 
                   first_timestamp, last_timestamp, created_at
            FROM conversation_summaries
            WHERE session_id = %s
            ORDER BY created_at ASC
        """
        rows = self._postgres.fetch_all(query, (session_id,))
        return [self._row_to_summary(row) for row in rows]

    def _row_to_summary(self, row: tuple[Any, ...]) -> ConversationSummary:
        return ConversationSummary(
            id=str(row[0]),
            session_id=str(row[1]),
            summary=row[2],
            message_count=row[3],
            first_message_id=str(row[4]),
            last_message_id=str(row[5]),
            first_timestamp=row[6],
            last_timestamp=row[7],
            created_at=row[8],
        )

    def delete_session(self, session_id: str) -> int:
        """Delete all messages and summaries for a session.

        Args:
            session_id: Session ID to delete

        Returns:
            Number of messages deleted
        """
        # Get all message IDs for this session
        messages = self.get_session(session_id)
        message_ids = [m.id for m in messages]

        # Delete messages from both PostgreSQL and Qdrant
        deleted_count = self.delete_messages(message_ids)

        # Delete summaries
        self._postgres.execute_returning(
            "DELETE FROM conversation_summaries WHERE session_id = %s RETURNING id",
            (session_id,),
        )

        # Delete summary vectors from Qdrant
        summaries = self.get_session_summaries(session_id)
        for summary in summaries:
            try:
                self._qdrant.delete(QdrantClient.COLLECTION_CONVERSATIONS, f"summary_{summary.id}")
            except Exception:
                pass

        return deleted_count

    def list_sessions(
        self,
        limit: int = 50,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """List all sessions with summary information.

        Args:
            limit: Maximum number of sessions to return
            offset: Offset for pagination

        Returns:
            List of session summary dictionaries
        """
        query = """
            SELECT 
                session_id,
                COUNT(*) as message_count,
                MIN(timestamp) as first_message,
                MAX(timestamp) as last_message
            FROM messages
            GROUP BY session_id
            ORDER BY MAX(timestamp) DESC
            LIMIT %s OFFSET %s
        """
        rows = self._postgres.fetch_all(query, (limit, offset))

        sessions = []
        for row in rows:
            sessions.append(
                {
                    "session_id": str(row[0]),
                    "message_count": int(row[1]),
                    "first_message": row[2].isoformat() if row[2] else None,
                    "last_message": row[3].isoformat() if row[3] else None,
                }
            )

        return sessions

    def get_total_session_count(self) -> int:
        """Get total number of unique sessions.

        Returns:
            Total session count
        """
        query = "SELECT COUNT(DISTINCT session_id) FROM messages"
        result = self._postgres.fetch_one(query, ())
        return int(result[0]) if result else 0

    def clear_session_messages(self, session_id: str) -> int:
        """Clear all messages in a session but keep the session available.

        This differs from delete_session in that it just clears messages
        but the session_id remains valid for new messages.

        Args:
            session_id: Session ID to clear

        Returns:
            Number of messages cleared
        """
        return self.delete_session(session_id)
