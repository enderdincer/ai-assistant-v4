"""Memory Manager - Core interface for the memory system.

Provides a unified interface for:
- Storing and retrieving conversation messages
- Managing facts (learned information)
- Semantic search across conversations and facts
- Session management

The memory system uses local embeddings (sentence-transformers) and does NOT
perform any summarization or compaction. Summarization should be handled
by external services if needed.
"""

from typing import Any, Optional
from uuid import uuid4

from ai_assistant.memory.config import MemoryConfig
from ai_assistant.memory.clients.embeddings import EmbeddingService
from ai_assistant.memory.clients.postgres import PostgresClient
from ai_assistant.memory.clients.qdrant import QdrantClient
from ai_assistant.memory.models import (
    ConversationSummary,
    Fact,
    FactCategory,
    MemoryContext,
    Message,
)
from ai_assistant.memory.stores.conversation_store import ConversationStore
from ai_assistant.memory.stores.event_log import EventLog
from ai_assistant.memory.stores.fact_store import FactStore


class MemoryManager:
    """Main interface for the memory system.

    Provides methods for:
    - Logging and retrieving conversations
    - Storing and searching facts
    - Getting context for queries (semantic search)
    - Session management

    Example:
        ```python
        config = MemoryConfig()
        manager = MemoryManager(config)
        manager.initialize()

        # Log a conversation
        manager.log_conversation(
            session_id="session-123",
            role="user",
            content="What's the weather like?"
        )

        # Search for relevant context
        context = manager.get_context_for_query("weather forecast")

        manager.close()
        ```
    """

    def __init__(self, config: Optional[MemoryConfig] = None):
        self._config = config or MemoryConfig.from_env()
        self._postgres: Optional[PostgresClient] = None
        self._qdrant: Optional[QdrantClient] = None
        self._embeddings: Optional[EmbeddingService] = None
        self._fact_store: Optional[FactStore] = None
        self._event_log: Optional[EventLog] = None
        self._conversation_store: Optional[ConversationStore] = None
        self._initialized = False

    def initialize(self) -> None:
        """Initialize all database connections and the embedding service."""
        if self._initialized:
            return

        self._postgres = PostgresClient(self._config)
        self._postgres.initialize()

        self._qdrant = QdrantClient(self._config)
        self._qdrant.initialize()

        self._embeddings = EmbeddingService(self._config)
        self._embeddings.initialize()

        self._fact_store = FactStore(self._postgres, self._qdrant, self._embeddings)
        self._event_log = EventLog(self._postgres)
        self._conversation_store = ConversationStore(self._postgres, self._qdrant, self._embeddings)

        self._initialized = True

    def close(self) -> None:
        """Close all connections and clean up resources."""
        if self._embeddings:
            self._embeddings.close()
            self._embeddings = None

        if self._qdrant:
            self._qdrant.close()
            self._qdrant = None

        if self._postgres:
            self._postgres.close()
            self._postgres = None

        self._fact_store = None
        self._event_log = None
        self._conversation_store = None
        self._initialized = False

    @property
    def facts(self) -> FactStore:
        """Get the fact store for direct fact operations."""
        if self._fact_store is None:
            raise RuntimeError("MemoryManager not initialized")
        return self._fact_store

    @property
    def events(self) -> EventLog:
        """Get the event log for direct event operations."""
        if self._event_log is None:
            raise RuntimeError("MemoryManager not initialized")
        return self._event_log

    @property
    def conversations(self) -> ConversationStore:
        """Get the conversation store for direct conversation operations."""
        if self._conversation_store is None:
            raise RuntimeError("MemoryManager not initialized")
        return self._conversation_store

    def get_context_for_query(
        self,
        query: str,
        session_id: Optional[str] = None,
        max_facts: int = 5,
        max_similar: int = 3,
    ) -> MemoryContext:
        """Get relevant context for a query.

        Searches for:
        - Relevant facts (semantic search)
        - Recent messages in the current session
        - Similar conversations from other sessions

        Args:
            query: The query text to find context for
            session_id: Optional session ID to get recent messages from
            max_facts: Maximum number of facts to return
            max_similar: Maximum number of similar conversations to return

        Returns:
            MemoryContext with relevant facts, messages, and similar conversations
        """
        relevant_facts = self.facts.search_facts(query, limit=max_facts)

        recent_messages: list[Message] = []
        if session_id:
            # Include any existing summaries
            summaries = self.conversations.get_session_summaries(session_id)
            for summary in summaries:
                recent_messages.append(
                    Message(
                        id=summary.id,
                        session_id=session_id,
                        role="summary",
                        content=f"[Previous conversation summary: {summary.summary}]",
                        timestamp=summary.created_at,
                    )
                )

            # Get recent messages from the session
            session_messages = self.conversations.get_session(session_id)
            recent_messages.extend(session_messages[-10:])

        similar_conversations = self.conversations.search_similar(
            query,
            limit=max_similar,
            exclude_session=session_id,
        )

        return MemoryContext(
            relevant_facts=relevant_facts,
            recent_messages=recent_messages,
            similar_conversations=similar_conversations,
        )

    def remember_fact(
        self,
        subject: str,
        attribute: str,
        value: str,
        confidence: float = 0.8,
        source: str = "conversation",
    ) -> str:
        """Store a learned fact.

        Args:
            subject: The subject of the fact (e.g., "user", "project")
            attribute: The attribute name (e.g., "name", "preference")
            value: The attribute value
            confidence: Confidence score (0-1)
            source: Source of the fact

        Returns:
            The fact ID
        """
        fact = Fact.create_learned_fact(
            subject=subject,
            attribute=attribute,
            value=value,
            confidence=confidence,
            source=source,
        )
        return self.facts.add_fact(fact)

    def log_conversation(
        self,
        session_id: str,
        role: str,
        content: str,
        metadata: Optional[dict[str, Any]] = None,
    ) -> str:
        """Log a conversation message.

        Args:
            session_id: Session ID for the conversation
            role: Message role (user, assistant, system)
            content: Message content
            metadata: Optional metadata dict

        Returns:
            The message ID
        """
        message_id = self.conversations.store_message(
            session_id=session_id,
            role=role,
            content=content,
            metadata=metadata,
        )

        self.events.log_event(
            event_type="conversation",
            session_id=session_id,
            data={
                "message_id": message_id,
                "role": role,
                "content_preview": content[:200],
            },
            metadata=metadata,
        )

        return message_id

    def add_system_fact(
        self,
        subject: str,
        attribute: str,
        value: str,
    ) -> str:
        """Add an immutable system fact.

        System facts cannot be modified or deleted once created.

        Args:
            subject: The subject of the fact
            attribute: The attribute name
            value: The attribute value

        Returns:
            The fact ID
        """
        fact = Fact.create_system_fact(
            subject=subject,
            attribute=attribute,
            value=value,
        )
        return self.facts.add_fact(fact)

    def create_session(self) -> str:
        """Create a new session ID.

        Returns:
            A new UUID string for the session
        """
        return str(uuid4())

    def store_summary(
        self,
        session_id: str,
        summary: str,
        message_ids: list[str],
        delete_summarized: bool = False,
    ) -> str:
        """Store a conversation summary.

        This allows external services to provide summaries for conversations.
        The memory service does NOT generate summaries itself.

        Args:
            session_id: Session ID the summary belongs to
            summary: The summary text
            message_ids: IDs of messages that were summarized
            delete_summarized: Whether to delete the summarized messages

        Returns:
            The summary ID
        """
        # Get the messages to extract timestamps
        messages_or_none = [self.conversations.get_message(mid) for mid in message_ids]
        messages = [m for m in messages_or_none if m is not None]

        if not messages:
            raise ValueError("No valid messages found for the given IDs")

        # Sort by timestamp
        messages.sort(key=lambda m: m.timestamp)

        first_msg = messages[0]
        last_msg = messages[-1]

        summary_obj = ConversationSummary(
            session_id=session_id,
            summary=summary,
            message_count=len(messages),
            first_message_id=first_msg.id,
            last_message_id=last_msg.id,
            first_timestamp=first_msg.timestamp,
            last_timestamp=last_msg.timestamp,
        )

        summary_id = self.conversations.store_summary(summary_obj)

        # Optionally delete the summarized messages
        if delete_summarized:
            self.conversations.delete_messages(message_ids)

            self.events.log_event(
                event_type="summary_with_deletion",
                session_id=session_id,
                data={
                    "summary_id": summary_id,
                    "messages_summarized": len(messages),
                    "messages_deleted": len(message_ids),
                },
            )
        else:
            self.events.log_event(
                event_type="summary",
                session_id=session_id,
                data={
                    "summary_id": summary_id,
                    "messages_summarized": len(messages),
                },
            )

        return summary_id

    # =========================================================================
    # Session Management
    # =========================================================================

    def list_sessions(
        self,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[dict[str, Any]], int]:
        """List all sessions with pagination.

        Args:
            limit: Maximum number of sessions to return
            offset: Offset for pagination

        Returns:
            Tuple of (sessions list, total count)
        """
        sessions = self.conversations.list_sessions(limit=limit, offset=offset)
        total = self.conversations.get_total_session_count()
        return sessions, total

    def get_session_info(self, session_id: str) -> Optional[dict[str, Any]]:
        """Get information about a specific session.

        Args:
            session_id: Session ID to look up

        Returns:
            Session info dict or None if not found
        """
        summary = self.conversations.get_session_summary(session_id)
        if summary.get("message_count", 0) == 0:
            # Check if there are any messages at all
            messages = self.conversations.get_session(session_id)
            if not messages:
                return None
        return summary

    def delete_session(self, session_id: str) -> int:
        """Delete a session and all its messages.

        Args:
            session_id: Session ID to delete

        Returns:
            Number of messages deleted
        """
        return self.conversations.delete_session(session_id)

    def clear_session(self, session_id: str) -> int:
        """Clear all messages in a session.

        Args:
            session_id: Session ID to clear

        Returns:
            Number of messages cleared
        """
        return self.conversations.clear_session_messages(session_id)

    def get_session_messages(
        self,
        session_id: str,
        limit: int = 100,
        offset: int = 0,
    ) -> tuple[list[Message], int]:
        """Get messages for a session with pagination.

        Args:
            session_id: Session ID
            limit: Maximum number of messages to return
            offset: Offset for pagination

        Returns:
            Tuple of (messages list, total count)
        """
        all_messages = self.conversations.get_session(session_id)
        total = len(all_messages)

        # Apply pagination
        paginated = all_messages[offset : offset + limit]
        return paginated, total

    def __enter__(self) -> "MemoryManager":
        self.initialize()
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
