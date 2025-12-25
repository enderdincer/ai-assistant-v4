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

try:
    from ai_assistant.shared.ollama import OllamaClient
except ImportError:
    OllamaClient = None  # type: ignore


class MemoryManager:
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
        if self._fact_store is None:
            raise RuntimeError("MemoryManager not initialized")
        return self._fact_store

    @property
    def events(self) -> EventLog:
        if self._event_log is None:
            raise RuntimeError("MemoryManager not initialized")
        return self._event_log

    @property
    def conversations(self) -> ConversationStore:
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
        relevant_facts = self.facts.search_facts(query, limit=max_facts)

        recent_messages: list[Message] = []
        if session_id:
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
        auto_compact: bool = True,
    ) -> str:
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

        if auto_compact:
            self.maybe_compact_session(session_id)

        return message_id

    def add_system_fact(
        self,
        subject: str,
        attribute: str,
        value: str,
    ) -> str:
        fact = Fact.create_system_fact(
            subject=subject,
            attribute=attribute,
            value=value,
        )
        return self.facts.add_fact(fact)

    def create_session(self) -> str:
        return str(uuid4())

    def compact_session(self, session_id: str, force: bool = False) -> Optional[str]:
        message_count = self.conversations.get_session_message_count(session_id)

        if not force and message_count < self._config.compaction_threshold:
            return None

        messages_to_compact = self.conversations.get_compactable_messages(
            session_id=session_id,
            keep_recent=self._config.compaction_keep_recent,
        )

        if len(messages_to_compact) < 5:
            return None

        summary_text = self._generate_summary(messages_to_compact)
        if not summary_text:
            return None

        summary = ConversationSummary(
            session_id=session_id,
            summary=summary_text,
            message_count=len(messages_to_compact),
            first_message_id=messages_to_compact[0].id,
            last_message_id=messages_to_compact[-1].id,
            first_timestamp=messages_to_compact[0].timestamp,
            last_timestamp=messages_to_compact[-1].timestamp,
        )

        summary_id = self.conversations.store_summary(summary)

        message_ids = [m.id for m in messages_to_compact]
        deleted = self.conversations.delete_messages(message_ids)

        self.events.log_event(
            event_type="compaction",
            session_id=session_id,
            data={
                "summary_id": summary_id,
                "messages_compacted": len(messages_to_compact),
                "messages_deleted": deleted,
                "summary_preview": summary_text[:200],
            },
        )

        return summary_id

    def _generate_summary(self, messages: list[Message]) -> Optional[str]:
        if OllamaClient is None:
            return self._generate_simple_summary(messages)

        model = self._config.compaction_model
        if not model:
            return self._generate_simple_summary(messages)

        conversation_text = "\n".join(f"{m.role.upper()}: {m.content}" for m in messages)

        prompt = f"""Summarize the following conversation in 2-3 sentences, capturing the key topics discussed and any important information exchanged:

{conversation_text}

Summary:"""

        try:
            client = OllamaClient(
                base_url=self._config.ollama_host,
                timeout=60,
            )
            response = client.generate(
                model=model,
                prompt=prompt,
                options={"temperature": 0.3},
            )
            client.close()

            if isinstance(response, dict):
                return response.get("response", "").strip()
            return None
        except Exception:
            return self._generate_simple_summary(messages)

    def _generate_simple_summary(self, messages: list[Message]) -> str:
        user_messages = [m for m in messages if m.role == "user"]
        assistant_messages = [m for m in messages if m.role == "assistant"]

        topics: list[str] = []
        for m in user_messages[:3]:
            preview = m.content[:50]
            if len(m.content) > 50:
                preview += "..."
            topics.append(preview)

        summary_parts = [
            f"Conversation with {len(messages)} messages.",
            f"User asked about: {'; '.join(topics)}" if topics else "",
            f"({len(user_messages)} user, {len(assistant_messages)} assistant messages)",
        ]

        return " ".join(part for part in summary_parts if part)

    def maybe_compact_session(self, session_id: str) -> Optional[str]:
        message_count = self.conversations.get_session_message_count(session_id)
        if message_count >= self._config.compaction_threshold:
            return self.compact_session(session_id)
        return None

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
