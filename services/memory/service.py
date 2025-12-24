"""Memory Service implementation.

Provides persistent memory storage and retrieval:
1. Listens to transcriptions and assistant responses for auto-logging
2. Handles memory query requests and returns context
3. Stores facts from extraction service
4. Auto-compacts long conversations
"""

import os
import threading
from dataclasses import dataclass
from typing import Any, Optional

from ai_assistant.shared.logging import get_logger, LogLevel
from ai_assistant.shared.services import BaseService, ServiceConfig
from ai_assistant.shared.messages import (
    TranscriptionMessage,
    AssistantResponseMessage,
    MemoryQueryMessage,
    MemoryResponseMessage,
    MemoryStoreMessage,
    FactMessage,
)
from ai_assistant.shared.mqtt.topics import Topics
from ai_assistant.memory import MemoryManager, MemoryConfig, Fact

logger = get_logger(__name__)


@dataclass
class MemoryServiceConfig(ServiceConfig):
    """Configuration for Memory Service.

    Inherits database configuration from MemoryConfig.

    Attributes:
        auto_log_conversations: Whether to auto-log transcriptions/responses
        auto_compact: Whether to auto-compact long conversations
        default_session_id: Default session ID for voice transcriptions
    """

    # Database connection settings (matching MemoryConfig structure)
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_user: str = "ai_assistant"
    postgres_password: str = "ai_assistant_secret"
    postgres_db: str = "ai_assistant"
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    ollama_host: str = "http://localhost:11434"
    embedding_model: str = "nomic-embed-text"
    compaction_model: str = "qwen3:1.7b"
    compaction_threshold: int = 20
    compaction_keep_recent: int = 5

    # Service-specific settings
    auto_log_conversations: bool = True
    auto_compact: bool = True
    default_session_id: str = "default"

    @classmethod
    def from_env(cls) -> "MemoryServiceConfig":
        """Create configuration from environment variables."""
        return cls(
            service_name="memory-service",
            postgres_host=os.getenv("POSTGRES_HOST", "localhost"),
            postgres_port=int(os.getenv("POSTGRES_PORT", "5432")),
            postgres_user=os.getenv("POSTGRES_USER", "ai_assistant"),
            postgres_password=os.getenv("POSTGRES_PASSWORD", "ai_assistant_secret"),
            postgres_db=os.getenv("POSTGRES_DB", "ai_assistant"),
            qdrant_host=os.getenv("QDRANT_HOST", "localhost"),
            qdrant_port=int(os.getenv("QDRANT_PORT", "6333")),
            ollama_host=os.getenv("OLLAMA_HOST", "http://localhost:11434"),
            embedding_model=os.getenv("EMBEDDING_MODEL", "nomic-embed-text"),
            compaction_model=os.getenv("COMPACTION_MODEL", "qwen3:1.7b"),
            compaction_threshold=int(os.getenv("COMPACTION_THRESHOLD", "20")),
            compaction_keep_recent=int(os.getenv("COMPACTION_KEEP_RECENT", "5")),
            auto_log_conversations=os.getenv("AUTO_LOG_CONVERSATIONS", "true").lower()
            in ("1", "true", "yes"),
            auto_compact=os.getenv("AUTO_COMPACT", "true").lower() in ("1", "true", "yes"),
            default_session_id=os.getenv("DEFAULT_SESSION_ID", "default"),
            log_level=LogLevel.DEBUG
            if os.getenv("DEBUG", "").lower() in ("1", "true", "yes")
            else LogLevel.INFO,
        )

    def to_memory_config(self) -> MemoryConfig:
        """Convert to MemoryConfig for MemoryManager."""
        return MemoryConfig(
            postgres_host=self.postgres_host,
            postgres_port=self.postgres_port,
            postgres_user=self.postgres_user,
            postgres_password=self.postgres_password,
            postgres_db=self.postgres_db,
            qdrant_host=self.qdrant_host,
            qdrant_port=self.qdrant_port,
            ollama_host=self.ollama_host,
            embedding_model=self.embedding_model,
            compaction_model=self.compaction_model,
            compaction_threshold=self.compaction_threshold,
            compaction_keep_recent=self.compaction_keep_recent,
        )


class MemoryService(BaseService):
    """Service providing persistent memory storage and retrieval.

    This service:
    1. Subscribes to transcriptions, responses, and memory requests
    2. Auto-logs conversations to the memory store
    3. Handles memory queries and returns relevant context
    4. Stores facts received from extraction service
    """

    def __init__(self, config: MemoryServiceConfig) -> None:
        """Initialize the memory service.

        Args:
            config: Service configuration
        """
        super().__init__(config)
        self._memory_config = config

        # Memory manager (initialized in _setup)
        self._memory: Optional[MemoryManager] = None

        # Processing lock
        self._processing_lock = threading.Lock()

        # Topics
        self._transcription_topic = Topics.EVENT_AUDIO_TRANSCRIBED.topic
        self._response_topic = Topics.EVENT_ASSISTANT_RESPONSE.topic
        self._query_topic = Topics.MEMORY_QUERY.topic
        self._query_response_topic = Topics.MEMORY_RESPONSE.topic
        self._store_topic = Topics.MEMORY_STORE.topic
        self._facts_topic = Topics.MEMORY_FACTS.topic

    def _setup(self) -> None:
        """Initialize memory manager and subscribe to topics."""
        # Initialize memory manager
        memory_config = self._memory_config.to_memory_config()
        self._memory = MemoryManager(memory_config)

        try:
            self._memory.initialize()
            self._logger.info("Memory manager initialized")
        except Exception as e:
            self._logger.error(f"Failed to initialize memory: {e}")
            raise

        # Subscribe to topics
        self._subscribe(self._query_topic, self._on_query)
        self._subscribe(self._store_topic, self._on_store)
        self._subscribe(self._facts_topic, self._on_facts)

        # Auto-log if enabled
        if self._memory_config.auto_log_conversations:
            self._subscribe(self._transcription_topic, self._on_transcription)
            self._subscribe(self._response_topic, self._on_response)
            self._logger.info("Auto-logging enabled for conversations")

        self._logger.info("Memory service ready")

    def _cleanup(self) -> None:
        """Clean up memory manager."""
        if self._memory:
            self._memory.close()
            self._memory = None

        self._logger.info("Memory service cleaned up")

    def _on_transcription(self, topic: str, payload: bytes) -> None:
        """Handle incoming transcription for auto-logging.

        Args:
            topic: MQTT topic
            payload: Message payload
        """
        try:
            message = TranscriptionMessage.from_bytes(payload)
            text = message.text.strip()

            if not text:
                return

            # Use session from metadata or default
            session_id = self._memory_config.default_session_id

            # Log as user message
            self._log_message(
                session_id=session_id,
                role="user",
                content=text,
                metadata={
                    "source": "voice",
                    "audio_source": message.audio_source,
                    "confidence": message.confidence,
                },
            )

        except Exception as e:
            self._logger.error(f"Error handling transcription: {e}")

    def _on_response(self, topic: str, payload: bytes) -> None:
        """Handle incoming assistant response for auto-logging.

        Args:
            topic: MQTT topic
            payload: Message payload
        """
        try:
            message = AssistantResponseMessage.from_bytes(payload)
            text = message.text.strip()

            if not text:
                return

            # Use session from message or default
            session_id = message.session_id or self._memory_config.default_session_id

            # Log as assistant message
            self._log_message(
                session_id=session_id,
                role="assistant",
                content=text,
                metadata={
                    "model": message.model_name,
                    "tokens_used": message.tokens_used,
                    "input_text": message.input_text,
                },
            )

        except Exception as e:
            self._logger.error(f"Error handling response: {e}")

    def _on_store(self, topic: str, payload: bytes) -> None:
        """Handle explicit store request.

        Args:
            topic: MQTT topic
            payload: Message payload
        """
        try:
            message = MemoryStoreMessage.from_bytes(payload)

            if not message.content:
                return

            self._log_message(
                session_id=message.session_id,
                role=message.role,
                content=message.content,
                metadata=message.metadata,
            )

        except Exception as e:
            self._logger.error(f"Error handling store request: {e}")

    def _on_query(self, topic: str, payload: bytes) -> None:
        """Handle memory query request.

        Args:
            topic: MQTT topic
            payload: Message payload
        """
        try:
            message = MemoryQueryMessage.from_bytes(payload)

            if not message.query:
                self._logger.warning("Empty query received")
                return

            # Query memory
            context = self._query_memory(message)

            # Build response
            response = MemoryResponseMessage(
                request_id=message.request_id,
                relevant_facts=[self._fact_to_dict(f) for f in context.relevant_facts],
                recent_messages=[self._message_to_dict(m) for m in context.recent_messages],
                similar_conversations=[
                    self._message_to_dict(m) for m in context.similar_conversations
                ],
                source=self.service_name,
            )

            # Publish response
            self._publish(self._query_response_topic, response.to_bytes())
            self._logger.debug(
                f"Query response sent: {len(context.relevant_facts)} facts, "
                f"{len(context.recent_messages)} messages"
            )

        except Exception as e:
            self._logger.error(f"Error handling query: {e}")

    def _on_facts(self, topic: str, payload: bytes) -> None:
        """Handle incoming facts from extraction service.

        Args:
            topic: MQTT topic
            payload: Message payload
        """
        try:
            message = FactMessage.from_bytes(payload)

            if not message.facts:
                return

            self._store_facts(message.facts)

        except Exception as e:
            self._logger.error(f"Error handling facts: {e}")

    def _log_message(
        self,
        session_id: str,
        role: str,
        content: str,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        """Log a message to memory.

        Args:
            session_id: Session ID
            role: Message role (user/assistant)
            content: Message content
            metadata: Optional metadata
        """
        if self._memory is None:
            self._logger.warning("Memory not initialized")
            return

        with self._processing_lock:
            try:
                message_id = self._memory.log_conversation(
                    session_id=session_id,
                    role=role,
                    content=content,
                    metadata=metadata,
                    auto_compact=self._memory_config.auto_compact,
                )
                self._logger.debug(f"Logged {role} message: {message_id}")

            except Exception as e:
                self._logger.error(f"Failed to log message: {e}")

    def _query_memory(self, query: MemoryQueryMessage) -> Any:
        """Query memory for context.

        Args:
            query: Query message

        Returns:
            MemoryContext with relevant facts, messages, etc.
        """
        if self._memory is None:
            raise RuntimeError("Memory not initialized")

        return self._memory.get_context_for_query(
            query=query.query,
            session_id=query.session_id or None,
            max_facts=query.max_facts,
            max_similar=query.max_similar,
        )

    def _store_facts(self, facts: list[dict[str, Any]]) -> None:
        """Store extracted facts.

        Args:
            facts: List of fact dictionaries
        """
        if self._memory is None:
            self._logger.warning("Memory not initialized")
            return

        with self._processing_lock:
            for fact_data in facts:
                try:
                    # Create fact using the appropriate factory method
                    # All extracted facts are "learned" facts
                    fact = Fact.create_learned_fact(
                        subject=fact_data.get("subject", "unknown"),
                        attribute=fact_data.get("attribute", ""),
                        value=fact_data.get("value", ""),
                        confidence=fact_data.get("confidence", 0.8),
                        source=fact_data.get("source", "extraction"),
                    )

                    # Store
                    fact_id = self._memory.facts.add_fact(fact)
                    self._logger.debug(
                        f"Stored fact: {fact.subject}.{fact.attribute} = {fact.value} (id={fact_id})"
                    )

                except Exception as e:
                    self._logger.error(f"Failed to store fact: {e}")

    def _fact_to_dict(self, fact: Fact) -> dict[str, Any]:
        """Convert Fact to dictionary for serialization.

        Args:
            fact: Fact instance

        Returns:
            Dictionary representation
        """
        return {
            "id": fact.id,
            "subject": fact.subject,
            "attribute": fact.attribute,
            "value": fact.value,
            "category": fact.category.value
            if hasattr(fact.category, "value")
            else str(fact.category),
            "confidence": fact.confidence,
            "source": fact.source,
            "created_at": fact.created_at.isoformat() if fact.created_at else None,
            "updated_at": fact.updated_at.isoformat() if fact.updated_at else None,
        }

    def _message_to_dict(self, message: Any) -> dict[str, Any]:
        """Convert Message to dictionary for serialization.

        Args:
            message: Message instance

        Returns:
            Dictionary representation
        """
        return {
            "id": getattr(message, "id", ""),
            "session_id": getattr(message, "session_id", ""),
            "role": getattr(message, "role", ""),
            "content": getattr(message, "content", ""),
            "timestamp": message.timestamp.isoformat()
            if hasattr(message, "timestamp") and message.timestamp
            else None,
            "metadata": getattr(message, "metadata", {}),
        }
