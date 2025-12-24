#!/usr/bin/env python3
"""
Test script for the memory system.

Requirements:
1. PostgreSQL and Qdrant running (docker-compose up -d)
2. Ollama running with nomic-embed-text model (ollama pull nomic-embed-text)
3. Dependencies installed: pip install 'ai-assistant[memory]'

Usage:
    python test_memory.py
"""

import os
import sys

os.environ.setdefault("POSTGRES_HOST", "localhost")
os.environ.setdefault("POSTGRES_PORT", "5432")
os.environ.setdefault("POSTGRES_USER", "ai_assistant")
os.environ.setdefault("POSTGRES_PASSWORD", "ai_assistant_secret")
os.environ.setdefault("POSTGRES_DB", "ai_assistant")
os.environ.setdefault("QDRANT_HOST", "localhost")
os.environ.setdefault("QDRANT_HTTP_PORT", "6333")
os.environ.setdefault("EMBEDDING_MODEL", "nomic-embed-text")
os.environ.setdefault("OLLAMA_HOST", "http://localhost:11434")

from ai_assistant.memory import (
    MemoryManager,
    MemoryConfig,
    Fact,
    FactCategory,
    MemoryError,
)


def test_memory_system() -> None:
    print("Testing Memory System")
    print("=" * 50)

    config = MemoryConfig.from_env()
    print(f"PostgreSQL: {config.postgres_host}:{config.postgres_port}/{config.postgres_db}")
    print(f"Qdrant: {config.qdrant_host}:{config.qdrant_port}")
    print(f"Embedding model: {config.embedding_model}")
    print()

    try:
        with MemoryManager(config) as memory:
            print("[OK] Memory system initialized")

            session_id = memory.create_session()
            print(f"[OK] Created session: {session_id[:8]}...")

            print("\n--- Testing Fact Store ---")

            # Try to add system fact, but it might already exist from previous runs
            try:
                system_fact_id = memory.add_system_fact(
                    subject="assistant",
                    attribute="name",
                    value="Nova",
                )
                print(
                    f"[OK] Added system fact: assistant.name = Nova (id: {system_fact_id[:8]}...)"
                )
            except MemoryError as e:
                if "immutable" in str(e).lower():
                    print("[OK] System fact assistant.name already exists (immutable)")
                else:
                    raise

            learned_fact_id = memory.remember_fact(
                subject="user",
                attribute="favorite_color",
                value="blue",
                confidence=0.9,
            )
            print(
                f"[OK] Added learned fact: user.favorite_color = blue (id: {learned_fact_id[:8]}...)"
            )

            facts = memory.facts.search_facts("What is the assistant's name?", limit=3)
            print(f"[OK] Searched facts, found {len(facts)} results")
            for fact in facts:
                print(f"     - {fact.subject}.{fact.attribute} = {fact.value}")

            print("\n--- Testing Conversation Store ---")

            msg1_id = memory.log_conversation(
                session_id=session_id,
                role="user",
                content="Hello, what's your name?",
            )
            print(f"[OK] Logged user message (id: {msg1_id[:8]}...)")

            msg2_id = memory.log_conversation(
                session_id=session_id,
                role="assistant",
                content="Hello! My name is Nova. How can I help you today?",
            )
            print(f"[OK] Logged assistant message (id: {msg2_id[:8]}...)")

            session_messages = memory.conversations.get_session(session_id)
            print(f"[OK] Retrieved {len(session_messages)} messages from session")

            similar = memory.conversations.search_similar("What is your name?", limit=3)
            print(f"[OK] Found {len(similar)} similar conversations")

            print("\n--- Testing Memory Context ---")

            context = memory.get_context_for_query(
                query="What color does the user like?",
                session_id=session_id,
                max_facts=5,
                max_similar=3,
            )
            print(f"[OK] Got memory context:")
            print(f"     - {len(context.relevant_facts)} relevant facts")
            print(f"     - {len(context.recent_messages)} recent messages")
            print(f"     - {len(context.similar_conversations)} similar conversations")

            if not context.is_empty():
                prompt_addition = context.to_system_prompt_addition()
                print(f"     - Context text length: {len(prompt_addition)} chars")

            print("\n--- Testing Event Log ---")

            event_id = memory.events.log_event(
                event_type="test",
                session_id=session_id,
                data={"action": "memory_test", "result": "success"},
            )
            print(f"[OK] Logged event (id: {event_id[:8]}...)")

            events = memory.events.get_events(session_id=session_id)
            print(f"[OK] Retrieved {len(events)} events from session")

            print("\n" + "=" * 50)
            print("All tests passed!")

    except MemoryError as e:
        print(f"\n[ERROR] Memory error: {e}")
        print("\nMake sure:")
        print("  1. PostgreSQL is running: docker-compose up -d postgres")
        print("  2. Qdrant is running: docker-compose up -d qdrant")
        print("  3. Ollama has the embedding model: ollama pull nomic-embed-text")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] Unexpected error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    test_memory_system()
