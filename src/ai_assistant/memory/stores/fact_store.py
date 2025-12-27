from datetime import datetime
from typing import Any, Optional

from ai_assistant.memory.clients.embeddings import EmbeddingService
from ai_assistant.memory.clients.postgres import PostgresClient
from ai_assistant.memory.clients.qdrant import QdrantClient
from ai_assistant.memory.exceptions import FactNotFoundError, ImmutableFactError
from ai_assistant.memory.models import Fact, FactCategory


class FactStore:
    def __init__(
        self,
        postgres: PostgresClient,
        qdrant: QdrantClient,
        embeddings: EmbeddingService,
    ):
        self._postgres = postgres
        self._qdrant = qdrant
        self._embeddings = embeddings

    def add_fact(self, fact: Fact) -> str:
        query = """
            INSERT INTO facts (id, category, subject, attribute, value, confidence, source, is_immutable, created_at, updated_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (category, subject, attribute) DO UPDATE SET
                value = EXCLUDED.value,
                confidence = EXCLUDED.confidence,
                updated_at = NOW()
            WHERE NOT facts.is_immutable
            RETURNING id
        """
        result = self._postgres.execute_returning(
            query,
            (
                fact.id,
                fact.category.value,
                fact.subject,
                fact.attribute,
                fact.value,
                fact.confidence,
                fact.source,
                fact.is_immutable,
                fact.created_at,
                fact.updated_at,
            ),
        )

        if result is None:
            existing = self.get_fact_by_key(fact.category, fact.subject, fact.attribute)
            if existing and existing.is_immutable:
                raise ImmutableFactError(
                    f"Cannot update immutable fact: {fact.subject}.{fact.attribute}"
                )
            return fact.id

        fact_id = str(result[0])

        embedding = self._embeddings.embed(fact.to_text())
        self._qdrant.upsert(
            collection=QdrantClient.COLLECTION_FACTS,
            point_id=fact_id,
            vector=embedding,
            payload={
                "fact_id": fact_id,
                "category": fact.category.value,
                "subject": fact.subject,
                "attribute": fact.attribute,
                "value": fact.value,
            },
        )

        return fact_id

    def get_fact(self, fact_id: str) -> Optional[Fact]:
        query = """
            SELECT id, category, subject, attribute, value, confidence, source, is_immutable, created_at, updated_at
            FROM facts WHERE id = %s
        """
        row = self._postgres.fetch_one(query, (fact_id,))
        if row is None:
            return None
        return self._row_to_fact(row)

    def get_fact_by_key(
        self,
        category: FactCategory,
        subject: str,
        attribute: str,
    ) -> Optional[Fact]:
        query = """
            SELECT id, category, subject, attribute, value, confidence, source, is_immutable, created_at, updated_at
            FROM facts WHERE category = %s AND subject = %s AND attribute = %s
        """
        row = self._postgres.fetch_one(query, (category.value, subject, attribute))
        if row is None:
            return None
        return self._row_to_fact(row)

    def update_fact(
        self,
        fact_id: str,
        value: str,
        confidence: float = 1.0,
    ) -> bool:
        existing = self.get_fact(fact_id)
        if existing is None:
            raise FactNotFoundError(f"Fact not found: {fact_id}")
        if existing.is_immutable:
            raise ImmutableFactError(f"Cannot update immutable fact: {fact_id}")

        query = """
            UPDATE facts SET value = %s, confidence = %s, updated_at = NOW()
            WHERE id = %s AND NOT is_immutable
            RETURNING id
        """
        result = self._postgres.execute_returning(query, (value, confidence, fact_id))
        if result is None:
            return False

        updated_fact = Fact(
            id=existing.id,
            category=existing.category,
            subject=existing.subject,
            attribute=existing.attribute,
            value=value,
            confidence=confidence,
            source=existing.source,
            is_immutable=existing.is_immutable,
            created_at=existing.created_at,
            updated_at=datetime.now(),
        )
        embedding = self._embeddings.embed(updated_fact.to_text())
        self._qdrant.upsert(
            collection=QdrantClient.COLLECTION_FACTS,
            point_id=fact_id,
            vector=embedding,
            payload={
                "fact_id": fact_id,
                "category": updated_fact.category.value,
                "subject": updated_fact.subject,
                "attribute": updated_fact.attribute,
                "value": value,
            },
        )
        return True

    def delete_fact(self, fact_id: str) -> bool:
        existing = self.get_fact(fact_id)
        if existing is None:
            return False
        if existing.is_immutable:
            raise ImmutableFactError(f"Cannot delete immutable fact: {fact_id}")

        query = "DELETE FROM facts WHERE id = %s AND NOT is_immutable RETURNING id"
        result = self._postgres.execute_returning(query, (fact_id,))
        if result is None:
            return False

        self._qdrant.delete(QdrantClient.COLLECTION_FACTS, fact_id)
        return True

    def search_facts(self, query: str, limit: int = 5) -> list[Fact]:
        embedding = self._embeddings.embed(query)
        results = self._qdrant.search(
            collection=QdrantClient.COLLECTION_FACTS,
            query_vector=embedding,
            limit=limit,
        )

        facts: list[Fact] = []
        for result in results:
            fact_id = result["payload"].get("fact_id")
            if fact_id:
                fact = self.get_fact(fact_id)
                if fact:
                    facts.append(fact)
        return facts

    def get_facts_by_category(self, category: FactCategory) -> list[Fact]:
        query = """
            SELECT id, category, subject, attribute, value, confidence, source, is_immutable, created_at, updated_at
            FROM facts WHERE category = %s ORDER BY subject, attribute
        """
        rows = self._postgres.fetch_all(query, (category.value,))
        return [self._row_to_fact(row) for row in rows]

    def get_facts_by_subject(self, subject: str) -> list[Fact]:
        query = """
            SELECT id, category, subject, attribute, value, confidence, source, is_immutable, created_at, updated_at
            FROM facts WHERE subject = %s ORDER BY attribute
        """
        rows = self._postgres.fetch_all(query, (subject,))
        return [self._row_to_fact(row) for row in rows]

    def get_all_system_facts(self) -> list[Fact]:
        return self.get_facts_by_category(FactCategory.SYSTEM)

    def _row_to_fact(self, row: tuple[Any, ...]) -> Fact:
        return Fact(
            id=str(row[0]),
            category=FactCategory(row[1]),
            subject=row[2],
            attribute=row[3],
            value=row[4],
            confidence=row[5],
            source=row[6],
            is_immutable=row[7],
            created_at=row[8],
            updated_at=row[9],
        )
