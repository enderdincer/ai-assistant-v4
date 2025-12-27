from typing import Any, Optional, Sequence

from ai_assistant.memory.config import MemoryConfig
from ai_assistant.memory.exceptions import CollectionNotFoundError, MemoryConnectionError

try:
    from qdrant_client import QdrantClient as QdrantClientLib
    from qdrant_client.http import models as qdrant_models
    from qdrant_client.http.exceptions import UnexpectedResponse
except ImportError:
    QdrantClientLib = None  # type: ignore
    qdrant_models = None  # type: ignore
    UnexpectedResponse = Exception  # type: ignore


class QdrantClient:
    COLLECTION_FACTS = "facts"
    COLLECTION_CONVERSATIONS = "conversations"

    def __init__(self, config: Optional[MemoryConfig] = None):
        if QdrantClientLib is None:
            raise ImportError(
                "qdrant-client is not installed. Install with: pip install qdrant-client"
            )
        self._config = config or MemoryConfig.from_env()
        self._client: Optional[QdrantClientLib] = None
        self._dimension = self._config.embedding_dimension

    def initialize(self) -> None:
        try:
            self._client = QdrantClientLib(
                host=self._config.qdrant_host,
                port=self._config.qdrant_port,
            )
            self._ensure_collections()
        except Exception as e:
            raise MemoryConnectionError(f"Failed to connect to Qdrant: {e}") from e

    def _ensure_collections(self) -> None:
        if self._client is None:
            raise MemoryConnectionError("QdrantClient not initialized")

        collections = [self.COLLECTION_FACTS, self.COLLECTION_CONVERSATIONS]
        existing = {c.name for c in self._client.get_collections().collections}

        for collection_name in collections:
            if collection_name not in existing:
                self._client.create_collection(
                    collection_name=collection_name,
                    vectors_config=qdrant_models.VectorParams(
                        size=self._dimension,
                        distance=qdrant_models.Distance.COSINE,
                    ),
                )

    def upsert(
        self,
        collection: str,
        point_id: str,
        vector: list[float],
        payload: dict[str, Any],
    ) -> None:
        if self._client is None:
            raise MemoryConnectionError("QdrantClient not initialized")
        try:
            self._client.upsert(
                collection_name=collection,
                points=[
                    qdrant_models.PointStruct(
                        id=point_id,
                        vector=vector,
                        payload=payload,
                    )
                ],
            )
        except UnexpectedResponse as e:
            if "not found" in str(e).lower():
                raise CollectionNotFoundError(f"Collection '{collection}' not found") from e
            raise

    def search(
        self,
        collection: str,
        query_vector: list[float],
        limit: int = 5,
        filter_conditions: Optional[dict[str, Any]] = None,
    ) -> list[dict[str, Any]]:
        if self._client is None:
            raise MemoryConnectionError("QdrantClient not initialized")

        query_filter = None
        if filter_conditions:
            must_conditions: Sequence[qdrant_models.FieldCondition] = []
            conditions_list: list[qdrant_models.FieldCondition] = []
            for key, value in filter_conditions.items():
                conditions_list.append(
                    qdrant_models.FieldCondition(
                        key=key,
                        match=qdrant_models.MatchValue(value=value),
                    )
                )
            must_conditions = conditions_list
            query_filter = qdrant_models.Filter(must=must_conditions)

        try:
            response = self._client.query_points(
                collection_name=collection,
                query=query_vector,
                limit=limit,
                query_filter=query_filter,
                with_payload=True,
            )
            return [
                {
                    "id": str(r.id),
                    "score": r.score,
                    "payload": r.payload or {},
                }
                for r in response.points
            ]
        except UnexpectedResponse as e:
            if "not found" in str(e).lower():
                raise CollectionNotFoundError(f"Collection '{collection}' not found") from e
            raise

    def delete(self, collection: str, point_id: str) -> None:
        if self._client is None:
            raise MemoryConnectionError("QdrantClient not initialized")
        try:
            self._client.delete(
                collection_name=collection,
                points_selector=qdrant_models.PointIdsList(points=[point_id]),
            )
        except UnexpectedResponse as e:
            if "not found" in str(e).lower():
                raise CollectionNotFoundError(f"Collection '{collection}' not found") from e
            raise

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None

    def __enter__(self) -> "QdrantClient":
        self.initialize()
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
