from typing import Optional

from ai_assistant.memory.config import MemoryConfig
from ai_assistant.memory.exceptions import EmbeddingError, MemoryConnectionError
from ai_assistant.shared.ollama import OllamaClient, OllamaError


class EmbeddingService:
    def __init__(self, config: Optional[MemoryConfig] = None):
        self._config = config or MemoryConfig.from_env()
        self._client: Optional[OllamaClient] = None
        self._model = self._config.embedding_model
        self._dimension = self._config.embedding_dimension

    def initialize(self) -> None:
        self._client = OllamaClient(
            base_url=self._config.ollama_host,
            timeout=60,
        )
        try:
            models = self._client.list_models()
            available = [m["name"] for m in models.get("models", [])]
            if self._model not in available and f"{self._model}:latest" not in available:
                raise MemoryConnectionError(
                    f"Embedding model '{self._model}' not available. Run: ollama pull {self._model}"
                )
        except OllamaError as e:
            raise MemoryConnectionError(f"Failed to connect to Ollama: {e}") from e

    def embed(self, text: str) -> list[float]:
        if self._client is None:
            raise EmbeddingError("EmbeddingService not initialized")
        try:
            response = self._client.embeddings(model=self._model, prompt=text)
            embedding = response.get("embedding", [])
            if not embedding:
                raise EmbeddingError("Empty embedding returned from Ollama")
            return embedding
        except OllamaError as e:
            raise EmbeddingError(f"Failed to generate embedding: {e}") from e

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(text) for text in texts]

    @property
    def dimension(self) -> int:
        return self._dimension

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None

    def __enter__(self) -> "EmbeddingService":
        self.initialize()
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
