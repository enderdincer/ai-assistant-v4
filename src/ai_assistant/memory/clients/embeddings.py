"""Local embedding service using sentence-transformers.

Provides text embeddings without external API dependencies.
The embedding model runs locally, making the memory service self-contained.
"""

from typing import Optional, TYPE_CHECKING

from ai_assistant.memory.config import MemoryConfig
from ai_assistant.memory.exceptions import EmbeddingError

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer


class EmbeddingService:
    """Local embedding service using sentence-transformers.

    Uses a lightweight model that runs locally without requiring
    external services like Ollama.
    """

    # Default model - high quality for semantic search
    # all-mpnet-base-v2: 768 dimensions, ~420MB, best quality
    DEFAULT_MODEL = "all-mpnet-base-v2"
    DEFAULT_DIMENSION = 768

    def __init__(self, config: Optional[MemoryConfig] = None):
        self._config = config or MemoryConfig.from_env()
        self._model_name = self._config.embedding_model or self.DEFAULT_MODEL
        self._dimension = self._config.embedding_dimension or self.DEFAULT_DIMENSION
        self._model: Optional["SentenceTransformer"] = None

    def initialize(self) -> None:
        """Initialize the embedding model.

        Loads the sentence-transformers model. The first call will
        download the model if not already cached.
        """
        try:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self._model_name)

            # Update dimension based on actual model
            model_dim = self._model.get_sentence_embedding_dimension()
            if model_dim is not None:
                self._dimension = model_dim

        except ImportError as e:
            raise EmbeddingError(
                "sentence-transformers not installed. "
                "Install with: pip install sentence-transformers"
            ) from e
        except Exception as e:
            raise EmbeddingError(f"Failed to load embedding model '{self._model_name}': {e}") from e

    def embed(self, text: str) -> list[float]:
        """Generate embedding for a single text.

        Args:
            text: Text to embed

        Returns:
            List of floats representing the embedding vector

        Raises:
            EmbeddingError: If model not initialized or embedding fails
        """
        if self._model is None:
            raise EmbeddingError("EmbeddingService not initialized. Call initialize() first.")

        try:
            # SentenceTransformer.encode returns numpy array
            embedding = self._model.encode(text, convert_to_numpy=True)
            result: list[float] = embedding.tolist()
            return result
        except Exception as e:
            raise EmbeddingError(f"Failed to generate embedding: {e}") from e

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts.

        More efficient than calling embed() multiple times as it
        batches the computation.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors

        Raises:
            EmbeddingError: If model not initialized or embedding fails
        """
        if self._model is None:
            raise EmbeddingError("EmbeddingService not initialized. Call initialize() first.")

        if not texts:
            return []

        try:
            embeddings = self._model.encode(texts, convert_to_numpy=True)
            return [e.tolist() for e in embeddings]
        except Exception as e:
            raise EmbeddingError(f"Failed to generate embeddings: {e}") from e

    @property
    def dimension(self) -> int:
        """Get the embedding dimension."""
        return self._dimension

    @property
    def model_name(self) -> str:
        """Get the model name."""
        return self._model_name

    def close(self) -> None:
        """Clean up resources."""
        self._model = None

    def __enter__(self) -> "EmbeddingService":
        self.initialize()
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
