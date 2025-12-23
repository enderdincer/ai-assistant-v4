"""Ollama client module for AI Assistant."""

from .client import (
    OllamaAPIError,
    OllamaClient,
    OllamaConnectionError,
    OllamaError,
)

__all__ = [
    "OllamaClient",
    "OllamaError",
    "OllamaConnectionError",
    "OllamaAPIError",
]
