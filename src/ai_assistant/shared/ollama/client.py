"""Ollama client wrapper for interacting with local Ollama instances."""

import json
from typing import Any, Dict, Iterator, List, Optional, Union

import requests


class OllamaError(Exception):
    """Base exception for Ollama client errors."""

    pass


class OllamaConnectionError(OllamaError):
    """Raised when connection to Ollama server fails."""

    pass


class OllamaAPIError(OllamaError):
    """Raised when Ollama API returns an error."""

    pass


class OllamaClient:
    """
    Client wrapper for Ollama API.

    Provides methods for chat completions, embeddings, and model management
    with a local Ollama instance.

    Args:
        base_url: Base URL for the Ollama API (default: http://localhost:11434)
        timeout: Request timeout in seconds (default: 30)

    Example:
        >>> client = OllamaClient()
        >>> response = client.chat(
        ...     model="llama2",
        ...     messages=[{"role": "user", "content": "Hello!"}]
        ... )
        >>> print(response["message"]["content"])
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        timeout: int = 30,
    ) -> None:
        """Initialize Ollama client."""
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._session = requests.Session()

    def _make_request(
        self,
        endpoint: str,
        method: str = "POST",
        json_data: Optional[Dict[str, Any]] = None,
        stream: bool = False,
    ) -> Union[Dict[str, Any], requests.Response]:
        """
        Make HTTP request to Ollama API.

        Args:
            endpoint: API endpoint path
            method: HTTP method (GET, POST, etc.)
            json_data: JSON payload for request
            stream: Whether to stream the response

        Returns:
            Response data as dict, or Response object if streaming

        Raises:
            OllamaConnectionError: If connection fails
            OllamaAPIError: If API returns error response
        """
        url = f"{self.base_url}{endpoint}"

        try:
            response = self._session.request(
                method=method,
                url=url,
                json=json_data,
                timeout=self.timeout,
                stream=stream,
            )
            response.raise_for_status()

            if stream:
                return response

            return response.json()

        except requests.exceptions.ConnectionError as e:
            raise OllamaConnectionError(
                f"Failed to connect to Ollama server at {self.base_url}"
            ) from e
        except requests.exceptions.Timeout as e:
            raise OllamaConnectionError(
                f"Request to Ollama server timed out after {self.timeout}s"
            ) from e
        except requests.exceptions.HTTPError as e:
            error_msg = f"Ollama API error: {e}"
            if e.response is not None:
                try:
                    error_data = e.response.json()
                    error_msg = f"Ollama API error: {error_data.get('error', str(e))}"
                except json.JSONDecodeError:
                    pass
            raise OllamaAPIError(error_msg) from e
        except requests.exceptions.RequestException as e:
            raise OllamaError(f"Request failed: {e}") from e

    def chat(
        self,
        model: str,
        messages: List[Dict[str, str]],
        stream: bool = False,
        options: Optional[Dict[str, Any]] = None,
        format: Optional[str] = None,
        keep_alive: Optional[str] = None,
        think: Optional[bool] = None,
    ) -> Union[Dict[str, Any], Iterator[Dict[str, Any]]]:
        """
        Generate a chat completion.

        Args:
            model: Name of the model to use
            messages: List of message dicts with 'role' and 'content' keys
            stream: Whether to stream the response
            options: Model-specific options (temperature, top_p, etc.)
            format: Response format ("json" for JSON mode)
            keep_alive: How long to keep model loaded (e.g., "5m", "1h")
            think: Whether to enable chain-of-thought reasoning (for models that support it)

        Returns:
            Response dict or iterator of response chunks if streaming

        Example:
            >>> response = client.chat(
            ...     model="llama2",
            ...     messages=[{"role": "user", "content": "Hello!"}],
            ...     options={"temperature": 0.7}
            ... )
        """
        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": stream,
        }

        if options is not None:
            payload["options"] = options
        if format is not None:
            payload["format"] = format
        if keep_alive is not None:
            payload["keep_alive"] = keep_alive
        if think is not None:
            payload["think"] = think

        if stream:
            response = self._make_request("/api/chat", json_data=payload, stream=True)
            return self._stream_response(response)  # type: ignore

        return self._make_request("/api/chat", json_data=payload)  # type: ignore

    def generate(
        self,
        model: str,
        prompt: str,
        stream: bool = False,
        options: Optional[Dict[str, Any]] = None,
        format: Optional[str] = None,
        images: Optional[List[str]] = None,
        keep_alive: Optional[str] = None,
    ) -> Union[Dict[str, Any], Iterator[Dict[str, Any]]]:
        """
        Generate a completion from a prompt.

        Args:
            model: Name of the model to use
            prompt: Input prompt text
            stream: Whether to stream the response
            options: Model-specific options
            format: Response format ("json" for JSON mode)
            images: List of base64-encoded images for multimodal models
            keep_alive: How long to keep model loaded

        Returns:
            Response dict or iterator of response chunks if streaming
        """
        payload: Dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "stream": stream,
        }

        if options is not None:
            payload["options"] = options
        if format is not None:
            payload["format"] = format
        if images is not None:
            payload["images"] = images
        if keep_alive is not None:
            payload["keep_alive"] = keep_alive

        if stream:
            response = self._make_request("/api/generate", json_data=payload, stream=True)
            return self._stream_response(response)  # type: ignore

        return self._make_request("/api/generate", json_data=payload)  # type: ignore

    def embeddings(
        self,
        model: str,
        prompt: str,
        options: Optional[Dict[str, Any]] = None,
        keep_alive: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Generate embeddings for a prompt.

        Args:
            model: Name of the model to use
            prompt: Input text to embed
            options: Model-specific options
            keep_alive: How long to keep model loaded

        Returns:
            Response dict with 'embedding' key containing the vector

        Example:
            >>> response = client.embeddings(
            ...     model="llama2",
            ...     prompt="Hello world"
            ... )
            >>> embedding = response["embedding"]
        """
        payload: Dict[str, Any] = {
            "model": model,
            "prompt": prompt,
        }

        if options is not None:
            payload["options"] = options
        if keep_alive is not None:
            payload["keep_alive"] = keep_alive

        return self._make_request("/api/embeddings", json_data=payload)  # type: ignore

    def list_models(self) -> Dict[str, Any]:
        """
        List available models.

        Returns:
            Dict with 'models' key containing list of model info

        Example:
            >>> models = client.list_models()
            >>> for model in models["models"]:
            ...     print(model["name"])
        """
        return self._make_request("/api/tags", method="GET")  # type: ignore

    def show_model(self, model: str) -> Dict[str, Any]:
        """
        Show information about a model.

        Args:
            model: Name of the model

        Returns:
            Dict with model information including modelfile, parameters, etc.

        Example:
            >>> info = client.show_model("llama2")
            >>> print(info["modelfile"])
        """
        payload = {"name": model}
        return self._make_request("/api/show", json_data=payload)  # type: ignore

    def pull_model(
        self,
        model: str,
        stream: bool = False,
        insecure: bool = False,
    ) -> Union[Dict[str, Any], Iterator[Dict[str, Any]]]:
        """
        Pull a model from the registry.

        Args:
            model: Name of the model to pull
            stream: Whether to stream download progress
            insecure: Allow insecure connections

        Returns:
            Response dict or iterator of progress updates if streaming

        Example:
            >>> for progress in client.pull_model("llama2", stream=True):
            ...     print(progress.get("status"))
        """
        payload: Dict[str, Any] = {
            "name": model,
            "stream": stream,
        }

        if insecure:
            payload["insecure"] = insecure

        if stream:
            response = self._make_request("/api/pull", json_data=payload, stream=True)
            return self._stream_response(response)  # type: ignore

        return self._make_request("/api/pull", json_data=payload)  # type: ignore

    def delete_model(self, model: str) -> Dict[str, Any]:
        """
        Delete a model.

        Args:
            model: Name of the model to delete

        Returns:
            Response dict

        Example:
            >>> client.delete_model("llama2")
        """
        payload = {"name": model}
        return self._make_request("/api/delete", method="DELETE", json_data=payload)  # type: ignore

    def copy_model(self, source: str, destination: str) -> Dict[str, Any]:
        """
        Copy a model.

        Args:
            source: Name of source model
            destination: Name for the copy

        Returns:
            Response dict

        Example:
            >>> client.copy_model("llama2", "my-llama2")
        """
        payload = {"source": source, "destination": destination}
        return self._make_request("/api/copy", json_data=payload)  # type: ignore

    def _stream_response(self, response: requests.Response) -> Iterator[Dict[str, Any]]:
        """
        Stream response chunks from Ollama API.

        Args:
            response: Response object with streaming enabled

        Yields:
            Dict containing response chunk data
        """
        try:
            for line in response.iter_lines():
                if line:
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError as e:
                        raise OllamaAPIError(f"Invalid JSON in stream: {e}") from e
        finally:
            response.close()

    def close(self) -> None:
        """Close the HTTP session."""
        self._session.close()

    def __enter__(self) -> "OllamaClient":
        """Context manager entry."""
        return self

    def __exit__(self, *args: Any) -> None:
        """Context manager exit."""
        self.close()
