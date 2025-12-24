"""Extraction Service implementation.

Extracts facts from conversations using an LLM:
1. Listens to assistant responses (which include user input)
2. Uses LLM to extract facts about the user
3. Publishes extracted facts to memory service
"""

import json
import os
import re
import threading
from dataclasses import dataclass
from typing import Any, Optional

from ai_assistant.shared.logging import get_logger, LogLevel
from ai_assistant.shared.services import BaseService, ServiceConfig
from ai_assistant.shared.messages import (
    AssistantResponseMessage,
    FactMessage,
)
from ai_assistant.shared.mqtt.topics import Topics
from ai_assistant.shared.ollama import OllamaClient, OllamaError

logger = get_logger(__name__)

# Prompt template for fact extraction
EXTRACTION_PROMPT = """Analyze the following conversation exchange and extract any facts about the user.

USER: {user_input}
ASSISTANT: {assistant_response}

Extract facts in the following categories:
- Personal information (name, age, location, occupation, etc.)
- Preferences (likes, dislikes, preferences)
- Habits or routines
- Relationships (family, friends, pets)
- Goals or interests

Return the facts as a JSON array. Each fact should have:
- "subject": who/what the fact is about (usually "user")
- "attribute": the type of information (e.g., "name", "favorite_food", "has_pet")
- "value": the actual information
- "confidence": 0.0 to 1.0 how confident you are in this fact

If no facts can be extracted, return an empty array: []

Only extract facts that are explicitly stated or strongly implied. Do not make assumptions.

JSON output:"""


@dataclass
class ExtractionServiceConfig(ServiceConfig):
    """Configuration for Extraction Service.

    Attributes:
        ollama_host: Ollama API host
        ollama_model: LLM model to use for extraction
        ollama_timeout: Request timeout in seconds
        min_confidence: Minimum confidence to keep a fact
        extraction_enabled: Whether extraction is enabled
    """

    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "qwen3:1.7b"
    ollama_timeout: int = 60
    min_confidence: float = 0.5
    extraction_enabled: bool = True

    @classmethod
    def from_env(cls) -> "ExtractionServiceConfig":
        """Create configuration from environment variables."""
        return cls(
            service_name="extraction-service",
            ollama_host=os.getenv("OLLAMA_HOST", "http://localhost:11434"),
            ollama_model=os.getenv("EXTRACTION_MODEL", os.getenv("OLLAMA_MODEL", "qwen3:1.7b")),
            ollama_timeout=int(os.getenv("EXTRACTION_TIMEOUT", "60")),
            min_confidence=float(os.getenv("MIN_CONFIDENCE", "0.5")),
            extraction_enabled=os.getenv("EXTRACTION_ENABLED", "true").lower()
            in ("1", "true", "yes"),
            log_level=LogLevel.DEBUG
            if os.getenv("DEBUG", "").lower() in ("1", "true", "yes")
            else LogLevel.INFO,
        )


class ExtractionService(BaseService):
    """Service that extracts facts from conversations using LLM.

    This service:
    1. Subscribes to assistant responses
    2. Extracts facts from user input using LLM
    3. Publishes facts to all/memory/facts
    """

    def __init__(self, config: ExtractionServiceConfig) -> None:
        """Initialize the extraction service.

        Args:
            config: Service configuration
        """
        super().__init__(config)
        self._extraction_config = config

        # LLM client
        self._ollama: Optional[OllamaClient] = None

        # Processing state
        self._processing_lock = threading.Lock()

        # Topics
        self._response_topic = Topics.EVENT_ASSISTANT_RESPONSE.topic
        self._facts_topic = Topics.MEMORY_FACTS.topic

    def _setup(self) -> None:
        """Set up LLM client and subscribe to topics."""
        if not self._extraction_config.extraction_enabled:
            self._logger.info("Extraction is disabled")
            return

        # Initialize Ollama client
        self._ollama = OllamaClient(
            base_url=self._extraction_config.ollama_host,
            timeout=self._extraction_config.ollama_timeout,
        )
        self._logger.info(f"Ollama client initialized: {self._extraction_config.ollama_host}")

        # Verify connection
        try:
            models_response = self._ollama.list_models()
            available = [m["name"] for m in models_response.get("models", [])]
            self._logger.info(f"Available Ollama models: {available}")
        except OllamaError as e:
            self._logger.warning(f"Could not verify Ollama connection: {e}")

        # Subscribe to responses
        self._subscribe(self._response_topic, self._on_response)

        self._logger.info(f"Using model: {self._extraction_config.ollama_model}")
        self._logger.info("Extraction service ready")

    def _cleanup(self) -> None:
        """Clean up resources."""
        if self._ollama:
            self._ollama.close()
            self._ollama = None

        self._logger.info("Extraction service cleaned up")

    def _on_response(self, topic: str, payload: bytes) -> None:
        """Handle incoming assistant response.

        Args:
            topic: MQTT topic
            payload: Message payload
        """
        try:
            message = AssistantResponseMessage.from_bytes(payload)

            # Need both user input and response
            if not message.input_text or not message.text:
                return

            # Process in background
            thread = threading.Thread(
                target=self._extract_facts,
                args=(message,),
                name="Extraction-Worker",
                daemon=True,
            )
            thread.start()

        except Exception as e:
            self._logger.error(f"Error handling response: {e}")

    def _extract_facts(self, message: AssistantResponseMessage) -> None:
        """Extract facts from conversation and publish.

        Args:
            message: The assistant response message
        """
        with self._processing_lock:
            if not self._running or self._ollama is None:
                return

            try:
                # Build prompt
                prompt = EXTRACTION_PROMPT.format(
                    user_input=message.input_text,
                    assistant_response=message.text,
                )

                # Call LLM
                response = self._ollama.generate(
                    model=self._extraction_config.ollama_model,
                    prompt=prompt,
                    options={
                        "temperature": 0.1,  # Low temperature for consistent extraction
                        "top_p": 0.9,
                    },
                )

                # Parse response (generate returns dict when not streaming)
                if isinstance(response, dict):
                    response_text = response.get("response", "")
                else:
                    response_text = ""
                facts = self._parse_facts(response_text)

                if not facts:
                    self._logger.debug("No facts extracted from conversation")
                    return

                # Filter by confidence
                facts = [
                    f
                    for f in facts
                    if f.get("confidence", 0) >= self._extraction_config.min_confidence
                ]

                if not facts:
                    self._logger.debug("All facts below confidence threshold")
                    return

                # Publish facts
                fact_message = FactMessage.create(
                    facts=facts,
                    source_event_id=message.message_id,
                )
                self._publish(self._facts_topic, fact_message.to_bytes())

                self._logger.info(f"Extracted {len(facts)} facts from conversation")
                for fact in facts:
                    self._logger.debug(
                        f"  {fact.get('subject')}.{fact.get('attribute')} = "
                        f"{fact.get('value')} (confidence: {fact.get('confidence')})"
                    )

            except OllamaError as e:
                self._logger.error(f"LLM request failed: {e}")
            except Exception as e:
                self._logger.error(f"Error extracting facts: {e}", exc_info=True)

    def _parse_facts(self, text: str) -> list[dict[str, Any]]:
        """Parse LLM response to extract facts.

        Args:
            text: LLM response text

        Returns:
            List of fact dictionaries
        """
        # Remove thinking tags if present
        text = re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL)
        text = text.strip()

        # Try to find JSON array in response
        # First, try to parse the whole thing
        try:
            facts = json.loads(text)
            if isinstance(facts, list):
                return self._validate_facts(facts)
        except json.JSONDecodeError:
            pass

        # Try to find JSON array with regex
        json_match = re.search(r"\[.*\]", text, re.DOTALL)
        if json_match:
            try:
                facts = json.loads(json_match.group())
                if isinstance(facts, list):
                    return self._validate_facts(facts)
            except json.JSONDecodeError:
                pass

        # Try to find JSON-like content between code blocks
        code_match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
        if code_match:
            try:
                facts = json.loads(code_match.group(1))
                if isinstance(facts, list):
                    return self._validate_facts(facts)
            except json.JSONDecodeError:
                pass

        self._logger.warning(f"Could not parse facts from response: {text[:200]}")
        return []

    def _validate_facts(self, facts: list[Any]) -> list[dict[str, Any]]:
        """Validate and clean extracted facts.

        Args:
            facts: Raw fact list from LLM

        Returns:
            Validated fact dictionaries
        """
        valid_facts = []

        for fact in facts:
            if not isinstance(fact, dict):
                continue

            # Required fields
            subject = fact.get("subject", "").strip()
            attribute = fact.get("attribute", "").strip()
            value = fact.get("value", "")

            if not subject or not attribute or value is None:
                continue

            # Convert value to string if needed
            if not isinstance(value, str):
                value = str(value)

            # Get confidence, default to 0.7
            confidence = fact.get("confidence", 0.7)
            if not isinstance(confidence, (int, float)):
                try:
                    confidence = float(confidence)
                except (ValueError, TypeError):
                    confidence = 0.7

            # Clamp confidence to valid range
            confidence = max(0.0, min(1.0, confidence))

            valid_facts.append(
                {
                    "subject": subject,
                    "attribute": attribute,
                    "value": value.strip() if isinstance(value, str) else value,
                    "confidence": confidence,
                    "category": fact.get("category", "learned"),
                    "source": "extraction",
                }
            )

        return valid_facts
